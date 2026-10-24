"""Thin CityLearn -> contract adapter (TEMPORARY, Member B).

This exists so the MAPPO trainer can be run on real CityLearn before
Andrea's wrapper lands. It is deliberately minimal:

* decentralized (``central_agent=False``), one agent per building;
* only the electrical-storage action is active (cooling / DHW disabled);
* ``soc()`` reads each building's ``electrical_storage_soc`` observation;
* ``global_state()`` = shared observations once + every building's local
  (non-shared) observations, i.e. the true state with nothing dropped;
* the shared reward is a PLACEHOLDER (sum of CityLearn's per-building
  rewards) unless ``reward_fn`` is given. The proposal's reward
  (normalized cost + squared district load) belongs to Andrea's wrapper.

When Andrea's wrapper is ready it only has to satisfy
``marl.envs.base.check_env_contract``; nothing in the trainer changes.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

DATASET_2023_PHASE_3_3 = "citylearn_challenge_2023_phase_3_3"
SOC_NAME = "electrical_storage_soc"


class CityLearnAdapter:
    def __init__(
        self,
        schema: str = DATASET_2023_PHASE_3_3,
        reward_fn: Optional[Callable] = None,
        active_actions: Optional[list] = None,
        **citylearn_kwargs,
    ) -> None:
        from citylearn.citylearn import CityLearnEnv  # imported lazily

        kwargs = dict(central_agent=False)
        kwargs["active_actions"] = active_actions or ["electrical_storage"]
        kwargs.update(citylearn_kwargs)
        self.env = CityLearnEnv(schema, **kwargs)
        self.reward_fn = reward_fn

        self.n_agents = len(self.env.action_space)
        self.observation_names = [list(n) for n in self.env.observation_names]
        self.action_names = [list(n) for n in self.env.action_names]
        for i, names in enumerate(self.action_names):
            if names != ["electrical_storage"]:
                raise RuntimeError(
                    f"building {i} has actions {names}; expected only electrical_storage"
                )
        self.obs_dims = [len(n) for n in self.observation_names]
        self.action_dims = [int(np.prod(s.shape)) for s in self.env.action_space]

        self._soc_idx = []
        for i, names in enumerate(self.observation_names):
            if SOC_NAME not in names:
                raise RuntimeError(f"building {i} does not observe {SOC_NAME}")
            self._soc_idx.append(names.index(SOC_NAME))

        shared = set(getattr(self.env, "shared_observations", []) or [])
        names0 = self.observation_names[0]
        self._shared_idx0 = [k for k, nm in enumerate(names0) if nm in shared]
        self._local_idx = [[k for k, nm in enumerate(names) if nm not in shared]
                           for names in self.observation_names]
        self.state_dim = len(self._shared_idx0) + sum(len(ix) for ix in self._local_idx)
        self._obs = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = [np.asarray(o, dtype=np.float32) for o in obs]
        return self._obs, dict(info or {})

    def step(self, actions):
        acts = [np.clip(np.asarray(a, dtype=np.float32).reshape(-1), -1.0, 1.0).tolist()
                for a in actions]
        obs, rewards, terminated, truncated, info = self.env.step(acts)
        self._obs = [np.asarray(o, dtype=np.float32) for o in obs]
        if self.reward_fn is not None:
            reward = float(self.reward_fn(self.env, rewards))
        else:
            reward = float(np.sum(rewards))
        info = dict(info or {})
        info["building_rewards"] = np.asarray(rewards, dtype=np.float32)
        return self._obs, reward, bool(terminated), bool(truncated), info

    def soc(self):
        return np.array([self._obs[i][k] for i, k in enumerate(self._soc_idx)],
                        dtype=np.float32)

    def global_state(self):
        parts = [self._obs[0][self._shared_idx0]]
        parts += [self._obs[i][ix] for i, ix in enumerate(self._local_idx)]
        return np.concatenate(parts).astype(np.float32)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
