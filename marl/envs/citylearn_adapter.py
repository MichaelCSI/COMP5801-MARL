"""Thin CityLearn -> contract adapter (TEMPORARY, Member B).

This exists so the MAPPO trainer can be run on real CityLearn before
Andrea's wrapper lands. It is deliberately minimal:

* decentralized (``central_agent=False``), one agent per building;
* only the electrical-storage action is active (cooling / DHW disabled);
* the proposal's chronological split (Section 6): ``split="train"`` gives
  random contiguous windows inside days 1-62 (steps 0-1487);
  ``"validation"`` (1488-1847) and ``"test"`` (1848-2207) are one episode
  covering the whole block;
* every episode end is reported as ``truncated`` (a time limit), never
  ``terminated``. CityLearn 3.0.2 always reports the end of an episode as
  ``terminated`` and hard-codes ``truncated=False``, but nothing in this task
  is a true terminal state;
* ``info["battery_energy"]`` is each battery's actual electricity use at the
  meter in the step just taken (kWh, + charging). It is what the battery
  really did after capacity, SoC and outage limits, as opposed to the action;
* ``soc()`` reads each building's ``electrical_storage_soc`` observation;
* ``global_state()`` = shared observations once + every building's local
  (non-shared) observations, i.e. the true state with nothing dropped;
* the shared reward is a PLACEHOLDER (sum of CityLearn's per-building
  rewards) unless ``reward_fn`` is given. The proposal's reward
  (normalized cost + squared district load) belongs to Andrea's wrapper.

Not handled here (Andrea's wrapper): outage seeding. In the Phase 3.3 schema
every building's stochastic outage model has a fixed seed (57841) that
``reset(seed)`` does not reach, so outages are identical in every episode and
absent from windows shorter than ~43 days.

When Andrea's wrapper is ready it only has to satisfy
``marl.envs.base.check_env_contract``; nothing in the trainer changes.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

DATASET_2023_PHASE_3_3 = "citylearn_challenge_2023_phase_3_3"
SOC_NAME = "electrical_storage_soc"

#: Chronological split from the proposal (inclusive time-step ranges).
SPLITS = {
    "train": (0, 1487),          # days 1-62
    "validation": (1488, 1847),  # days 63-77
    "test": (1848, 2207),        # days 78-92
}

_SEED_RANGE = 100_000


def citylearn_seed(seed: Optional[int]) -> Optional[int]:
    """Map a reset seed into the range CityLearn can use.

    With random episode splits CityLearn picks the window with
    ``np.random.RandomState(random_seed * (episode + 1))``, which fails once
    the product reaches 2**32. Seeds in [1, 100000] stay valid for over 40,000
    episodes. 0 is avoided because it would select the same window every time.
    """
    if seed is None:
        return None
    return 1 + int(seed) % _SEED_RANGE


def split_kwargs(split: str, episode_time_steps: int) -> dict:
    """CityLearnEnv keyword arguments for one block of the split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
    start, end = SPLITS[split]
    if split == "train":
        if not 1 < episode_time_steps <= end - start + 1:
            raise ValueError(f"episode_time_steps must be in (1, {end - start + 1}]")
        # every time step in the block is a candidate start; picked at random
        return dict(simulation_start_time_step=start, simulation_end_time_step=end,
                    episode_time_steps=int(episode_time_steps),
                    rolling_episode_split=True, random_episode_split=True)
    return dict(simulation_start_time_step=start, simulation_end_time_step=end,
                episode_time_steps=end - start + 1,
                rolling_episode_split=False, random_episode_split=False)


class CityLearnAdapter:
    def __init__(
        self,
        schema: str = DATASET_2023_PHASE_3_3,
        split: str = "train",
        episode_time_steps: int = 168,
        reward_fn: Optional[Callable] = None,
        active_actions: Optional[list] = None,
        **citylearn_kwargs,
    ) -> None:
        from citylearn.citylearn import CityLearnEnv  # imported lazily

        self.split = split
        kwargs = dict(central_agent=False)
        kwargs["active_actions"] = active_actions or ["electrical_storage"]
        kwargs.update(split_kwargs(split, episode_time_steps))
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
        obs, info = self.env.reset(seed=citylearn_seed(seed), options=options)
        self._obs = [np.asarray(o, dtype=np.float32) for o in obs]
        info = dict(info or {})
        info["episode_start_time_step"] = self.episode_window()[0]
        return self._obs, info

    def episode_window(self):
        """(start, end) time steps of the current episode in the dataset."""
        tr = self.env.episode_tracker
        return int(tr.episode_start_time_step), int(tr.episode_end_time_step)

    def battery_energy(self) -> np.ndarray:
        """Each battery's electricity use in the step just taken (kWh, + charging)."""
        # The env clock always advances after a step; on the terminal step the
        # buildings' clocks do not (CityLearn 3.0.2 runtime.next_time_step),
        # so the step just taken is env.time_step - 1, not building.time_step - 1.
        t = max(0, int(self.env.time_step) - 1)
        return np.asarray([float(b.electrical_storage.electricity_consumption[t])
                           for b in self.env.buildings], dtype=np.float32)

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
        info["battery_energy"] = self.battery_energy()
        # The episode ends because the window ran out, never because the task did.
        ended = bool(terminated or truncated)
        return self._obs, reward, False, ended, info

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
