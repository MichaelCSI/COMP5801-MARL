"""The environment contract the MAPPO trainer depends on.

Andrea's CityLearn wrapper (reward, KPIs, split, outage seeding) only has to
expose this interface for the trainer, the message channel and Vishal's
runner to work unchanged. ``check_env_contract`` verifies it.

    env.n_agents                 int, number of buildings (6)
    env.obs_dims                 list[int], local observation length per agent
    env.action_dims              list[int], action length per agent (1)
    env.state_dim                int, length of the critic's true global state
    env.reset(seed=None, options=None) -> (obs, info)
        obs: list of n 1-D float arrays (agent i's LOCAL observation o_i)
    env.step(actions) -> (obs, reward, terminated, truncated, info)
        actions: list of n arrays, each in [-1, 1]
        reward:  ONE float, the shared district reward r_t
        terminated / truncated: bool, end of episode
    env.soc() -> np.ndarray (n,)   true current battery SoC per building,
                                   used as the message payload
    env.global_state() -> np.ndarray (state_dim,)
                                   true global state for the centralized critic
                                   (never contains dropped messages)

``soc()`` and ``global_state()`` must describe the state the agents are
about to act in, i.e. the state matching the most recent ``obs``.

Episode ends: a training window running out is a time limit, so report it as
``truncated=True`` (``terminated`` only for a true terminal state, which this
task does not have). The trainer then treats it as a cut-off, not an ending.

Optional ``info["battery_energy"]``: (n,) array, each battery's actual
electricity use at the meter in the step just taken (kWh, + charging). It is
recorded by evaluation and is the signal for the synchrony diagnostic.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MultiAgentEnv(Protocol):
    n_agents: int
    obs_dims: list
    action_dims: list
    state_dim: int

    def reset(self, seed: Any = None, options: Any = None): ...

    def step(self, actions): ...

    def soc(self) -> np.ndarray: ...

    def global_state(self) -> np.ndarray: ...


class EnvContractError(AssertionError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise EnvContractError(msg)


def check_env_contract(env, n_steps: int = 5, seed: int = 0) -> None:
    """Raise ``EnvContractError`` if ``env`` does not follow the contract.

    Runs ``n_steps`` steps with zero actions; leaves the env mid-episode.
    """
    for attr in ("n_agents", "obs_dims", "action_dims", "state_dim"):
        _check(hasattr(env, attr), f"env is missing attribute {attr!r}")
    n = env.n_agents
    _check(isinstance(n, int) and n >= 2, "n_agents must be an int >= 2")
    _check(len(env.obs_dims) == n, "obs_dims must have one entry per agent")
    _check(len(env.action_dims) == n, "action_dims must have one entry per agent")

    out = env.reset(seed=seed)
    _check(isinstance(out, tuple) and len(out) == 2, "reset must return (obs, info)")
    obs, info = out
    _check(isinstance(info, dict), "reset info must be a dict")

    def check_obs(o, where):
        _check(len(o) == n, f"{where}: expected {n} observations, got {len(o)}")
        for i, oi in enumerate(o):
            a = np.asarray(oi, dtype=np.float64)
            _check(a.ndim == 1, f"{where}: obs[{i}] must be 1-D")
            _check(a.shape[0] == env.obs_dims[i],
                   f"{where}: obs[{i}] has length {a.shape[0]}, obs_dims says {env.obs_dims[i]}")
            _check(np.all(np.isfinite(a)), f"{where}: obs[{i}] has non-finite values")

    def check_soc_state(where):
        s = np.asarray(env.soc(), dtype=np.float64)
        _check(s.shape == (n,), f"{where}: soc() must have shape ({n},), got {s.shape}")
        _check(np.all((s >= -1e-6) & (s <= 1 + 1e-6)), f"{where}: soc() outside [0, 1]: {s}")
        g = np.asarray(env.global_state(), dtype=np.float64)
        _check(g.shape == (env.state_dim,),
               f"{where}: global_state() shape {g.shape}, state_dim says {env.state_dim}")
        _check(np.all(np.isfinite(g)), f"{where}: global_state() has non-finite values")

    check_obs(obs, "reset")
    check_soc_state("reset")

    for t in range(n_steps):
        actions = [np.zeros(d, dtype=np.float32) for d in env.action_dims]
        res = env.step(actions)
        _check(isinstance(res, tuple) and len(res) == 5,
               "step must return (obs, reward, terminated, truncated, info)")
        obs, reward, terminated, truncated, info = res
        _check(np.ndim(reward) == 0,
               f"step reward must be ONE shared scalar, got shape {np.shape(reward)}")
        _check(np.isfinite(float(reward)), "step reward is not finite")
        _check(isinstance(bool(terminated), bool) and isinstance(bool(truncated), bool),
               "terminated/truncated must be bools")
        _check(isinstance(info, dict), "step info must be a dict")
        check_obs(obs, f"step {t}")
        if terminated or truncated:
            break
        check_soc_state(f"step {t}")
