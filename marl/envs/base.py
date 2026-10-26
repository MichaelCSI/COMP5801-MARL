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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EnvContractError(message)


def check_env_contract(env, n_steps: int = 5, seed: int = 0) -> None:
    """Raise ``EnvContractError`` if ``env`` does not follow the contract.

    Runs ``n_steps`` steps with zero actions; leaves the env mid-episode.
    """
    for attr in ("n_agents", "obs_dims", "action_dims", "state_dim"):
        _require(hasattr(env, attr), f"env is missing attribute {attr!r}")

    n_agents = env.n_agents
    _require(isinstance(n_agents, int) and n_agents >= 2, "n_agents must be an int >= 2")
    _require(len(env.obs_dims) == n_agents, "obs_dims must have one entry per agent")
    _require(len(env.action_dims) == n_agents, "action_dims must have one entry per agent")

    reset_result = env.reset(seed=seed)
    _require(isinstance(reset_result, tuple) and len(reset_result) == 2,
             "reset must return (obs, info)")
    obs, info = reset_result
    _require(isinstance(info, dict), "reset info must be a dict")

    def check_obs(obs_list, where):
        _require(len(obs_list) == n_agents,
                 f"{where}: expected {n_agents} observations, got {len(obs_list)}")
        for agent, agent_obs in enumerate(obs_list):
            agent_obs = np.asarray(agent_obs, dtype=np.float64)
            _require(agent_obs.ndim == 1, f"{where}: obs[{agent}] must be 1-D")
            _require(agent_obs.shape[0] == env.obs_dims[agent],
                     f"{where}: obs[{agent}] has length {agent_obs.shape[0]}, "
                     f"obs_dims says {env.obs_dims[agent]}")
            _require(np.all(np.isfinite(agent_obs)),
                     f"{where}: obs[{agent}] has non-finite values")

    def check_soc_and_state(where):
        soc = np.asarray(env.soc(), dtype=np.float64)
        _require(soc.shape == (n_agents,),
                 f"{where}: soc() must have shape ({n_agents},), got {soc.shape}")
        # small tolerance for float rounding at the bounds
        _require(np.all((soc >= -1e-6) & (soc <= 1 + 1e-6)), f"{where}: soc() outside [0, 1]: {soc}")

        state = np.asarray(env.global_state(), dtype=np.float64)
        _require(state.shape == (env.state_dim,),
                 f"{where}: global_state() shape {state.shape}, state_dim says {env.state_dim}")
        _require(np.all(np.isfinite(state)), f"{where}: global_state() has non-finite values")

    check_obs(obs, "reset")
    check_soc_and_state("reset")

    for step in range(n_steps):
        zero_actions = [np.zeros(dim, dtype=np.float32) for dim in env.action_dims]
        step_result = env.step(zero_actions)
        _require(isinstance(step_result, tuple) and len(step_result) == 5,
                 "step must return (obs, reward, terminated, truncated, info)")

        obs, reward, terminated, truncated, info = step_result
        _require(np.ndim(reward) == 0,
                 f"step reward must be ONE shared scalar, got shape {np.shape(reward)}")
        _require(np.isfinite(float(reward)), "step reward is not finite")
        _require(np.ndim(terminated) == 0 and np.ndim(truncated) == 0,
                 "terminated/truncated must be single bools")
        _require(isinstance(info, dict), "step info must be a dict")
        check_obs(obs, f"step {step}")

        if terminated or truncated:
            break
        check_soc_and_state(f"step {step}")
