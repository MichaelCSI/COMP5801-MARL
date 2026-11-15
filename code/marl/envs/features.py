"""Data for the peer-SoC informativeness check (proposal feedback, after G1).

Question: can agent i predict its peers' SoC from its own observation? If a
regression of peer SoC on o_i fits closely, the peer-SoC message carries
little new information, which would predict (and explain) a null H1.

This module only builds the data; the regression and its report belong to
Part C. Typical use, once a no-message policy M0 exists::

    obs_seq, soc_seq = collect(env, controller, seed=VAL_SEED)
    data = peer_soc_dataset(obs_seq, soc_seq, env.observation_names)
    X, Y = data[i]["X"], data[i]["Y"]     # regress Y (T, n-1) on X (T, d)

``controller`` is any callable ``obs_list -> actions_list`` on raw
observations (R0 / R1 from ``marl.baselines``, or a wrapper around a trained
policy). Peers are in the channel's order (``marl.comms.channel.peer_indices``),
so column k of ``Y`` is the SoC that would arrive in message slot k.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from marl.comms.channel import peer_indices
from .citylearn_adapter import SOC_NAME


def collect(env, controller: Callable, seed: int,
            max_steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Run one episode and return (obs_seq (T, n, d), soc_seq (T, n)).

    Row t is the observation each agent acted on and the true SoC at that
    moment, i.e. what the message at step t would carry. All agents must
    have the same observation size (true for Phase 3.3: 30 each).
    """
    dims = set(getattr(env, "obs_dims", []) or [])
    if len(dims) > 1:
        raise ValueError(f"agents have different observation sizes {sorted(dims)}; "
                         "collect() stacks them into (T, n, d)")
    obs, _ = env.reset(seed=seed)
    obs_log, soc_log = [], []
    while True:
        obs_log.append(np.stack([np.asarray(o, dtype=np.float32) for o in obs]))
        soc_log.append(np.asarray(env.soc(), dtype=np.float32))
        obs, _, terminated, truncated, _ = env.step(controller(obs))
        if terminated or truncated or (max_steps is not None and len(obs_log) >= max_steps):
            break
    return np.stack(obs_log), np.stack(soc_log)


def peer_soc_dataset(obs_seq: np.ndarray, soc_seq: np.ndarray,
                     observation_names: Optional[Sequence[Sequence[str]]] = None,
                     drop_own_soc: bool = False) -> list:
    """Per-agent regression data: one dict per agent i.

    ``X`` (T, d): agent i's own observation; ``Y`` (T, n-1): its peers' SoC;
    ``peers``: their indices; ``feature_names``: names of X's columns if
    ``observation_names`` is given. With ``drop_own_soc`` the agent's own SoC
    column is removed from X (to see how much the shared signals alone explain).
    """
    obs_seq = np.asarray(obs_seq, dtype=np.float64)
    soc_seq = np.asarray(soc_seq, dtype=np.float64)
    if obs_seq.ndim != 3 or soc_seq.shape != obs_seq.shape[:2]:
        raise ValueError("expected obs_seq (T, n, d) and soc_seq (T, n)")
    n_agents = obs_seq.shape[1]
    data = []
    for agent in range(n_agents):
        names = list(observation_names[agent]) if observation_names is not None else None
        X = obs_seq[:, agent, :]
        if drop_own_soc:
            if names is None:
                raise ValueError("drop_own_soc needs observation_names")
            keep = [k for k, name in enumerate(names) if name != SOC_NAME]
            if len(keep) == len(names):
                raise ValueError(f"agent {agent} has no {SOC_NAME!r} feature to drop")
            X, names = X[:, keep], [names[k] for k in keep]
        peers = peer_indices(n_agents, agent)
        data.append({"agent": agent, "X": X, "Y": soc_seq[:, peers],
                     "peers": peers, "feature_names": names})
    return data
