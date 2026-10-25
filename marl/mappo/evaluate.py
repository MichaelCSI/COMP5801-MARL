"""Evaluation at a fixed delivery level (numpy, no torch).

``policy`` is anything with ``act(actor_inputs, deterministic) -> (actions (n, a), logp)``:
the trained on-policy actors (``OnPolicyRunner.actors()``), or the random
and scripted policies used in tests.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..comms.channel import DeliverySchedule, PeerSoCChannel
from .message_env import MessageEnv
from .normalize import Preprocessor


def evaluate(
    env,
    policy,
    preprocessor: Preprocessor,
    use_messages: bool,
    p: float = 1.0,
    n_replicates: int = 10,
    env_seed: int = 0,
    channel_seed: int = 0,
    force_mask: bool = False,
    deterministic: bool = True,
    critic: str = "central",
    on_episode_end: Optional[Callable] = None,
    max_steps: Optional[int] = None,
) -> list:
    """Run ``n_replicates`` episodes at delivery probability ``p``.

    Every replicate resets the env with the SAME ``env_seed`` (identical
    outages and window) and uses an independent channel stream
    (``channel_seed``, replicate), so replicates differ only in which messages
    drop. Normalizer statistics are frozen. Inputs are built by the same
    ``MessageEnv`` used in training.

    One record per replicate:
        actions   (T, n)      battery action sent (clipped to [-1, 1])
        battery_energy (T, n) what each battery actually drew/delivered at the
                              meter (kWh, + charging), from info["battery_energy"];
                              use this for the synchrony diagnostic. None if the
                              env does not report it
        soc       (T, n)      true SoC each agent acted in
        delivered (T, n, n)   delivery matrix, [i, j] = j -> i  (None for M0)
        rewards   (T,)
        infos     list of step info dicts
        return    float
        kpis      ``on_episode_end(env)`` (Andrea's KPIs), or None
    """
    n = env.n_agents
    records = []
    for r in range(n_replicates):
        channel = schedule = None
        if use_messages:
            channel = PeerSoCChannel(n, p, seed=np.random.SeedSequence([channel_seed, r]),
                                     force_mask=force_mask)
            schedule = DeliverySchedule("fixed", p=p)
        menv = MessageEnv(env, use_messages, channel, schedule, preprocessor, critic,
                          update_normalizers=False, auto_reset=False)
        obs, _ = menv.reset(seed=env_seed)
        acts, socs, dels, rews, infos, energy = [], [], [], [], [], []
        t = 0
        while True:
            a, _ = policy.act(obs, deterministic=deterministic)
            a = np.clip(np.asarray(a, dtype=np.float32).reshape(n, -1), -1.0, 1.0)
            socs.append(menv.last_soc.copy())
            dels.append(menv.last_delivered.copy() if use_messages else None)
            obs, _, rew, done, _, info = menv.step(a)
            acts.append(a[:, 0])
            rews.append(float(rew[0, 0]))
            infos.append(info)
            energy.append(info.get("battery_energy"))
            t += 1
            if done or (max_steps is not None and t >= max_steps):
                break
        records.append({
            "p": p if use_messages else None,
            "force_mask": force_mask,
            "replicate": r,
            "actions": np.stack(acts),
            "battery_energy": (np.stack([np.asarray(e, np.float32) for e in energy])
                               if all(e is not None for e in energy) else None),
            "soc": np.stack(socs),
            "delivered": np.stack(dels) if use_messages else None,
            "rewards": np.asarray(rews, dtype=np.float32),
            "infos": infos,
            "return": float(np.sum(rews)),
            "kpis": on_episode_end(env) if on_episode_end else None,
        })
    return records


class RandomPolicy:
    """Uniform actions in [-1, 1]. Used in tests and smoke runs."""

    def __init__(self, n_agents: int, action_dims, seed=None):
        self.n_agents, self.action_dims = n_agents, list(action_dims)
        self.rng = np.random.default_rng(seed)

    def act(self, actor_inputs, deterministic=False):
        a = self.rng.uniform(-1, 1, (self.n_agents, max(self.action_dims))).astype(np.float32)
        return a, np.zeros(self.n_agents, np.float32)
