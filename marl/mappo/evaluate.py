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
    n_agents = env.n_agents
    records = []
    for replicate in range(n_replicates):
        # fresh channel per replicate: same p, independent drops
        channel = schedule = None
        if use_messages:
            channel = PeerSoCChannel(n_agents, p,
                                     seed=np.random.SeedSequence([channel_seed, replicate]),
                                     force_mask=force_mask)
            schedule = DeliverySchedule("fixed", p=p)
        menv = MessageEnv(env, use_messages, channel, schedule, preprocessor, critic,
                          update_normalizers=False, auto_reset=False)

        # same env seed every replicate -> identical outages and window
        actor_inputs, _ = menv.reset(seed=env_seed)

        actions_log, soc_log, delivered_log = [], [], []
        rewards_log, infos_log, energy_log = [], [], []
        n_steps = 0
        while True:
            actions, _ = policy.act(actor_inputs, deterministic=deterministic)
            actions = np.clip(np.asarray(actions, dtype=np.float32).reshape(n_agents, -1),
                              -1.0, 1.0)
            # record the state the agents acted in, before stepping
            soc_log.append(menv.last_soc.copy())
            delivered_log.append(menv.last_delivered.copy() if use_messages else None)

            actor_inputs, _, rewards, done, _, info = menv.step(actions)
            actions_log.append(actions[:, 0])
            rewards_log.append(float(rewards[0, 0]))   # shared, so agent 0's copy
            infos_log.append(info)
            energy_log.append(info.get("battery_energy"))

            n_steps += 1
            if done or (max_steps is not None and n_steps >= max_steps):
                break

        has_energy = all(energy is not None for energy in energy_log)
        records.append({
            "p": p if use_messages else None,
            "force_mask": force_mask,
            "replicate": replicate,
            "actions": np.stack(actions_log),
            "battery_energy": (np.stack([np.asarray(e, np.float32) for e in energy_log])
                               if has_energy else None),
            "soc": np.stack(soc_log),
            "delivered": np.stack(delivered_log) if use_messages else None,
            "rewards": np.asarray(rewards_log, dtype=np.float32),
            "infos": infos_log,
            "return": float(np.sum(rewards_log)),
            "kpis": on_episode_end(env) if on_episode_end else None,
        })
    return records


class RandomPolicy:
    """Uniform actions in [-1, 1]. Used in tests and smoke runs."""

    def __init__(self, n_agents: int, action_dims, seed=None):
        self.n_agents, self.action_dims = n_agents, list(action_dims)
        self.rng = np.random.default_rng(seed)

    def act(self, actor_inputs, deterministic=False):
        actions = self.rng.uniform(-1, 1, (self.n_agents, max(self.action_dims)))
        return actions.astype(np.float32), np.zeros(self.n_agents, np.float32)
