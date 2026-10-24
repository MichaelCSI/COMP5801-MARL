"""Environment interaction for training and evaluation (numpy, no torch).

The policy is anything with

    policy.act(actor_inputs, deterministic) -> (actions (n, a), logp (n,))
    policy.value(critic_inputs)             -> values (K,)   [training only]

so this loop is shared by MAPPO, IPPO and the random / scripted policies
used in tests and by reference controllers.

Per step, the order is the proposal's (Section 1):
    1. every agent's true SoC is read from the env       (env.soc())
    2. the channel drops each directed link w.p. 1-p     (channel.transmit)
    3. agent i acts on [o_i, m_-i]                       (policy.act)
    4. the critic sees the TRUE global state             (env.global_state())
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..comms.channel import DeliverySchedule, PeerSoCChannel
from .buffer import RolloutBuffer
from .normalize import Preprocessor


def build_actor_inputs(local_obs, messages: Optional[np.ndarray]):
    """o~_i = [o_i, m_-i]; just o_i when messages is None."""
    if messages is None:
        return [np.asarray(o, dtype=np.float32) for o in local_obs]
    return [np.concatenate([np.asarray(o, dtype=np.float32), messages[i]])
            for i, o in enumerate(local_obs)]


class RolloutWorker:
    def __init__(
        self,
        env,
        policy,
        preprocessor: Preprocessor,
        use_messages: bool,
        channel: Optional[PeerSoCChannel],
        schedule: Optional[DeliverySchedule],
        critic: str = "central",
        env_seed: Optional[np.random.SeedSequence] = None,
        bootstrap_on_truncation: bool = True,
    ) -> None:
        if use_messages and (channel is None or schedule is None):
            raise ValueError("message arms need a channel and a schedule")
        self.env, self.policy, self.pre = env, policy, preprocessor
        self.use_messages = use_messages
        self.channel, self.schedule = channel, schedule
        self.critic = critic
        self.env_seed_rng = np.random.default_rng(env_seed)
        self.bootstrap_on_truncation = bootstrap_on_truncation
        self.episodes: list = []         # finished episode summaries
        self.reset_seeds: list = []      # every env reset seed, for pairing checks
        self.total_steps = 0
        self._obs = None

    # -- helpers --------------------------------------------------------------
    def _start_episode(self) -> None:
        p = None
        if self.use_messages:
            p = self.schedule.sample()
            self.channel.set_p(p)
        seed = int(self.env_seed_rng.integers(2 ** 31 - 1))
        self.reset_seeds.append(seed)
        self._obs, _ = self.env.reset(seed=seed)
        self._ep = {"return": 0.0, "length": 0, "p": p, "delivered": 0, "links": 0}

    def _inputs(self, obs, update: bool):
        local = self.pre.local(obs, update)
        msgs = None
        if self.use_messages:
            msgs = self.channel.transmit(self.env.soc())
            d = self.channel.last_delivered
            self._ep["delivered"] += int(d.sum())
            self._ep["links"] += d.shape[0] * (d.shape[0] - 1)
        actor_in = build_actor_inputs(local, msgs)
        if self.critic == "central":
            critic_in = [self.pre.state(self.env.global_state(), update)]
        else:
            critic_in = actor_in
        return actor_in, critic_in

    def _critic_only_inputs(self, obs):
        """Critic input for a bootstrap state without consuming channel draws."""
        if self.critic == "central":
            return [self.pre.state(self.env.global_state(), False)]
        local = self.pre.local(obs, False)
        msgs = None
        if self.use_messages:
            soc = self.env.soc()
            msgs = self.channel.encode(soc, self.channel.sample_delivery())
        return build_actor_inputs(local, msgs)

    # -- training rollout -------------------------------------------------------
    def collect(self, buffer: RolloutBuffer, update_normalizers: bool = True) -> dict:
        buffer.reset()
        if self._obs is None:
            self._start_episode()
        n_eps_before = len(self.episodes)
        while not buffer.full:
            actor_in, critic_in = self._inputs(self._obs, update_normalizers)
            actions, logp = self.policy.act(actor_in, deterministic=False)
            values = self.policy.value(critic_in)
            obs, reward, terminated, truncated, info = self.env.step(list(actions))
            done = bool(terminated or truncated)
            boot = np.zeros_like(values)
            if done and truncated and not terminated and self.bootstrap_on_truncation:
                boot = self.policy.value(self._critic_only_inputs(obs))
            buffer.add(actor_in, critic_in, actions, logp, reward, done, values, boot)
            self.total_steps += 1
            self._ep["return"] += float(reward)
            self._ep["length"] += 1
            self._obs = obs
            if done:
                self.episodes.append(dict(self._ep))
                self._start_episode()
        # value of the state after the last stored step
        buffer.last_value = self.policy.value(self._critic_only_inputs(self._obs))
        return {"episodes": self.episodes[n_eps_before:]}


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
    on_episode_end: Optional[Callable] = None,
    max_steps: Optional[int] = None,
) -> list:
    """Run ``n_replicates`` evaluation episodes at delivery probability ``p``.

    Every replicate resets the env with the SAME ``env_seed`` (identical
    outages and window) and uses an independent channel stream
    (``channel_seed``, replicate index), so replicates differ only in which
    messages drop. Normalizer statistics are frozen.

    Returns one record per replicate:
        actions   (T, n)      battery action actually sent (clipped to [-1, 1])
        soc       (T, n)      true SoC each agent acted in
        delivered (T, n, n)   delivery matrix, [i, j] = j -> i (None for M0)
        rewards   (T,)
        infos     list of step info dicts
        return    float
        kpis      whatever ``on_episode_end(env)`` returns (Andrea's KPIs)
    """
    records = []
    n = env.n_agents
    for r in range(n_replicates):
        channel = None
        if use_messages:
            channel = PeerSoCChannel(n, p, seed=np.random.SeedSequence([channel_seed, r]),
                                     force_mask=force_mask)
        obs, _ = env.reset(seed=env_seed)
        acts, socs, dels, rews, infos = [], [], [], [], []
        t = 0
        while True:
            soc = env.soc()
            local = preprocessor.local(obs, update=False)
            msgs = channel.transmit(soc) if use_messages else None
            a, _ = policy.act(build_actor_inputs(local, msgs), deterministic=deterministic)
            a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(list(a))
            acts.append(a.reshape(n, -1)[:, 0])
            socs.append(np.asarray(soc, dtype=np.float32))
            dels.append(channel.last_delivered.copy() if use_messages else None)
            rews.append(float(reward))
            infos.append(info)
            t += 1
            if terminated or truncated or (max_steps is not None and t >= max_steps):
                break
        records.append({
            "p": p if use_messages else None,
            "force_mask": force_mask,
            "replicate": r,
            "actions": np.stack(acts),
            "soc": np.stack(socs),
            "delivered": np.stack(dels) if use_messages else None,
            "rewards": np.asarray(rews, dtype=np.float32),
            "infos": infos,
            "return": float(np.sum(rews)),
            "kpis": on_episode_end(env) if on_episode_end else None,
        })
    return records


class RandomPolicy:
    """Uniform actions in [-1, 1]; zero value. Used in tests and smoke runs."""

    def __init__(self, n_agents: int, action_dims, seed=None, n_critics: int = 1):
        self.n_agents, self.action_dims = n_agents, list(action_dims)
        self.rng = np.random.default_rng(seed)
        self.n_critics = n_critics

    def act(self, actor_inputs, deterministic=False):
        a = self.rng.uniform(-1, 1, (self.n_agents, max(self.action_dims))).astype(np.float32)
        return a, np.full(self.n_agents, -np.log(2.0) * max(self.action_dims), np.float32)

    def value(self, critic_inputs):
        return np.zeros(self.n_critics, np.float32)
