"""Contract env + message channel -> on-policy's per-agent input format (numpy).

This is where the project's contribution meets the reference learner. Per
decision step, in the proposal's order (Section 1):

    1. read every building's true SoC                  env.soc()
    2. drop each directed link w.p. 1 - p               channel.transmit
    3. actor input  o~_i = [norm(o_i), m_-i]            -> obs[i]
    4. critic input = norm(true global state)           -> share_obs[i]
       (IPPO: the agent's own actor input instead)

The critic input never contains dropped messages. Messages stay raw
([SoC, 1] / [0, 0]); only the env's local features are normalized.

At each episode start the delivery schedule picks p (fixed for M1, random
for M1-RND), and the reset seed comes from a stream that depends only on the
run seed, so outages and windows are paired across arms.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..comms.channel import DeliverySchedule, PeerSoCChannel, message_dim
from .normalize import Preprocessor
from .spaces import Box


def build_actor_inputs(local_obs, messages: Optional[np.ndarray]):
    """o~_i = [o_i, m_-i]; just o_i when messages is None."""
    if messages is None:
        return [np.asarray(o, dtype=np.float32) for o in local_obs]
    return [np.concatenate([np.asarray(o, dtype=np.float32), messages[i]])
            for i, o in enumerate(local_obs)]


class MessageEnv:
    def __init__(
        self,
        env,
        use_messages: bool,
        channel: Optional[PeerSoCChannel] = None,
        schedule: Optional[DeliverySchedule] = None,
        preprocessor: Optional[Preprocessor] = None,
        critic: str = "central",
        env_seed=None,
        update_normalizers: bool = True,
        auto_reset: bool = True,
    ) -> None:
        if use_messages and (channel is None or schedule is None):
            raise ValueError("message arms need a channel and a schedule")
        if critic not in ("central", "local"):
            raise ValueError("critic must be 'central' or 'local'")
        self.env = env
        self.n_agents = env.n_agents
        self.use_messages = use_messages
        self.channel, self.schedule = channel, schedule
        self.pre = preprocessor or Preprocessor(env.obs_dims, env.state_dim, enabled=False)
        self.critic = critic
        self.env_seed_rng = np.random.default_rng(env_seed)
        self.update_normalizers = update_normalizers
        self.auto_reset = auto_reset

        msg = message_dim(self.n_agents) if use_messages else 0
        self.actor_dims = [d + msg for d in env.obs_dims]
        self.critic_dims = ([env.state_dim] * self.n_agents if critic == "central"
                            else list(self.actor_dims))
        self.observation_space = [Box(-np.inf, np.inf, (d,)) for d in self.actor_dims]
        self.share_observation_space = [Box(-np.inf, np.inf, (d,)) for d in self.critic_dims]
        self.action_space = [Box(-1.0, 1.0, (a,)) for a in env.action_dims]

        self.reset_seeds: list = []
        self.episodes: list = []
        self.last_soc: Optional[np.ndarray] = None
        self.last_delivered: Optional[np.ndarray] = None
        self.p: Optional[float] = None

    # ------------------------------------------------------------------
    def _inputs(self, obs):
        soc = np.asarray(self.env.soc(), dtype=np.float32)
        self.last_soc = soc
        local = self.pre.local(obs, self.update_normalizers)
        msgs = None
        if self.use_messages:
            msgs = self.channel.transmit(soc)
            self.last_delivered = self.channel.last_delivered.copy()
            self._ep["delivered"] += int(self.last_delivered.sum())
            self._ep["links"] += self.n_agents * (self.n_agents - 1)
        actor_in = build_actor_inputs(local, msgs)
        if self.critic == "central":
            state = self.pre.state(self.env.global_state(), self.update_normalizers)
            share = [state] * self.n_agents
        else:
            share = actor_in
        return actor_in, share

    def reset(self, seed: Optional[int] = None):
        """Start an episode; returns (obs list, share_obs list)."""
        if self.use_messages:
            self.p = self.schedule.sample()
            self.channel.set_p(self.p)
        if seed is None:
            seed = int(self.env_seed_rng.integers(2 ** 31 - 1))
        self.reset_seeds.append(int(seed))
        obs, _ = self.env.reset(seed=int(seed))
        self._ep = {"return": 0.0, "length": 0, "p": self.p, "delivered": 0, "links": 0}
        return self._inputs(obs)

    def step(self, actions):
        """actions: (n, a) array or list. Returns
        (obs, share_obs, rewards (n, 1), done, truncated, info).

        With ``auto_reset`` the returned obs/share_obs already belong to the
        next episode when ``done`` (on-policy's convention). The finished
        episode's summary is in ``info["episode"]``.
        """
        acts = [np.clip(np.asarray(a, dtype=np.float32).reshape(-1), -1.0, 1.0)
                for a in actions]
        obs, reward, terminated, truncated, info = self.env.step(acts)
        info = dict(info)
        self._ep["return"] += float(reward)
        self._ep["length"] += 1
        done = bool(terminated or truncated)
        rewards = np.full((self.n_agents, 1), float(reward), dtype=np.float32)
        if done:
            ep = dict(self._ep)
            ep["delivery_rate"] = ep["delivered"] / ep["links"] if ep["links"] else None
            self.episodes.append(ep)
            info["episode"] = ep
            if self.auto_reset:
                actor_in, share = self.reset()
                return actor_in, share, rewards, done, bool(truncated and not terminated), info
            return None, None, rewards, done, bool(truncated and not terminated), info
        actor_in, share = self._inputs(obs)
        return actor_in, share, rewards, done, False, info


def make_message_env(env, cfg, preprocessor: Optional[Preprocessor] = None,
                     update_normalizers: bool = True) -> MessageEnv:
    """MessageEnv for training an arm described by ``MAPPOConfig`` ``cfg``."""
    from .config import STREAM_CHANNEL, STREAM_ENV, STREAM_SCHEDULE, stream

    channel = schedule = None
    if cfg.use_messages:
        channel = PeerSoCChannel(env.n_agents, 1.0, seed=stream(cfg.seed, STREAM_CHANNEL),
                                 force_mask=cfg.force_mask_train)
        if cfg.train_schedule == "fixed":
            schedule = DeliverySchedule("fixed", p=cfg.train_p)
        else:
            schedule = DeliverySchedule("random", choices=tuple(cfg.rnd_p_choices),
                                        seed=stream(cfg.seed, STREAM_SCHEDULE))
    pre = preprocessor or Preprocessor(env.obs_dims, env.state_dim, cfg.normalize_obs,
                                       cfg.obs_clip)
    return MessageEnv(env, cfg.use_messages, channel, schedule, pre, cfg.critic,
                      stream(cfg.seed, STREAM_ENV), update_normalizers)
