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

M-FULL (``full_peer_info``) replaces the lossy messages with every peer's
true [SoC, normalized net load], never dropped: the same 2(n-1) slots in the
same peer order, an upper bound on what a SoC/load message could add.

At each episode start the delivery schedule picks p (fixed for M1, random
for M1-RND), and the reset seed comes from a stream that depends only on the
run seed, so outages and windows are paired across arms.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..comms.channel import DeliverySchedule, PeerSoCChannel, message_dim, peer_indices
from .normalize import Preprocessor
from .spaces import Box


def build_actor_inputs(local_obs, messages: Optional[np.ndarray]):
    """o~_i = [o_i, m_-i]; just o_i when messages is None."""
    if messages is None:
        return [np.asarray(agent_obs, dtype=np.float32) for agent_obs in local_obs]
    return [np.concatenate([np.asarray(agent_obs, dtype=np.float32), messages[agent]])
            for agent, agent_obs in enumerate(local_obs)]


def full_peer_block(soc: np.ndarray, net_load: np.ndarray) -> np.ndarray:
    """(n, 2(n-1)): row i is [SoC_j, net_load_j] for each peer j, in channel order."""
    n_agents = len(soc)
    peer_table = np.array([peer_indices(n_agents, receiver) for receiver in range(n_agents)])

    block = np.empty((n_agents, message_dim(n_agents)), dtype=np.float32)
    block[:, 0::2] = np.asarray(soc, dtype=np.float32)[peer_table]
    block[:, 1::2] = np.asarray(net_load, dtype=np.float32)[peer_table]
    return block


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
        full_peer_info: bool = False,
    ) -> None:
        if use_messages and (channel is None or schedule is None):
            raise ValueError("message arms need a channel and a schedule")
        if use_messages and full_peer_info:
            raise ValueError("an arm gets either lossy messages or full peer information")
        if full_peer_info and not hasattr(env, "net_load"):
            raise ValueError("full peer information needs env.net_load()")
        if critic not in ("central", "local"):
            raise ValueError("critic must be 'central' or 'local'")

        self.env = env
        self.n_agents = env.n_agents
        self.use_messages = use_messages
        self.full_peer_info = full_peer_info
        self.channel, self.schedule = channel, schedule
        self.preprocessor = preprocessor or Preprocessor(env.obs_dims, env.state_dim,
                                                         enabled=False)
        self.critic = critic
        self.env_seed_rng = np.random.default_rng(env_seed)
        self.update_normalizers = update_normalizers
        self.auto_reset = auto_reset

        # Input sizes and on-policy space objects (critic: global state, or the
        # agent's own input for IPPO)
        has_peer_block = use_messages or full_peer_info
        message_len = message_dim(self.n_agents) if has_peer_block else 0
        self.actor_dims = [dim + message_len for dim in env.obs_dims]
        self.critic_dims = ([env.state_dim] * self.n_agents if critic == "central"
                            else list(self.actor_dims))
        self.observation_space = [Box(-np.inf, np.inf, (dim,)) for dim in self.actor_dims]
        self.share_observation_space = [Box(-np.inf, np.inf, (dim,)) for dim in self.critic_dims]
        self.action_space = [Box(-1.0, 1.0, (dim,)) for dim in env.action_dims]

        self.reset_seeds: list = []
        self.episodes: list = []
        self.last_soc: Optional[np.ndarray] = None
        self.last_delivered: Optional[np.ndarray] = None
        self.p: Optional[float] = None


    def _build_inputs(self, obs):
        """Actor and critic inputs for the state the agents are about to act in."""
        soc = np.asarray(self.env.soc(), dtype=np.float32)
        self.last_soc = soc
        local_obs = self.preprocessor.normalize_local_obs(obs, self.update_normalizers)

        peer_block = None
        if self.use_messages:
            peer_block = self.channel.transmit(soc)
            self.last_delivered = self.channel.last_delivered.copy()
            self._episode_stats["delivered"] += int(self.last_delivered.sum())
            self._episode_stats["links"] += self.n_agents * (self.n_agents - 1)
        elif self.full_peer_info:
            net_load = self.preprocessor.normalize_net_load(self.env.net_load(),
                                                            self.update_normalizers)
            peer_block = full_peer_block(soc, net_load)

        actor_inputs = build_actor_inputs(local_obs, peer_block)
        if self.critic == "central":
            state = self.preprocessor.normalize_state(self.env.global_state(),
                                                      self.update_normalizers)
            critic_inputs = [state] * self.n_agents
        else:
            critic_inputs = actor_inputs
        return actor_inputs, critic_inputs

    def reset(self, seed: Optional[int] = None):
        """Start an episode; returns (actor inputs, critic inputs)."""
        if self.use_messages:
            self.p = self.schedule.sample()
            self.channel.set_p(self.p)
            self.channel.reset()

        # seeds come from a stream that depends only on the run seed -> paired across arms
        if seed is None:
            seed = int(self.env_seed_rng.integers(2 ** 31 - 1))
        self.reset_seeds.append(int(seed))
        obs, _ = self.env.reset(seed=int(seed))

        self._episode_stats = {"return": 0.0, "length": 0, "p": self.p,
                               "delivered": 0, "links": 0}
        return self._build_inputs(obs)

    def step(self, actions):
        """actions: (n, a) array or list. Returns
        (actor inputs, critic inputs, rewards (n, 1), done, truncated, info).

        With ``auto_reset`` the returned inputs already belong to the next
        episode when ``done`` (on-policy's convention). The finished
        episode's summary is in ``info["episode"]``.
        """
        clipped_actions = [np.clip(np.asarray(a, dtype=np.float32).reshape(-1), -1.0, 1.0)
                           for a in actions]
        obs, reward, terminated, truncated, info = self.env.step(clipped_actions)
        info = dict(info)

        self._episode_stats["return"] += float(reward)
        self._episode_stats["length"] += 1
        done = bool(terminated or truncated)
        # the shared reward, copied for every agent's buffer
        rewards = np.full((self.n_agents, 1), float(reward), dtype=np.float32)

        if not done:
            actor_inputs, critic_inputs = self._build_inputs(obs)
            return actor_inputs, critic_inputs, rewards, done, False, info

        finished = dict(self._episode_stats)
        finished["delivery_rate"] = (finished["delivered"] / finished["links"]
                                     if finished["links"] else None)
        self.episodes.append(finished)
        info["episode"] = finished

        # a time-limit cut, not a true terminal: on-policy's bad_masks use this flag
        time_limit = bool(truncated and not terminated)
        if self.auto_reset:
            actor_inputs, critic_inputs = self.reset()
            return actor_inputs, critic_inputs, rewards, done, time_limit, info
        return None, None, rewards, done, time_limit, info


def make_message_env(env, config, preprocessor: Optional[Preprocessor] = None,
                     update_normalizers: bool = True) -> MessageEnv:
    """MessageEnv for training an arm described by a ``MAPPOConfig``."""
    from .config import STREAM_CHANNEL, STREAM_ENV, STREAM_SCHEDULE, stream

    channel = schedule = None
    if config.use_messages:
        channel = PeerSoCChannel(env.n_agents, 1.0, seed=stream(config.seed, STREAM_CHANNEL),
                                 force_mask=config.force_mask_train)
        if config.train_schedule == "fixed":
            schedule = DeliverySchedule("fixed", p=config.train_p)
        else:
            schedule = DeliverySchedule("random", choices=tuple(config.rnd_p_choices),
                                        seed=stream(config.seed, STREAM_SCHEDULE))

    preprocessor = preprocessor or Preprocessor(env.obs_dims, env.state_dim,
                                                config.normalize_obs, config.obs_clip)
    return MessageEnv(env, config.use_messages, channel, schedule, preprocessor, config.critic,
                      stream(config.seed, STREAM_ENV), update_normalizers,
                      full_peer_info=config.full_peer_info)
