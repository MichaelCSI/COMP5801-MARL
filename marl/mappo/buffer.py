"""On-policy rollout storage and generalized advantage estimation (numpy)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_gae(rewards, values, dones, bootstrap, last_value, gamma: float, lam: float):
    """GAE(gamma, lambda) for one stream of steps that may span episodes.

    rewards   (T,)    shared reward r_t
    values    (T, K)  V(s_t) for each of K critics (K=1 central, K=n local)
    dones     (T,)    True if the episode ended after step t
    bootstrap (T, K)  V(s_{t+1}) to use when dones[t] (0 for a true terminal)
    last_value (K,)   V(s_T) for the state after the last stored step
    Returns (advantages, returns), both (T, K).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    bootstrap = np.asarray(bootstrap, dtype=np.float64)
    T, K = values.shape
    adv = np.zeros((T, K), dtype=np.float64)
    gae = np.zeros(K)
    for t in reversed(range(T)):
        if dones[t]:
            next_v = bootstrap[t]
            gae = np.zeros(K)          # do not carry advantage across episodes
        else:
            next_v = values[t + 1] if t + 1 < T else np.asarray(last_value, dtype=np.float64)
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv.astype(np.float32), (adv + values).astype(np.float32)


class RolloutBuffer:
    def __init__(self, size: int, actor_dims: Sequence[int], action_dims: Sequence[int],
                 critic_dims: Sequence[int]):
        self.size = size
        self.n_agents = len(actor_dims)
        self.K = len(critic_dims)
        self.actor_inputs = [np.zeros((size, d), np.float32) for d in actor_dims]
        self.actions = [np.zeros((size, d), np.float32) for d in action_dims]
        self.critic_inputs = [np.zeros((size, d), np.float32) for d in critic_dims]
        self.logp = np.zeros((size, self.n_agents), np.float32)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, bool)
        self.values = np.zeros((size, self.K), np.float32)
        self.bootstrap = np.zeros((size, self.K), np.float32)
        self.last_value = np.zeros(self.K, np.float32)
        self.ptr = 0

    @property
    def full(self) -> bool:
        return self.ptr >= self.size

    def reset(self) -> None:
        self.ptr = 0

    def add(self, actor_inputs, critic_inputs, actions, logp, reward, done, values,
            bootstrap) -> None:
        if self.full:
            raise IndexError("rollout buffer is full")
        t = self.ptr
        for i in range(self.n_agents):
            self.actor_inputs[i][t] = actor_inputs[i]
            self.actions[i][t] = np.asarray(actions[i]).reshape(-1)
        for k in range(self.K):
            self.critic_inputs[k][t] = critic_inputs[k]
        self.logp[t] = logp
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = values
        self.bootstrap[t] = bootstrap
        self.ptr += 1

    def compute(self, gamma: float, lam: float):
        n = self.ptr
        self.advantages, self.returns = compute_gae(
            self.rewards[:n], self.values[:n], self.dones[:n], self.bootstrap[:n],
            self.last_value, gamma, lam)
        return self.advantages, self.returns

    def minibatches(self, num_minibatches: int, rng: np.random.Generator):
        idx = rng.permutation(self.ptr)
        for mb in np.array_split(idx, num_minibatches):
            yield mb
