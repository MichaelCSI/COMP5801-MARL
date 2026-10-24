"""Small, fast environments that follow the contract in ``base.py``.

They let the trainer, the channel and the runner be developed and tested
before the real CityLearn wrapper exists, and they run thousands of steps per
second.

``ToyDistrictEnv``   a CityLearn-shaped district: n buildings with batteries,
                     a daily load/solar/price profile and the shared
                     cost + squared-load reward from Section 2.
``PeerSoCMatchEnv``  a diagnostic task that can only be solved with messages:
                     each agent is rewarded for outputting its right-hand
                     peer's SoC, which is NOT in its own observation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class ToyDistrictEnv:
    def __init__(
        self,
        n_agents: int = 6,
        episode_steps: int = 48,
        capacity: Optional[list] = None,
        power: Optional[list] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        noise: float = 0.05,
    ) -> None:
        self.n_agents = n_agents
        self.episode_steps = episode_steps
        self.capacity = np.asarray(capacity or [4.0, 4.0, 3.3, 3.3, 4.0, 3.3][:n_agents] +
                                   [4.0] * max(0, n_agents - 6), dtype=np.float64)
        self.power = np.asarray(power or [3.32, 3.32, 1.61, 1.61, 3.32, 1.61][:n_agents] +
                                [3.32] * max(0, n_agents - 6), dtype=np.float64)
        self.alpha, self.beta, self.noise = alpha, beta, noise
        # local: soc, net_load, load, solar | shared: sin h, cos h, price
        self._n_local, self._n_shared = 4, 3
        self.obs_dims = [self._n_local + self._n_shared] * n_agents
        self.action_dims = [1] * n_agents
        self.state_dim = self._n_shared + self._n_local * n_agents
        self.rng = np.random.default_rng(0)
        self._soc = np.zeros(n_agents)
        self.t = 0
        # normalizers for the reward, from no-battery control on the profile
        self._c_bar, self._h_bar = self._no_battery_means()

    # profiles ---------------------------------------------------------
    def _profile(self, t: int):
        h = t % 24
        base = 1.0 + 0.8 * np.exp(-((h - 19) ** 2) / 8.0) + 0.3 * np.exp(-((h - 8) ** 2) / 4.0)
        solar = np.clip(np.sin(np.pi * (h - 6) / 12.0), 0.0, None) * 1.5
        price = 0.2 + 0.2 * (16 <= h <= 21)
        return h, base, solar, price

    def _no_battery_means(self):
        c, hsq = [], []
        for t in range(24):
            _, base, solar, price = self._profile(t)
            L = self.n_agents * (base - solar)
            c.append(price * max(0.0, L))
            hsq.append(L ** 2)
        return float(np.mean(c)), float(np.mean(hsq))

    def _obs(self):
        h, base, solar, price = self._profile(self.t)
        net = self._load - self._solar
        shared = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), price]
        return [np.array([self._soc[i], net[i], self._load[i], self._solar[i], *shared],
                         dtype=np.float32) for i in range(self.n_agents)]

    def _draw_exogenous(self):
        _, base, solar, _ = self._profile(self.t)
        self._load = base * (1 + self.noise * self.rng.standard_normal(self.n_agents))
        self._solar = solar * (1 + self.noise * self.rng.standard_normal(self.n_agents)).clip(0)

    # contract ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = int(self.rng.integers(24))
        self._t0 = self.t
        self._soc = self.rng.uniform(0.2, 0.8, self.n_agents)
        self._draw_exogenous()
        return self._obs(), {}

    def step(self, actions):
        a = np.clip(np.array([float(np.asarray(x).reshape(-1)[0]) for x in actions]), -1, 1)
        energy = a * self.power  # kWh this hour, + = charge
        energy = np.clip(energy, -self._soc * self.capacity, (1 - self._soc) * self.capacity)
        self._soc = self._soc + energy / self.capacity
        L = float(np.sum(self._load - self._solar + energy))
        _, _, _, price = self._profile(self.t)
        c = price * max(0.0, L)
        reward = -(self.alpha * c / self._c_bar + self.beta * L ** 2 / self._h_bar)
        info = {"district_load": L, "battery_energy": energy.copy()}
        self.t += 1
        truncated = (self.t - self._t0) >= self.episode_steps
        self._draw_exogenous()
        return self._obs(), float(reward), False, bool(truncated), info

    def soc(self):
        return self._soc.astype(np.float32).copy()

    def global_state(self):
        o = self._obs()
        shared = o[0][self._n_local:]
        local = np.concatenate([oi[: self._n_local] for oi in o])
        return np.concatenate([shared, local]).astype(np.float32)


class PeerSoCMatchEnv:
    """Agent i must output 2*SoC_{i+1}-1; SoCs are redrawn every step.

    Its own observation holds only its own SoC plus noise, so without
    messages the best it can do is a constant. The reward is the negative mean
    squared error, so the optimum is 0.
    """

    def __init__(self, n_agents: int = 3, episode_steps: int = 32, n_noise: int = 2):
        self.n_agents = n_agents
        self.episode_steps = episode_steps
        self.n_noise = n_noise
        self.obs_dims = [1 + n_noise] * n_agents
        self.action_dims = [1] * n_agents
        self.state_dim = n_agents
        self.rng = np.random.default_rng(0)

    def _draw(self):
        self._soc = self.rng.uniform(0, 1, self.n_agents)
        self._noise = self.rng.standard_normal((self.n_agents, self.n_noise))

    def _obs(self):
        return [np.concatenate([[self._soc[i]], self._noise[i]]).astype(np.float32)
                for i in range(self.n_agents)]

    def target(self):
        return 2.0 * np.roll(self._soc, -1) - 1.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self._draw()
        return self._obs(), {}

    def step(self, actions):
        a = np.clip(np.array([float(np.asarray(x).reshape(-1)[0]) for x in actions]), -1, 1)
        reward = -float(np.mean((a - self.target()) ** 2))
        self.t += 1
        self._draw()
        return self._obs(), reward, False, self.t >= self.episode_steps, {}

    def soc(self):
        return self._soc.astype(np.float32).copy()

    def global_state(self):
        return self._soc.astype(np.float32).copy()
