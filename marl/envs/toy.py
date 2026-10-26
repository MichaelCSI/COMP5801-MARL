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

# Battery parameters of the proposal's six buildings
DEFAULT_CAPACITY_KWH = [4.0, 4.0, 3.3, 3.3, 4.0, 3.3]
DEFAULT_POWER_KW = [3.32, 3.32, 1.61, 1.61, 3.32, 1.61]


def _per_building(values: Optional[list], defaults: list, n_agents: int) -> np.ndarray:
    """Given values, or the defaults padded with their first entry to n_agents."""
    if values is not None:
        return np.asarray(values, dtype=np.float64)
    padded = defaults[:n_agents] + [defaults[0]] * max(0, n_agents - len(defaults))
    return np.asarray(padded, dtype=np.float64)


class ToyDistrictEnv:
    N_LOCAL = 4    # soc, net_load, load, solar
    N_SHARED = 3   # sin(hour), cos(hour), price

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
        self.capacity = _per_building(capacity, DEFAULT_CAPACITY_KWH, n_agents)
        self.power = _per_building(power, DEFAULT_POWER_KW, n_agents)
        self.alpha, self.beta, self.noise = alpha, beta, noise

        self.obs_dims = [self.N_LOCAL + self.N_SHARED] * n_agents
        self.action_dims = [1] * n_agents
        self.state_dim = self.N_SHARED + self.N_LOCAL * n_agents

        self.rng = np.random.default_rng(0)
        self._soc = np.zeros(n_agents)
        self.time_step = 0

        # Reward normalizers (proposal Section 2): mean per-step cost and
        # squared load under no-battery control
        self._mean_cost_no_battery, self._mean_sq_load_no_battery = self._no_battery_means()

    # profiles ---------------------------------------------------------
    def _profile(self, time_step: int):
        """(hour, base load per building, solar per building, price) at a time step."""
        hour = time_step % 24
        # evening peak around 19:00 plus a smaller morning peak around 08:00
        base_load = (1.0 + 0.8 * np.exp(-((hour - 19) ** 2) / 8.0)
                     + 0.3 * np.exp(-((hour - 8) ** 2) / 4.0))
        solar = np.clip(np.sin(np.pi * (hour - 6) / 12.0), 0.0, None) * 1.5
        price = 0.2 + 0.2 * (16 <= hour <= 21)
        return hour, base_load, solar, price

    def _no_battery_means(self):
        costs, squared_loads = [], []
        for hour in range(24):
            _, base_load, solar, price = self._profile(hour)
            district_load = self.n_agents * (base_load - solar)
            costs.append(price * max(0.0, district_load))
            squared_loads.append(district_load ** 2)
        return float(np.mean(costs)), float(np.mean(squared_loads))

    def _obs(self):
        hour, _, _, price = self._profile(self.time_step)
        net_load = self._load - self._solar
        shared = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), price]
        return [
            np.array([self._soc[i], net_load[i], self._load[i], self._solar[i], *shared],
                     dtype=np.float32)
            for i in range(self.n_agents)
        ]

    def _draw_exogenous(self):
        """Noisy load and solar for the current hour (5% multiplicative noise)."""
        _, base_load, solar, _ = self._profile(self.time_step)
        self._load = base_load * (1 + self.noise * self.rng.standard_normal(self.n_agents))
        self._solar = solar * (1 + self.noise * self.rng.standard_normal(self.n_agents)).clip(0)

    # contract ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # random start hour so episodes don't all begin at midnight
        self.time_step = int(self.rng.integers(24))
        self._episode_start = self.time_step
        self._soc = self.rng.uniform(0.2, 0.8, self.n_agents)
        self._draw_exogenous()
        return self._obs(), {}

    def step(self, actions):
        action = np.clip(np.array([float(np.asarray(a).reshape(-1)[0]) for a in actions]), -1, 1)

        # kWh this hour (+ = charge), limited by what each battery can take or give
        energy = action * self.power
        energy = np.clip(energy, -self._soc * self.capacity, (1 - self._soc) * self.capacity)
        self._soc = self._soc + energy / self.capacity

        # shared reward: -(alpha * normalized cost + beta * normalized squared load)
        district_load = float(np.sum(self._load - self._solar + energy))
        _, _, _, price = self._profile(self.time_step)
        cost = price * max(0.0, district_load)
        reward = -(self.alpha * cost / self._mean_cost_no_battery
                   + self.beta * district_load ** 2 / self._mean_sq_load_no_battery)
        info = {"district_load": district_load, "battery_energy": energy.copy()}

        self.time_step += 1
        truncated = (self.time_step - self._episode_start) >= self.episode_steps
        self._draw_exogenous()
        return self._obs(), float(reward), False, bool(truncated), info

    def soc(self):
        return self._soc.astype(np.float32).copy()

    def global_state(self):
        obs = self._obs()
        shared = obs[0][self.N_LOCAL:]    # identical in every building, so taken once
        local = np.concatenate([agent_obs[: self.N_LOCAL] for agent_obs in obs])
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
        # agent i's target is its right-hand neighbour's SoC, rescaled to [-1, 1]
        return 2.0 * np.roll(self._soc, -1) - 1.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.time_step = 0
        self._draw()
        return self._obs(), {}

    def step(self, actions):
        action = np.clip(np.array([float(np.asarray(a).reshape(-1)[0]) for a in actions]), -1, 1)
        reward = -float(np.mean((action - self.target()) ** 2))

        self.time_step += 1
        self._draw()
        return self._obs(), reward, False, self.time_step >= self.episode_steps, {}

    def soc(self):
        return self._soc.astype(np.float32).copy()

    def global_state(self):
        return self._soc.astype(np.float32).copy()
