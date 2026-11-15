"""Non-learning references (proposal Section 4).

* R0: no battery operation (zero action every step). Normalizes all KPIs.
* R1: CityLearn's rule-based battery controller, ``BasicBatteryRBC``: an
  hour-of-day schedule that charges 11% of capacity per hour from 06:00 to
  14:00 (to soak up solar) and discharges 6.7% per hour otherwise.

Controllers act on the env's raw per-building observations (R1 needs the
hour), so they run through ``rollout`` rather than the MAPPO ``evaluate``
harness, which feeds normalized actor inputs. ``rollout`` returns records
with the same keys Vishal's analysis reads from ``evaluate``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

#: BasicBatteryRBC's storage schedule in CityLearn 3.0.2 (citylearn/agents/rbc.py),
#: used if the class cannot be built (e.g. CityLearn not installed).
_BASIC_BATTERY_RBC_SCHEDULE = {hour: (0.11 if 6 <= hour <= 14 else -0.067) for hour in range(1, 25)}


class ZeroController:
    """R0: never use the battery."""
    name = "R0"

    def __init__(self, env):
        self.action_dims = list(env.action_dims)

    def __call__(self, obs_list):
        return [np.zeros(dim, dtype=np.float32) for dim in self.action_dims]


class BasicBatteryRBCController:
    """R1: CityLearn's BasicBatteryRBC hour-of-day battery schedule."""
    name = "R1"

    def __init__(self, env):
        self.hour_positions = [names.index("hour") for names in env.observation_names]
        self.schedule = self._citylearn_schedule(env)

    @staticmethod
    def _citylearn_schedule(env) -> dict:
        try:
            from citylearn.agents.rbc import BasicBatteryRBC
            action_map = BasicBatteryRBC(env.env).action_map
            if isinstance(action_map, list):
                action_map = action_map[0]
            return {int(hour): float(value)
                    for hour, value in action_map["electrical_storage"].items()}
        except Exception:     # pragma: no cover - only without a real CityLearn
            return dict(_BASIC_BATTERY_RBC_SCHEDULE)

    def action_for_hour(self, hour_observation: float) -> float:
        # CityLearn encodes hours as 1-24; map 0 (as in 0-23 encodings) to 24
        return self.schedule[(int(round(float(hour_observation))) - 1) % 24 + 1]

    def __call__(self, obs_list):
        return [np.array([self.action_for_hour(obs[position])], dtype=np.float32)
                for obs, position in zip(obs_list, self.hour_positions)]


CONTROLLERS = {"R0": ZeroController, "R1": BasicBatteryRBCController}


def rollout(env, controller, seed: int, max_steps: Optional[int] = None) -> dict:
    """Run one episode with a raw-observation controller and return its record."""
    obs, _ = env.reset(seed=seed)
    actions_log, energy_log, soc_log, rewards = [], [], [], []
    while True:
        soc_log.append(np.asarray(env.soc(), dtype=np.float32))
        actions = controller(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        actions_log.append(np.concatenate([np.asarray(a, np.float32).reshape(-1) for a in actions]))
        energy_log.append(np.asarray(info.get("battery_energy"), dtype=np.float32))
        rewards.append(float(reward))
        if terminated or truncated or (max_steps is not None and len(rewards) >= max_steps):
            break
    return {
        "controller": getattr(controller, "name", type(controller).__name__),
        "actions": np.stack(actions_log),
        "battery_energy": np.stack(energy_log),
        "soc": np.stack(soc_log),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "return": float(np.sum(rewards)),
        "kpis": env.kpis() if hasattr(env, "kpis") else None,
    }
