"""Episode KPIs (proposal Section 6, "Metrics").

* Primary: average daily district peak (mean over calendar days of the
  maximum hourly district net load), relative to R0 on the same window and
  outage seed. Lower is better; R0 scores 1.0 by construction.
* Secondary: district electricity cost, sum of price_t * max(0, L_t),
  relative to R0.

The synchrony and robustness metrics belong to Part C (Vishal) and are
computed from evaluation records (``battery_energy``), not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .reward import step_cost, step_import_sq

STEPS_PER_DAY = 24


@dataclass
class EpisodeLog:
    """Per-step district quantities of one episode."""
    steps: list = field(default_factory=list)     # dataset time step of each transition
    loads: list = field(default_factory=list)     # L_t, kWh
    prices: list = field(default_factory=list)
    rewards: list = field(default_factory=list)

    def append(self, step: int, load: float, price: float, reward: float) -> None:
        self.steps.append(int(step))
        self.loads.append(float(load))
        self.prices.append(float(price))
        self.rewards.append(float(reward))

    def __len__(self) -> int:
        return len(self.steps)

    def head(self, n: int) -> "EpisodeLog":
        return EpisodeLog(self.steps[:n], self.loads[:n], self.prices[:n], self.rewards[:n])


def daily_peaks(steps, loads, steps_per_day: int = STEPS_PER_DAY) -> np.ndarray:
    """Max load per calendar day (days are dataset_step // steps_per_day)."""
    steps = np.asarray(steps, dtype=np.int64)
    loads = np.asarray(loads, dtype=np.float64)
    if steps.size == 0:
        return np.zeros(0)
    days = steps // steps_per_day
    return np.array([loads[days == day].max() for day in np.unique(days)])


def raw_kpis(log: EpisodeLog) -> dict:
    """KPIs of one episode in physical units."""
    if len(log) == 0:
        raise ValueError("no steps logged")
    peaks = daily_peaks(log.steps, log.loads)
    loads = np.asarray(log.loads)
    return {
        "avg_daily_peak": float(peaks.mean()),
        "cost": float(sum(step_cost(load, price) for load, price in zip(log.loads, log.prices))),
        "max_peak": float(peaks.max()),
        "mean_load": float(loads.mean()),
        "mean_import_sq": float(np.mean([step_import_sq(load) for load in loads])),   # mean h_t
        "n_days": int(peaks.size),
        "n_steps": int(len(log)),
        "return": float(np.sum(log.rewards)),
    }


def relative_kpis(kpis: dict, r0_kpis: dict) -> dict:
    """Primary and secondary outcomes relative to R0 (lower is better)."""
    return {
        "primary": kpis["avg_daily_peak"] / r0_kpis["avg_daily_peak"],
        "cost_rel_r0": kpis["cost"] / r0_kpis["cost"] if r0_kpis["cost"] else float("nan"),
    }
