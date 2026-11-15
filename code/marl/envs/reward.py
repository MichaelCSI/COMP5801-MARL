"""Shared district reward (proposal Section 2).

    r_t = -(alpha * c_t / c_bar_R0 + beta * h_t / h_bar_R0)
    c_t = price_t * max(0, L_t)        district electricity cost
    h_t = max(0, L_t) ** 2             squared district import (peak signal)

``L_t`` is the district net load (sum of the six buildings' net electricity
consumption, kWh) in the step just taken. ``c_bar_R0`` and ``h_bar_R0`` are
the mean per-step values of ``c_t`` and ``h_t`` under no-battery control (R0)
on the training block; ``scripts/freeze_env.py`` computes them and stores
them in ``phase33_frozen.json``. With these norms the mean R0 reward on the
training block is exactly ``-(alpha + beta)``.

Exports (L_t < 0, midday solar surplus) are neither costed nor penalized, so
both terms agree with each other and with the primary KPI (daily import
peak). The first version squared L_t itself, which punished export like an
import peak (proposal feedback); ``scripts/export_check.py`` measures how
often L_t < 0.
"""
from __future__ import annotations

from dataclasses import dataclass


def step_cost(load: float, price: float) -> float:
    """c_t: district electricity cost of one step (exports earn nothing)."""
    return float(price) * max(0.0, float(load))


#: Tag of the peak-term definition, stored with the frozen norms and checked on
#: load, so norms frozen under a different h_t are never used silently.
H_FORMULA = "max(0,L)^2"


def step_import_sq(load: float) -> float:
    """h_t: squared district import of one step (exports count as zero)."""
    return max(0.0, float(load)) ** 2


@dataclass(frozen=True)
class DistrictReward:
    cost_norm: float          # c_bar_R0
    import_sq_norm: float     # h_bar_R0 = mean max(0, L)^2 under R0 (frozen key "load_sq")
    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self):
        if not (self.cost_norm > 0 and self.import_sq_norm > 0):
            raise ValueError("reward norms must be positive")

    def __call__(self, load: float, price: float) -> float:
        cost_hat = step_cost(load, price) / self.cost_norm
        import_sq_hat = step_import_sq(load) / self.import_sq_norm
        return -(self.alpha * cost_hat + self.beta * import_sq_hat)
