"""Calendar outage seeding (Part A, proposal Section 1 "Action and outages").

Problem (see WorkPlan.md): in the Phase 3.3 schema every building's
``ReliabilityMetricsPowerOutage`` has a fixed ``random_seed`` (57841) that
``CityLearnEnv.reset(seed)`` does not reach, and CityLearn draws the outage
signal per *episode*, starting from the episode's first step. So every
episode has the same outages, windows shorter than ~43 days have none, and
all six buildings fail at the same hours.

Fix: for a reset seed ``s`` we draw each building's signal once over the
whole 92-day calendar (steps 0-2207) with its own seed derived from
``(s, building)``, and every episode takes the slice of that calendar that
covers its window. Consequences:

* the outage realization depends only on the reset seed, so it is identical
  in every arm (reset seeds are paired across arms by ``MessageEnv``);
* outages sit on fixed calendar days for a given seed, whatever the window
  length, so short training windows and the test block can contain them;
* the six buildings fail independently.

The outage model itself (SAIFI/CAIDI, start-hour distribution) is the
schema's, unchanged; only its seed and its time axis are ours.
"""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

#: Length of the Phase 3.3 calendar (92 days of hourly steps).
CALENDAR_STEPS = 2208

#: Tag mixed into every outage seed so this stream never collides with the
#: other seed streams (windows, channel, schedule, networks).
_OUTAGE_STREAM_TAG = 5801


def outage_seed(reset_seed: int, building: int) -> int:
    """Seed for one building's outage model, derived from the reset seed."""
    state = np.random.SeedSequence([_OUTAGE_STREAM_TAG, int(reset_seed), int(building)])
    return int(state.generate_state(1, dtype=np.uint32)[0])


def calendar_outage_signals(models: Sequence, reset_seed: int,
                            n_steps: int = CALENDAR_STEPS,
                            seconds_per_time_step: float = 3600.0) -> np.ndarray:
    """(n_buildings, n_steps) 0/1 outage signal on the dataset calendar.

    ``models[b]`` is building b's CityLearn outage model (anything with a
    ``random_seed`` attribute and ``get_signals(time_steps, seconds_per_time_step=...)``),
    or ``None`` for a building that does not simulate outages. Models are
    copied, never modified.
    """
    signals = np.zeros((len(models), n_steps), dtype=np.float32)
    for building, model in enumerate(models):
        if model is None:
            continue
        seeded = copy.copy(model)
        seeded.random_seed = outage_seed(reset_seed, building)
        signal = np.asarray(seeded.get_signals(n_steps, seconds_per_time_step=seconds_per_time_step),
                            dtype=np.float32).reshape(-1)[:n_steps]
        signals[building, :len(signal)] = signal
    return signals
