"""The project's CityLearn environment (Part A, Andrea).

``DistrictBatteryEnv`` extends Michael's temporary ``CityLearnAdapter``, which
already handles the parts of the contract tested in
``tests/test_citylearn_adapter.py``: battery-only decentralized control, the
chronological split, window seeds, truncation at the window end,
``battery_energy``, ``soc()`` and ``global_state()``. This class adds:

* the proposal's shared district reward (``marl.envs.reward``), normalized by
  R0 on the training block;
* calendar outage seeding (``marl.envs.outages``): ``reset(seed)`` fixes the
  outage realization, which is identical in every arm;
* KPIs (``marl.envs.kpis``) through ``env.kpis()``, with the R0 reference run
  on the same window and outage seed;
* a frozen observation set (``phase33_frozen.json``), checked on construction.

``info`` per step also carries ``district_load``, ``price``, ``cost``, the
current ``outage`` flags and ``building_net_load`` (each building's net
consumption in the step just taken, for a full-observability arm or load
messages).
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from marl.baselines import ZeroController, rollout

from .citylearn_adapter import DATASET_2023_PHASE_3_3, SPLITS, CityLearnAdapter
from .kpis import EpisodeLog, raw_kpis, relative_kpis
from .outages import CALENDAR_STEPS, calendar_outage_signals
from .reward import H_FORMULA, DistrictReward, step_cost

FROZEN_PATH = os.path.join(os.path.dirname(__file__), "phase33_frozen.json")


def load_frozen(path: str = FROZEN_PATH) -> dict:
    """Frozen environment constants written by ``scripts/freeze_env.py``."""
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def resolve_schema(name: str = DATASET_2023_PHASE_3_3) -> str:
    """Path of the dataset's schema.json in CityLearn's cache.

    Passing the path (not the name) keeps ``CityLearnEnv`` from calling the
    GitHub API on every construction, which is rate limited.
    """
    if os.path.isfile(name):
        return name
    from citylearn.data import DataSet   # imported lazily
    return DataSet().get_dataset(name)


def _frozen_seed(key: str, default: int) -> int:
    return int(load_frozen().get(key, default))


#: Fixed reset seeds for the evaluation blocks (outage realization + replicates).
VAL_SEED = _frozen_seed("val_seed", 0)
TEST_SEED = _frozen_seed("test_seed", 0)
EVAL_SEEDS = {"validation": VAL_SEED, "test": TEST_SEED}


class DistrictBatteryEnv(CityLearnAdapter):
    """Six-building battery district with the proposal's reward, KPIs and outages.

    Parameters
    ----------
    split: "train" (random windows of ``episode_time_steps`` inside days 1-62),
        "validation" or "test" (one episode over the whole block).
    window: (start, end) dataset steps, inclusive. Overrides ``split`` with a
        single fixed episode; used for R0 references and normalization.
    alpha, beta: reward weights (1, 1 unless changed on the validation block).
    reward_norm: (c_bar_R0, h_bar_R0); defaults to the frozen values.
    frozen: dict of frozen constants, or a path; defaults to phase33_frozen.json.
    check_observations: assert the active observation set equals the frozen one.
    outages: use calendar outage seeding (True) or switch outages off (False).
    r0_reference: let ``kpis()`` compute values relative to R0.
    """

    def __init__(
        self,
        split: str = "train",
        episode_time_steps: int = 168,
        window: Optional[Tuple[int, int]] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        reward_norm: Optional[Sequence[float]] = None,
        frozen: Union[dict, str, None] = None,
        check_observations: bool = True,
        outages: bool = True,
        r0_reference: bool = True,
        schema: Optional[str] = None,
        **citylearn_kwargs,
    ) -> None:
        if frozen is None or isinstance(frozen, str):
            self.frozen = load_frozen(frozen or FROZEN_PATH)
        else:
            self.frozen = dict(frozen)
        self.schema = schema if schema is not None else resolve_schema()
        self.alpha, self.beta = float(alpha), float(beta)
        self.use_outages = bool(outages)
        self.use_r0_reference = bool(r0_reference)
        self._init_kwargs = dict(citylearn_kwargs)

        if window is not None:
            start, end = int(window[0]), int(window[1])
            train_length = SPLITS["train"][1] - SPLITS["train"][0] + 1
            if not (0 <= start < end < CALENDAR_STEPS and end - start + 1 <= train_length):
                raise ValueError(f"window must lie inside steps 0-{CALENDAR_STEPS - 1} "
                                 f"and be at most {train_length} steps long")
            # the adapter's "train" kwargs with the right length, then pinned
            # to this one window (no rolling or random choice)
            split, episode_time_steps = "train", end - start + 1
            citylearn_kwargs.update(
                simulation_start_time_step=start, simulation_end_time_step=end,
                rolling_episode_split=False, random_episode_split=False)

        super().__init__(schema=self.schema, split=split,
                         episode_time_steps=episode_time_steps, **citylearn_kwargs)

        if check_observations:
            self._check_frozen_observations()

        # reward
        if reward_norm is None:
            norm = self.frozen.get("reward_norm")
            if norm is None:
                raise RuntimeError(
                    "no reward normalization: run `python -m scripts.freeze_env` "
                    "or pass reward_norm=(c_bar, h_bar)")
            if norm.get("h_formula") != H_FORMULA:
                raise RuntimeError(
                    f"frozen reward norms were computed for h_formula="
                    f"{norm.get('h_formula')!r}, the reward uses {H_FORMULA!r}; "
                    "rerun `python -m scripts.freeze_env`")
            reward_norm = (norm["cost"], norm["load_sq"])
        self.reward = DistrictReward(float(reward_norm[0]), float(reward_norm[1]),
                                     self.alpha, self.beta)

        # outages: hook CityLearn's per-episode signal so it reads our calendar
        self._calendar = np.zeros((self.n_agents, CALENDAR_STEPS), dtype=np.float32)
        self._calendar_seed = None
        self._outage_models = []
        for index, building in enumerate(self.env.buildings):
            stochastic = (getattr(building, "simulate_power_outage", False)
                          and getattr(building, "stochastic_power_outage", False))
            self._outage_models.append(building.stochastic_power_outage_model
                                       if stochastic and self.use_outages else None)
            building.reset_power_outage_signal = self._make_outage_hook(index)

        self.seed = None
        self.log = EpisodeLog()
        self._r0_envs: dict = {}
        self._r0_logs: dict = {}

    # ------------------------------------------------------------------ setup
    def _check_frozen_observations(self) -> None:
        expected = self.frozen.get("observation_names")
        if expected is None:
            return       # nothing frozen yet (freeze_env.py is creating it)
        if self.observation_names != expected:
            raise RuntimeError(
                "active observation set differs from phase33_frozen.json; "
                "the schema or CityLearn version changed")

    def _make_outage_hook(self, building_index: int):
        def outage_signal():
            tracker = self.env.episode_tracker
            start = int(tracker.episode_start_time_step)
            end = int(tracker.episode_end_time_step)
            return self._calendar[building_index, start:end + 1].copy()
        return outage_signal

    def outage_calendar(self, seed: int) -> np.ndarray:
        """(n, 2208) outage signal on the dataset calendar for a reset seed."""
        return calendar_outage_signals(self._outage_models, seed,
                                       seconds_per_time_step=float(self.env.seconds_per_time_step))

    # ------------------------------------------------------------------- loop
    def reset(self, seed=None, options=None):
        if seed is None:
            seed = int(np.random.randint(2 ** 31 - 1))
        self.seed = int(seed)
        if self._calendar_seed != self.seed:
            self._calendar = self.outage_calendar(self.seed)
            self._calendar_seed = self.seed

        obs, info = super().reset(seed=self.seed, options=options)
        self.log = EpisodeLog()
        info["reset_seed"] = self.seed
        info["outage"] = self.outage_flags()
        return obs, info

    def _last_step(self) -> int:
        """Episode index of the step just taken.

        Uses the env clock like ``battery_energy``: on the terminal step the
        buildings' clocks do not advance, so ``env.time_step - 1`` is the step
        just taken. Building arrays are indexed by episode step.
        """
        return max(0, int(self.env.time_step) - 1)

    def current_price(self) -> float:
        """price_t of the step just taken."""
        return float(self.env.buildings[0].pricing.electricity_pricing[self._last_step()])

    def building_net_loads(self) -> np.ndarray:
        """(n,) each building's net electricity consumption in the step just taken (kWh)."""
        last_step = self._last_step()
        return np.asarray([float(b.net_electricity_consumption[last_step])
                           for b in self.env.buildings], dtype=np.float64)

    def outage_flags(self) -> np.ndarray:
        """(n,) 1 where a building is in an outage at the current step."""
        step = self.episode_window()[0] + int(self.env.time_step)
        step = min(step, CALENDAR_STEPS - 1)
        return self._calendar[:, step].copy()

    def step(self, actions):
        dataset_step = self.episode_window()[0] + int(self.env.time_step)
        obs, _, terminated, truncated, info = super().step(actions)

        building_loads = self.building_net_loads()
        load, price = float(building_loads.sum()), self.current_price()
        reward = self.reward(load, price)
        self.log.append(dataset_step, load, price, reward)

        info["district_load"] = load
        info["price"] = price
        info["cost"] = step_cost(load, price)
        info["dataset_step"] = dataset_step
        info["outage"] = self.outage_flags()
        info["building_net_load"] = building_loads.astype(np.float32)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------- KPIs
    def kpis(self) -> dict:
        """KPIs of the episode run so far (pass as ``on_episode_end``).

        Raw values in physical units, plus ``primary`` (average daily peak /
        R0's) and ``cost_rel_r0`` when the R0 reference is enabled. R0 is run
        on the same window and outage seed and cut to the same number of steps.
        """
        result = raw_kpis(self.log)
        result["window"] = list(self.episode_window())
        result["seed"] = self.seed
        result["outage_hours"] = float(self._calendar[:, self.log.steps].sum()) if len(self.log) else 0.0
        if self.use_r0_reference:
            r0 = raw_kpis(self.r0_log(self.episode_window(), self.seed).head(len(self.log)))
            result["r0"] = r0
            result.update(relative_kpis(result, r0))
        return result

    def r0_log(self, window: Tuple[int, int], seed: int) -> EpisodeLog:
        """District log of no-battery control (R0) on a window, cached."""
        key = (tuple(window), int(seed), self.alpha, self.beta)
        if key not in self._r0_logs:
            if tuple(window) not in self._r0_envs:
                self._r0_envs[tuple(window)] = type(self)(
                    window=window, alpha=self.alpha, beta=self.beta,
                    reward_norm=(self.reward.cost_norm, self.reward.import_sq_norm),
                    frozen=self.frozen, check_observations=False, outages=self.use_outages,
                    r0_reference=False, schema=self.schema, **self._init_kwargs)
            r0_env = self._r0_envs[tuple(window)]
            rollout(r0_env, ZeroController(r0_env), seed)
            self._r0_logs[key] = r0_env.log
        return self._r0_logs[key]
