"""Freeze the environment constants into marl/envs/phase33_frozen.json.

    python -m scripts.freeze_env

Writes, once, before any experiment (proposal Sections 1, 2 and 6):

* the active observation set per building (and which are shared);
* the reward normalization: mean per-step c_t = price_t * max(0, L_t) and
  h_t = max(0, L_t)^2 (``marl.envs.reward``; tagged with ``H_FORMULA``, which
  the env checks on load) under no-battery control (R0) over the whole training block (steps 0-1487), at a fixed
  outage seed (NORM_SEED);
* the evaluation seeds: for the validation and the test block, the smallest
  non-negative reset seed whose outage realization has at least one outage
  hour inside that block (so the evaluation blocks exercise outages);
* building parameters from the schema, to check the proposal's table.

Rerun whenever ``marl/envs/reward.py``, the schema or CityLearn changes, and
commit the JSON; the env refuses norms whose ``h_formula`` does not match.
"""
from __future__ import annotations

import argparse
import json

from marl.baselines import ZeroController, rollout
from marl.envs.citylearn_adapter import SPLITS
from marl.envs.district_env import FROZEN_PATH, DistrictBatteryEnv
from marl.envs.kpis import raw_kpis
from marl.envs.reward import H_FORMULA

NORM_SEED = 0


def r0_norms(env: DistrictBatteryEnv, seed: int):
    """(c_bar, h_bar, n_steps): mean per-step cost and h_t under R0."""
    rollout(env, ZeroController(env), seed)
    kpis = raw_kpis(env.log)
    return kpis["cost"] / kpis["n_steps"], kpis["mean_import_sq"], kpis["n_steps"]


def first_seed_with_outage(env: DistrictBatteryEnv, block, max_seed: int = 10_000) -> int:
    start, end = block
    for seed in range(max_seed):
        if env.outage_calendar(seed)[:, start:end + 1].sum() > 0:
            return seed
    raise RuntimeError(f"no seed below {max_seed} has an outage in {block}")


def building_parameters(env: DistrictBatteryEnv) -> list:
    """Battery and PV parameters per building, from the schema.

    ``battery_efficiency`` is the schema's nominal value (the proposal's
    table); CityLearn applies it through ``power_efficiency_curve``, so the
    realized efficiency depends on the charge rate.
    """
    with open(env.schema) as handle:
        schema = json.load(handle)
    rows = []
    for building in env.env.buildings:
        storage = schema["buildings"][building.name]["electrical_storage"]["attributes"]
        rows.append({
            "name": building.name,
            "battery_capacity_kwh": float(storage["capacity"]),
            "battery_power_kw": float(storage["nominal_power"]),
            "battery_efficiency": float(storage["efficiency"]),
            "battery_power_efficiency_curve": storage.get("power_efficiency_curve"),
            "battery_depth_of_discharge": float(storage["depth_of_discharge"]),
            "pv_kw": round(float(building.pv.nominal_power), 6),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=FROZEN_PATH)
    args = parser.parse_args()

    import citylearn

    env = DistrictBatteryEnv(window=SPLITS["train"], reward_norm=(1.0, 1.0), frozen={},
                             check_observations=False, r0_reference=False)
    cost_norm, import_sq_norm, n_steps = r0_norms(env, NORM_SEED)

    frozen = {
        "dataset": "citylearn_challenge_2023_phase_3_3",
        "citylearn_version": citylearn.__version__,
        "observation_names": env.observation_names,
        "shared_observation_names": sorted(getattr(env.env, "shared_observations", []) or []),
        "reward_norm": {
            "cost": cost_norm,
            "load_sq": import_sq_norm,
            "h_formula": H_FORMULA,
            "controller": "R0 (zero battery action)",
            "block": list(SPLITS["train"]),
            "outage_seed": NORM_SEED,
            "n_steps": n_steps,
        },
        "val_seed": first_seed_with_outage(env, SPLITS["validation"]),
        "test_seed": first_seed_with_outage(env, SPLITS["test"]),
        "eval_seed_rule": "smallest seed >= 0 with at least one outage hour inside the block",
        "buildings": building_parameters(env),
    }

    with open(args.out, "w") as handle:
        json.dump(frozen, handle, indent=2)
        handle.write("\n")

    print(f"wrote {args.out}")
    print(f"  c_bar_R0 = {cost_norm:.6f}  h_bar_R0 = {import_sq_norm:.6f}  ({n_steps} steps)")
    print(f"  val_seed = {frozen['val_seed']}  test_seed = {frozen['test_seed']}")
    for row in frozen["buildings"]:
        print("  ", row)


if __name__ == "__main__":
    main()
