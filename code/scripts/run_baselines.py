"""KPIs of the non-learning references R0 and R1 on the validation and test blocks.

    python -m scripts.run_baselines                 # both blocks
    python -m scripts.run_baselines --split validation

Each block is one episode at its frozen outage seed (VAL_SEED / TEST_SEED).
Writes results/baselines.json.
"""
from __future__ import annotations

import argparse
import json
import os

from marl.baselines import CONTROLLERS, rollout
from marl.envs.district_env import EVAL_SEEDS, DistrictBatteryEnv


def run_block(split: str) -> dict:
    env = DistrictBatteryEnv(split=split)
    results = {}
    for name, controller_class in CONTROLLERS.items():
        record = rollout(env, controller_class(env), seed=EVAL_SEEDS[split])
        results[name] = record["kpis"]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test", "both"], default="both")
    parser.add_argument("--out", default=os.path.join("results", "baselines.json"))
    args = parser.parse_args()

    splits = ["validation", "test"] if args.split == "both" else [args.split]
    output = {}
    print(f"{'block':<11}{'arm':<5}{'primary':>9}{'cost/R0':>9}{'peak kWh':>10}"
          f"{'cost':>9}{'return':>10}{'outage h':>9}")
    for split in splits:
        output[split] = run_block(split)
        for name, kpis in output[split].items():
            print(f"{split:<11}{name:<5}{kpis['primary']:>9.4f}{kpis['cost_rel_r0']:>9.4f}"
                  f"{kpis['avg_daily_peak']:>10.3f}{kpis['cost']:>9.3f}{kpis['return']:>10.1f}"
                  f"{kpis['outage_hours']:>9.0f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
