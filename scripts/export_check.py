"""How often does the district export (L_t < 0)? Proposal feedback, reward term.

    python -m scripts.export_check

The peak term was changed from h_t = L_t^2 to h_t = max(0, L_t)^2 so that
exports are not penalized like import peaks. This reports, for R0 and R1 on
each block, the share of steps with L_t < 0, the lowest L_t, and the share of
sum(L_t^2) that came from export steps (how much of the old peak term was
penalizing export). Writes results/export_check.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from marl.baselines import CONTROLLERS, rollout
from marl.envs.citylearn_adapter import SPLITS
from marl.envs.district_env import EVAL_SEEDS, DistrictBatteryEnv
from scripts.freeze_env import NORM_SEED

BLOCKS = {"train": NORM_SEED, **EVAL_SEEDS}


def export_stats(loads) -> dict:
    loads = np.asarray(loads, dtype=np.float64)
    exporting = loads < 0
    total_sq = float(np.sum(loads ** 2))
    return {
        "n_steps": int(loads.size),
        "export_steps": int(exporting.sum()),
        "export_fraction": float(exporting.mean()),
        "min_load": float(loads.min()),
        "export_share_of_old_load_sq": float(np.sum(loads[exporting] ** 2) / total_sq) if total_sq else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join("results", "export_check.json"))
    args = parser.parse_args()

    output = {}
    print(f"{'block':<11}{'arm':<5}{'export steps':>14}{'fraction':>10}{'min L':>9}{'share of old sum L^2':>22}")
    for block, seed in BLOCKS.items():
        env = DistrictBatteryEnv(window=SPLITS[block], r0_reference=False)
        output[block] = {}
        for name, controller_class in CONTROLLERS.items():
            rollout(env, controller_class(env), seed=seed)
            stats = export_stats(env.log.loads)
            stats["seed"] = seed
            output[block][name] = stats
            print(f"{block:<11}{name:<5}{stats['export_steps']:>8}/{stats['n_steps']:<5}"
                  f"{stats['export_fraction']:>10.3f}{stats['min_load']:>9.2f}"
                  f"{stats['export_share_of_old_load_sq']:>22.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
