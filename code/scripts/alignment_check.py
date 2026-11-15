"""Reward alignment check (proposal Section 2, part of gate G1).

    python -m scripts.alignment_check --m0 runs/M0_s0.pt [--m0 runs/M0_s1.pt ...]

On the validation block (frozen outage seed VAL_SEED), R0, R1 and the trained
no-message MAPPO policy M0 must rank the same under the reward (episode
return, higher is better) and under the primary outcome (average daily
district peak relative to R0, lower is better). If they don't, revise the
reward (alpha, beta) on the validation block before any message arm is
trained. With several M0 checkpoints, each seed is checked separately.

Exit code 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys

from marl.baselines import CONTROLLERS, rollout
from marl.envs.district_env import VAL_SEED, DistrictBatteryEnv


def rankings(results: dict):
    by_return = sorted(results, key=lambda name: -results[name]["return"])
    by_primary = sorted(results, key=lambda name: results[name]["primary"])
    return by_return, by_primary


def evaluate_m0(path: str, env) -> dict:
    from marl.mappo import OnPolicyRunner
    if OnPolicyRunner is None:
        raise SystemExit("the M0 evaluation needs torch")
    runner = OnPolicyRunner.load(path, env)
    record = runner.evaluate(env, p=1.0, n_replicates=1, env_seed=VAL_SEED,
                             on_episode_end=lambda e: e.kpis())[0]
    return record["kpis"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0", action="append", default=[],
                        help="M0 checkpoint (.pt); repeat for several seeds")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    env = DistrictBatteryEnv(split="validation", alpha=args.alpha, beta=args.beta)
    references = {name: rollout(env, controller(env), seed=VAL_SEED)["kpis"]
                  for name, controller in CONTROLLERS.items()}

    if args.m0:
        checks = [(path, {**references, "M0": evaluate_m0(path, env)}) for path in args.m0]
    else:
        print("no --m0 checkpoint given: checking R0 vs R1 only")
        checks = [(None, dict(references))]

    all_pass = True
    print(f"validation block, outage seed {VAL_SEED}, alpha={args.alpha}, beta={args.beta}")
    for path, results in checks:
        if path:
            print(f"\nM0 checkpoint: {path}")
        for name, kpis in results.items():
            print(f"  {name:<3} return {kpis['return']:>10.2f}   primary {kpis['primary']:.4f}"
                  f"   cost/R0 {kpis['cost_rel_r0']:.4f}")
        by_return, by_primary = rankings(results)
        passed = by_return == by_primary
        all_pass &= passed
        print(f"  ranking by return : {' > '.join(by_return)}")
        print(f"  ranking by primary: {' > '.join(by_primary)}")
        print(f"  alignment check: {'PASS' if passed else 'FAIL'}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
