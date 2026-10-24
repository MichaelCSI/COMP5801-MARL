"""Week-1 speed check: raw CityLearn steps/s and MAPPO rollout steps/s.

    python -m scripts.speed_check            # CityLearn 2023 phase 3.3
    python -m scripts.speed_check --env toy

Target (proposal Section 5): one training run <= ~2 h.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["toy", "citylearn"], default="citylearn")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--train-steps", type=int, default=200_000,
                    help="planned env steps per run, for the time estimate")
    a = ap.parse_args()

    from scripts.train import make_env
    env = make_env(a.env, a.steps)

    env.reset(seed=0)
    t0 = time.perf_counter()
    for _ in range(a.steps):
        _, _, te, tr, _ = env.step([np.random.uniform(-1, 1, d).astype(np.float32)
                                    for d in env.action_dims])
        if te or tr:
            env.reset()
    raw = a.steps / (time.perf_counter() - t0)
    print(f"raw env: {raw:.0f} steps/s")

    from marl.mappo import MAPPOConfig, MAPPOTrainer
    if MAPPOTrainer is None:
        print("torch not installed: skipped the MAPPO timing")
        return
    cfg = MAPPOConfig.for_arm("M1-RND", total_steps=a.steps, rollout_steps=a.steps,
                              num_minibatches=4 if a.steps % 4 == 0 else 1)
    t0 = time.perf_counter()
    hist = MAPPOTrainer(make_env(a.env, a.steps), cfg).train()
    total = time.perf_counter() - t0
    per_step = total / a.steps
    print(f"MAPPO M1-RND: rollout {hist[-1]['steps_per_sec']:.0f} steps/s, "
          f"rollout+update {1 / per_step:.0f} steps/s")
    print(f"estimated run time for {a.train_steps} steps: "
          f"{a.train_steps * per_step / 3600:.2f} h (target <= 2 h)")


if __name__ == "__main__":
    main()
