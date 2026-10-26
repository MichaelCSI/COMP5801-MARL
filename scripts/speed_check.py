"""Week-1 speed check: raw CityLearn steps/s and MAPPO rollout steps/s.

    python -m scripts.speed_check            # CityLearn 2023 phase 3.3
    python -m scripts.speed_check --env toy

Target (proposal Section 5): one training run <= ~2 h.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from scripts.train import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["toy", "citylearn"], default="citylearn")
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument(
        "--train-steps",
        type=int,
        default=200_000,
        help="planned env steps per run, for the time estimate"
    )
    args = parser.parse_args()

    # 1) the environment alone, with random actions
    env = make_env(args.env, args.steps)
    env.reset(seed=0)
    start = time.perf_counter()

    for _ in range(args.steps):
        random_actions = [np.random.uniform(-1, 1, dim).astype(np.float32)
                          for dim in env.action_dims]
        _, _, terminated, truncated, _ = env.step(random_actions)
        if terminated or truncated:
            env.reset()

    env_steps_per_sec = args.steps / (time.perf_counter() - start)
    print(f"raw env: {env_steps_per_sec:.0f} steps/s")

    # 2) MAPPO training on the same env: rollout plus one update
    from marl.mappo import MAPPOConfig, OnPolicyRunner
    if OnPolicyRunner is None:
        print("torch not installed: skipped the MAPPO timing")
        return

    config = MAPPOConfig.for_arm("M1-RND", total_steps=args.steps, rollout_steps=args.steps)
    start = time.perf_counter()
    history = OnPolicyRunner(make_env(args.env, args.steps), config).run()
    seconds_per_step = (time.perf_counter() - start) / args.steps

    print(f"MAPPO M1-RND: rollout {history[-1]['steps_per_sec']:.0f} steps/s, "
          f"rollout+update {1 / seconds_per_step:.0f} steps/s")
    print(f"estimated run time for {args.train_steps} steps: "
          f"{args.train_steps * seconds_per_step / 3600:.2f} h (target <= 2 h)")


if __name__ == "__main__":
    main()
