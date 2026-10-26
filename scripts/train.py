"""Train one (arm, seed) run and write a checkpoint + JSONL log.

The learner is on-policy's reference MAPPO (Yu et al. 2022), vendored in
third_party/onpolicy; see marl/mappo/runner.py.

    python -m scripts.train --arm M1-RND --seed 0 --steps 200000 --env citylearn
    python -m scripts.train --arm M0 --seed 0 --steps 20000 --env toy

Vishal's experiment runner can call ``run(arm, seed, ...)`` directly instead.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from marl.mappo import ARMS, MAPPOConfig, OnPolicyRunner


def make_env(name: str, episode_steps: int):
    if name == "toy":
        from marl.envs import ToyDistrictEnv
        return ToyDistrictEnv(episode_steps=episode_steps)
    if name == "citylearn":
        # TEMPORARY until Andrea's wrapper (reward, split, outage seeding) lands
        from marl.envs.citylearn_adapter import CityLearnAdapter
        return CityLearnAdapter(episode_time_steps=episode_steps)
    raise ValueError(name)


def run(arm: str, seed: int, steps: int, env_name: str = "toy", episode_steps: int = 168,
        out_dir: str = "runs", **overrides) -> str:
    if OnPolicyRunner is None:
        raise SystemExit("training needs torch: see the Setup section of README.md")

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{arm}_s{seed}"
    log_path = os.path.join(out_dir, f"{tag}.jsonl")
    env = make_env(env_name, episode_steps)
    config = MAPPOConfig.for_arm(arm, seed=seed, total_steps=steps, **overrides)

    with open(log_path, "w") as log_file:
        # first line: the full config; then one line per PPO update
        log_file.write(json.dumps({"config": config.to_dict(), "env": env_name}) + "\n")

        def log(stats):
            log_file.write(json.dumps(stats) + "\n")
            log_file.flush()
            print(f"[{tag}] update {stats['update']:4d} steps {stats['env_steps']:7d} "
                  f"ret {stats['ep_return_mean']} sps {stats['steps_per_sec']:.0f}")

        start = time.time()
        runner = OnPolicyRunner(env, config, logger=log)
        runner.run()

    checkpoint_path = os.path.join(out_dir, f"{tag}.pt")
    runner.save(checkpoint_path)
    print(f"[{tag}] done in {time.time() - start:.0f}s -> {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), default="M0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--env", choices=["toy", "citylearn"], default="toy")
    parser.add_argument("--episode-steps", type=int, default=168)
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--out", default="runs")
    args = parser.parse_args()

    run(args.arm, args.seed, args.steps, args.env, args.episode_steps, args.out,
        rollout_steps=args.rollout_steps)


if __name__ == "__main__":
    main()
