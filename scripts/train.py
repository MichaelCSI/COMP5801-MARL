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
    cfg = MAPPOConfig.for_arm(arm, seed=seed, total_steps=steps, **overrides)
    with open(log_path, "w") as f:
        f.write(json.dumps({"config": cfg.to_dict(), "env": env_name}) + "\n")

        def log(stats):
            f.write(json.dumps(stats) + "\n")
            f.flush()
            print(f"[{tag}] update {stats['update']:4d} steps {stats['env_steps']:7d} "
                  f"ret {stats['ep_return_mean']} sps {stats['steps_per_sec']:.0f}")

        t0 = time.time()
        runner = OnPolicyRunner(env, cfg, logger=log)
        runner.run()
    ck = os.path.join(out_dir, f"{tag}.pt")
    runner.save(ck)
    print(f"[{tag}] done in {time.time() - t0:.0f}s -> {ck}")
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), default="M0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--env", choices=["toy", "citylearn"], default="toy")
    ap.add_argument("--episode-steps", type=int, default=168)
    ap.add_argument("--rollout-steps", type=int, default=1000)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    run(a.arm, a.seed, a.steps, a.env, a.episode_steps, a.out, rollout_steps=a.rollout_steps)


if __name__ == "__main__":
    main()
