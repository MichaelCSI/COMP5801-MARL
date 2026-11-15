
"""Train reproducibly by (arm, seed) using your team's existing scripts.train.run.

    python -m scripts.part_c_train --arms M0 M1 --dry-run
    python -m scripts.part_c_train --arms M0 M1 --steps 200000
    # Only after decision D1:
    python -m scripts.part_c_train --arms M1-RND --steps 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=['M0', 'M1'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--steps', type=int, default=200_000)
    parser.add_argument('--rollout-steps', type=int, default=1_000)
    parser.add_argument('--episode-steps', type=int, default=168)
    parser.add_argument('--out', type=Path, default=Path('runs'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.steps < args.rollout_steps or args.rollout_steps < 1:
        parser.error('steps must be >= rollout-steps >= 1')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        parser.error('training seeds and arms must be unique')

    from marl.mappo.config import ARMS
    if any(arm not in ARMS for arm in args.arms):
        parser.error(f'arm not found; available: {sorted(ARMS)}')

    for arm in args.arms:
        for seed in args.seeds:
            ckpt = args.out / f'{arm}_s{seed}.pt'
            if ckpt.is_file():
                print(f'SKIP existing checkpoint: {ckpt}')
                continue
            print(f'TRAIN {arm} seed={seed} steps={args.steps} -> {ckpt}')
            if args.dry_run:
                continue
            from scripts.train import run
            run(arm=arm, seed=seed, steps=args.steps, env_name='citylearn',
                episode_steps=args.episode_steps, out_dir=str(args.out),
                rollout_steps=args.rollout_steps)


if __name__ == '__main__':
    main()
