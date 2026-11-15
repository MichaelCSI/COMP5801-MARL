
"""Part C: load trained checkpoints and evaluate the proposal's delivery sweep.

From the repository's `code` directory:
    python -m scripts.part_c_experiments --split validation --arms M0 M1 --dry-run
    python -m scripts.part_c_experiments --split validation --arms M0 M1
    python -m scripts.part_c_experiments --split test --final-test --arms M0 M1 M1-RND

The test split requires --final-test to reduce accidental holdout reuse.
Only training seeds are independent experimental units; replicates sample drops.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from marl.part_c.metrics import summarise_record

FIELDS = [
    'split', 'arm', 'condition', 'seed', 'replicate', 'p', 'hold_last',
    'force_mask', 'env_seed', 'channel_seed', 'checkpoint_sha256', 'steps',
    'n_agents', 'realized_delivery', 'primary', 'avg_daily_peak', 'cost',
    'cost_rel_r0', 'return', 'outage_hours', 'synchrony',
    'mean_abs_correlation', 'valid_pairs', 'total_pairs', 'trajectory_file',
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_label(label: str) -> str:
    slug = re.sub('[^a-zA-Z0-9_-]+', '_', label).strip('_')
    return slug + '_' + hashlib.sha1(label.encode()).hexdigest()[:8]


def save_trajectory(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = record['infos']
    def numeric_info(name, dtype=float):
        return np.asarray([d.get(name, np.nan) for d in info], dtype=dtype)

    np.savez_compressed(
        path,
        actions=np.asarray(record['actions']),
        battery_energy=(np.asarray(record['battery_energy']) if record['battery_energy'] is not None
                        else np.empty((0,), dtype=np.float32)),
        soc=np.asarray(record['soc']),
        delivered=(np.asarray(record['delivered']) if record['delivered'] is not None
                   else np.empty((0,), dtype=np.bool_)),
        rewards=np.asarray(record['rewards']),
        district_load=numeric_info('district_load'),
        price=numeric_info('price'),
        dataset_step=numeric_info('dataset_step'),
    )


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--final-test', action='store_true',
                        help='explicitly authorize held-out test evaluation')
    parser.add_argument('--arms', nargs='+', default=['M0', 'M1'],
                        help='default is the D1 comparison; add M1-RND after D1')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--replicates', type=int, default=10)
    parser.add_argument('--checkpoints', type=Path, default=Path('runs'))
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='skip already completed conditions with matching checkpoint hash')
    parser.add_argument('--no-trajectories', action='store_true')
    args = parser.parse_args()

    # Imports delayed so --help and standalone analysis work without CityLearn.
    from marl.mappo import ARMS, OnPolicyRunner, evaluation_conditions
    from marl.mappo.config import STREAM_EVAL, stream

    if args.split == 'test' and not args.final_test:
        parser.error('test is the held-out block: pass --final-test only after validation decisions')
    if args.replicates < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('replicates must be positive and training seeds must be unique')
    if any(arm not in ARMS for arm in args.arms):
        parser.error(f'unknown arm, available: {sorted(ARMS)}')
    if len(set(args.arms)) != len(args.arms):
        parser.error('arms must be unique')

    output = args.out or Path('results') / 'part_c' / args.split
    jobs = []
    for arm in args.arms:
        for seed in args.seeds:
            checkpoint = args.checkpoints / f'{arm}_s{seed}.pt'
            jobs.append((arm, seed, checkpoint, evaluation_conditions(arm)))

    print(f'Split: {args.split}; {args.replicates} replicates/condition; '
          f'{len(jobs)} trained checkpoints; {sum(len(conditions) for _, _, _, conditions in jobs)} conditions')
    for arm, seed, ckpt, conds in jobs:
        print(f'  {arm} seed={seed}: {ckpt}; {len(conds)} conditions')
    if args.dry_run:
        return

    missing = [str(path) for _, _, path, _ in jobs if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing trained checkpoints (run training first):\n' + '\n'.join(missing))

    path = output / 'replicates.csv'
    existing = read_existing(path)
    if existing and not args.resume:
        raise FileExistsError(f'{path} exists; use --resume only for the same experiment')

    from marl.envs.district_env import DistrictBatteryEnv, EVAL_SEEDS
    env_seed = int(EVAL_SEEDS[args.split])
    for arm, train_seed, checkpoint, conditions in jobs:
        digest = sha256(checkpoint)
        channel_seed = int(stream(train_seed, STREAM_EVAL).generate_state(1)[0])
        # Check once for any prior rows with this arm/seed: refuse changed weights.
        prior = [r for r in existing if r['arm'] == arm and int(r['seed']) == train_seed]
        if prior and any(r['checkpoint_sha256'] != digest or r['split'] != args.split or
                         int(r['env_seed']) != env_seed for r in prior):
            raise ValueError(f'cannot resume: checkpoint or eval config changed for {arm} seed {train_seed}')

        pending = []
        for condition, opts in conditions:
            matching = [r for r in prior if r['condition'] == condition]
            replicates = {int(r['replicate']) for r in matching}
            if matching:
                if len(matching) != args.replicates or replicates != set(range(args.replicates)):
                    raise ValueError(f'incomplete prior condition {condition!r} seed={train_seed}; '
                                     'move aside output folder and restart instead of silently mixing')
                print(f'SKIP {arm} seed={train_seed}: {condition} (already complete)')
            else:
                pending.append((condition, opts))
        if not pending:
            continue
        env = DistrictBatteryEnv(split=args.split)
        runner = OnPolicyRunner.load(str(checkpoint), env)
        if runner.config.arm != arm or runner.config.seed != train_seed:
            raise ValueError(f'checkpoint {checkpoint} contains {runner.config.arm} '
                             f'seed {runner.config.seed}, expected {arm} seed {train_seed}')
        for condition, opts in pending:
            records = runner.evaluate(
                env, n_replicates=args.replicates, env_seed=env_seed,
                channel_seed=channel_seed, deterministic=True,
                on_episode_end=lambda e: e.kpis(), **opts)
            rows = []
            for record in records:
                row = summarise_record(record, arm=arm, condition=condition,
                                       train_seed=train_seed, eval_env_seed=env_seed,
                                       channel_seed=channel_seed,
                                       checkpoint_sha256=digest, split=args.split)
                row['trajectory_file'] = ''
                if not args.no_trajectories:
                    target = output / 'trajectories' / (
                        f'{arm}_s{train_seed}_{safe_label(condition)}_r{record["replicate"]:02d}.npz')
                    save_trajectory(target, record)
                    row['trajectory_file'] = str(target)
                rows.append(row)
            append_rows(path, rows)
            print(f'DONE {arm} seed={train_seed} {condition}: '
                  f'primary={np.mean([r["primary"] for r in rows]):.4f}; '
                  f'n={len(rows)}; steps={rows[0]["steps"]}')

    manifest = {'split': args.split, 'eval_env_seed': env_seed,
                'training_seeds': args.seeds, 'arms_requested': args.arms,
                'replicates_per_condition': args.replicates,
                'metric_unit': 'peak relative to R0, lower better',
                'deterministic_policy': True,
                'replicates_vary': 'independent channel-drop realizations; same outage seed',
                'important': 'Do not perform t-tests across drop replicates; pair by training seed'}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
