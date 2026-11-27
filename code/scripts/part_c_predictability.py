from __future__ import annotations
import torch

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results/part_c/validation/peer_soc_regression.csv'))
    p.add_argument('--alpha', type=float, default=1.)
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error(f'missing checkpoint: {args.checkpoint}')

    from marl.envs.district_env import DistrictBatteryEnv, VAL_SEED
    from marl.envs.features import collect, peer_soc_dataset
    from marl.mappo.runner import OnPolicyRunner

    env = DistrictBatteryEnv(split='validation')
    runner = OnPolicyRunner.load(str(args.checkpoint), env)
    if runner.config.arm != 'M0':
        raise ValueError('peer-SoC predictability diagnostic requires a no-message M0 policy')
    obs, soc = collect(env, runner.as_controller(), seed=VAL_SEED)
    datasets = peer_soc_dataset(obs, soc, env.observation_names)
    rows = []
    for agent, data in enumerate(datasets):
        X = np.asarray(data['X'], dtype=float)
        Y = np.asarray(data['Y'], dtype=float)
        split = int(.7 * len(X))
        if split < 3 or len(X) - split < 3:
            raise ValueError('not enough observations for chronological split')
        for column, peer in enumerate(data['peers']):
            y = Y[:, column]
            model = make_pipeline(StandardScaler(), Ridge(alpha=args.alpha))
            model.fit(X[:split], y[:split])
            prediction = model.predict(X[split:])
            rows.append(dict(receiver=agent, peer=int(peer), n_train=split,
                             n_test=len(X)-split, test_r2=float(r2_score(y[split:], prediction)),
                             test_mae=float(mean_absolute_error(y[split:], prediction)),
                             test_soc_sd=float(y[split:].std())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} directed peer regressions: {args.out}')
    print('CAUTION: high R2 is predictive association, not proof that messages are useless.')


if __name__ == '__main__':
    main()
