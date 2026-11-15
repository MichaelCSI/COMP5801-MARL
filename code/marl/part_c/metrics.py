
"""One evaluation record -> comparable metrics; no CityLearn/torch dependency.

Use REALIZED battery meter energy, not requested normalized policy actions, for H3.
A constant or missing power trace has undefined Pearson correlation and is excluded;
we report how many of the n(n-1)/2 correlations were estimable.
"""
from __future__ import annotations

import numpy as np


def battery_synchrony(battery_energy, eps: float = 1e-10) -> dict:
    if battery_energy is None:
        return dict(synchrony=float('nan'), mean_abs_correlation=float('nan'),
                    valid_pairs=0, total_pairs=0)
    power = np.asarray(battery_energy, dtype=np.float64)
    if power.ndim != 2 or power.shape[1] < 2:
        raise ValueError('battery_energy must be a (time, n_agents) matrix')
    n = power.shape[1]
    coeffs = []
    for i in range(n):
        for j in range(i + 1, n):
            x, y = power[:, i], power[:, j]
            good = np.isfinite(x) & np.isfinite(y)
            if good.sum() < 3:
                continue
            x, y = x[good], y[good]
            if x.std() <= eps or y.std() <= eps:
                continue
            coeffs.append(float(np.corrcoef(x, y)[0, 1]))
    return dict(synchrony=float(np.mean(coeffs)) if coeffs else float('nan'),
                mean_abs_correlation=float(np.mean(np.abs(coeffs))) if coeffs else float('nan'),
                valid_pairs=len(coeffs), total_pairs=n * (n - 1) // 2)


def summarise_record(record: dict, *, arm: str, condition: str, train_seed: int,
                     eval_env_seed: int, channel_seed: int, checkpoint_sha256: str,
                     split: str) -> dict:
    """Summarize one test replicate. Requires env.kpis() callback to be supplied."""
    k = record.get('kpis')
    if not isinstance(k, dict) or 'primary' not in k or 'avg_daily_peak' not in k:
        raise ValueError('evaluation record needs DistrictBatteryEnv.kpis(); use '
                         'on_episode_end=lambda e: e.kpis()')
    actions = np.asarray(record['actions'])
    if actions.ndim != 2:
        raise ValueError('actions must have shape (time, n_agents)')
    delivered = record.get('delivered')
    if delivered is None:
        realized = float('nan')
    else:
        delivered = np.asarray(delivered)
        expected_shape = (actions.shape[0], actions.shape[1], actions.shape[1])
        if delivered.shape != expected_shape:
            raise ValueError(f'delivered has shape {delivered.shape}, expected {expected_shape}')
        possible = actions.shape[0] * actions.shape[1] * (actions.shape[1] - 1)
        realized = float(np.sum(delivered) / possible)

    sync = battery_synchrony(record.get('battery_energy'))
    return {
        'split': split, 'arm': arm, 'condition': condition, 'seed': int(train_seed),
        'replicate': int(record['replicate']), 'p': (
            float(record['p']) if record.get('p') is not None else float('nan')),
        'hold_last': bool(record.get('hold_last', False)),
        'force_mask': bool(record.get('force_mask', False)),
        'env_seed': int(eval_env_seed), 'channel_seed': int(channel_seed),
        'checkpoint_sha256': checkpoint_sha256,
        'steps': int(actions.shape[0]), 'n_agents': int(actions.shape[1]),
        'realized_delivery': realized,
        'primary': float(k['primary']),
        'avg_daily_peak': float(k['avg_daily_peak']),
        'cost': float(k['cost']), 'cost_rel_r0': float(k['cost_rel_r0']),
        'return': float(k['return']),
        'outage_hours': float(k.get('outage_hours', float('nan'))),
        **sync,
    }
