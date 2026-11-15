
"""Paired training-seed inference for the project (not replicate-level t-tests).

With fixed evaluation outage seed, independent channel-drop replicates for a
single trained network are repeated measurements, NOT independent network seeds.
Each (arm, condition, seed) is first averaged over evaluation replicates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t

METRICS = ('primary', 'avg_daily_peak', 'cost', 'cost_rel_r0',
           'return', 'synchrony', 'mean_abs_correlation', 'realized_delivery')


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError('no evaluation records')
    required = {'arm', 'condition', 'seed', 'replicate', *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'missing columns: {sorted(missing)}')
    if df.duplicated(['arm', 'condition', 'seed', 'replicate']).any():
        raise ValueError('duplicate replicate keys; do not combine incompatible runs')
    keys = ['arm', 'condition', 'seed', 'p', 'hold_last', 'force_mask']
    return df.groupby(keys, dropna=False, sort=True).agg(
        **{metric: (metric, 'mean') for metric in METRICS},
        replicates=('replicate', 'nunique'),
        valid_pairs=('valid_pairs', 'mean'),
        steps=('steps', 'min'),
    ).reset_index()


def paired_interval(differences, alpha: float = 0.05) -> dict:
    x = np.asarray(differences, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=float('nan'), sd=float('nan'),
                    ci_low=float('nan'), ci_high=float('nan'))
    mean = float(x.mean())
    if len(x) < 2:
        return dict(n=int(len(x)), mean=mean, sd=float('nan'),
                    ci_low=float('nan'), ci_high=float('nan'))
    sd = float(x.std(ddof=1))
    half = float(t.ppf(1 - alpha / 2, len(x) - 1) * sd / np.sqrt(len(x)))
    return dict(n=int(len(x)), mean=mean, sd=sd,
                ci_low=mean - half, ci_high=mean + half)


def _pair(base: pd.DataFrame, candidate: pd.DataFrame, metric: str) -> pd.DataFrame:
    left = base[['seed', metric]].rename(columns={metric: 'reference'})
    right = candidate[['seed', metric]].rename(columns={metric: 'candidate'})
    if left['seed'].duplicated().any() or right['seed'].duplicated().any():
        raise ValueError('each condition must contain only one value per training seed')
    return left.merge(right, on='seed', how='inner', validate='one_to_one')


def paired_comparisons(seed_df: pd.DataFrame, *, baseline: str = 'M0',
                       delta_pct: float = 5.0) -> pd.DataFrame:
    """Improvement = 100*(baseline - treatment)/baseline, positive = better.

    CI is computed over the SEED-SPECIFIC percentage improvements, after each
    seed's evaluation replicates are averaged. Do not pool 5x10 replicates.
    """
    base = seed_df[seed_df['condition'] == baseline]
    if len(base) == 0:
        return pd.DataFrame()
    rows = []
    for cond in sorted(seed_df.condition.unique()):
        if cond == baseline:
            continue
        candidate = seed_df[seed_df['condition'] == cond]
        for metric in ('primary', 'cost_rel_r0', 'synchrony'):
            paired = _pair(base, candidate, metric)
            if paired.empty:
                continue
            if metric == 'synchrony':
                # Difference of correlation coefficients, not a percent change;
                # positive means lower signed synchronization than M0.
                effect = paired['reference'] - paired['candidate']
                unit = 'correlation_difference'
            else:
                nonzero = paired['reference'].abs() > 1e-12
                paired = paired[nonzero]
                effect = 100 * (paired['reference'] - paired['candidate']) / paired['reference']
                unit = 'percentage_improvement'
            ci = paired_interval(effect)
            rows.append(dict(arm=str(candidate['arm'].iloc[0]), condition=cond, metric=metric,
                             unit=unit, **ci,
                             delta_pct=(delta_pct if metric == 'primary' else float('nan')),
                             mean_ge_delta=(ci['mean'] >= delta_pct if metric == 'primary' else None),
                             ci_excludes_zero=(ci['ci_low'] > 0 or ci['ci_high'] < 0)
                             if ci['n'] >= 2 else None,
                             ci_above_delta=(ci['ci_low'] >= delta_pct)
                             if metric == 'primary' and ci['n'] >= 2 else None,
                             ci_below_delta=(ci['ci_high'] < delta_pct)
                             if metric == 'primary' and ci['n'] >= 2 else None))
    return pd.DataFrame(rows)


def robustness_table(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Degradation at p vs same seed's p=1, in % of its p=1 primary KPI.

    A *positive* number means deterioration as delivery decreases.
    Excludes hold-last and forced-mask diagnostics; report these separately.
    """
    base = seed_df[(seed_df.p == 1.0) & ~seed_df.hold_last & ~seed_df.force_mask]
    rows = []
    for arm in ('M1', 'M1-RND', 'M1-RND-NM'):
        at_one = base[base.arm == arm]
        if at_one.empty:
            continue
        for p in sorted(seed_df.loc[(seed_df.arm == arm) & ~seed_df.hold_last &
                                     ~seed_df.force_mask, 'p'].dropna().unique()):
            current = seed_df[(seed_df.arm == arm) & (seed_df.p == p) &
                              ~seed_df.hold_last & ~seed_df.force_mask]
            paired = _pair(at_one, current, 'primary')
            if paired.empty:
                continue
            paired = paired[paired['reference'].abs() > 1e-12]
            loss = 100 * (paired.candidate - paired.reference) / paired.reference
            ci = paired_interval(loss)
            rows.append(dict(arm=arm, p=float(p), **ci))
    return pd.DataFrame(rows)


def robustness_arm_comparison(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Paired H2 comparison: M1's loss(p) minus M1-RND's loss(p).

    Positive is evidence that random-loss training is *less* fragile; negatives
    indicate M1-RND degrades more. Retains within-seed pairing across arms.
    """
    standard = seed_df[~seed_df.hold_last & ~seed_df.force_mask]
    losses = {}
    for arm in ('M1', 'M1-RND'):
        sub = standard[standard.arm == arm]
        reference = sub[sub.p == 1.0]
        if reference.empty:
            continue
        for p in sorted(sub.p.dropna().unique()):
            at_p = sub[sub.p == p]
            paired = _pair(reference, at_p, 'primary')
            paired = paired[paired['reference'].abs() > 1e-12]
            if len(paired):
                losses[arm, float(p)] = paired.assign(
                    loss_pct=100 * (paired.candidate - paired.reference) / paired.reference
                )[['seed', 'loss_pct']]
    rows = []
    for p in sorted({p for arm, p in losses if arm == 'M1'} &
                    {p for arm, p in losses if arm == 'M1-RND'}):
        left = losses['M1', p].rename(columns={'loss_pct': 'm1_loss'})
        right = losses['M1-RND', p].rename(columns={'loss_pct': 'rnd_loss'})
        aligned = left.merge(right, on='seed', validate='one_to_one')
        interval = paired_interval(aligned.m1_loss - aligned.rnd_loss)
        rows.append(dict(p=p, **interval))
    return pd.DataFrame(rows)
