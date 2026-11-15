
"""Part C: per-seed inference, delivery/robustness/synchrony plots, and H1/H2/H3 diagnostics.

    python -m scripts.part_c_report --split validation
    python -m scripts.part_c_report --split test

All inference is seed-paired; no replicate pseudo-replication.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from marl.part_c.statistics import (paired_comparisons, paired_interval,
                                    robustness_table, robustness_arm_comparison, seed_summary)


def by_seed_ci(values) -> tuple[float, float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (float('nan'), float('nan'), float('nan'))
    mean = float(x.mean())
    if len(x) < 2:
        return mean, float('nan'), float('nan')
    half = float(t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))
    return mean, mean-half, mean+half


def get_comparison(df, arm, p, metric='primary'):
    if df.empty:
        return None
    label = f'{arm} p={p}'
    subset = df[(df.condition == label) & (df.metric == metric)]
    return subset.iloc[0].to_dict() if len(subset) else None


def hypothesis_summary(comparisons: pd.DataFrame, robustness: pd.DataFrame,
                       robust_difference: pd.DataFrame, delta_pct: float) -> dict:
    """Interpretive checks, not automatic claims of hypothesis acceptance."""
    h1 = get_comparison(comparisons, 'M1', 1.0)
    message_use = [get_comparison(comparisons, 'M1-RND', p) for p in (1.0, 0.9)]
    message_use = [r for r in message_use if r is not None]
    rnd_high = next((r for r in message_use if np.isfinite(r['ci_low']) and
                     r['mean'] >= delta_pct and r['ci_low'] > 0), None)
    fragility = [get_comparison(comparisons, 'M1', p) for p in (0.5, 0.25, 0.0)]
    fragility = [r for r in fragility if r is not None]
    comparison_half = get_comparison(comparisons, 'M1', 0.5)
    h3 = get_comparison(comparisons, 'M1', 1.0, metric='synchrony')
    rnd_threshold = []
    for p in (1.0, 0.9, 0.5, 0.25, 0.0):
        r = get_comparison(comparisons, 'M1-RND', p)
        if r:
            rnd_threshold.append(dict(p=p, mean_improvement_pct=r['mean'],
                ci_low=r['ci_low'], ci_high=r['ci_high'],
                mean_within_delta_of_M0=(r['mean'] >= -delta_pct),
                ci_supports_within_delta=(r['ci_low'] >= -delta_pct)))

    pilot_sd = (float(h1['sd']) if h1 is not None and np.isfinite(h1['sd']) else float('nan'))
    notes = {
        'delta_pct': delta_pct,
        'pilot_precision': {'M1_p1_vs_M0_paired_seed_sd_pct': pilot_sd,
            'above_professor_4pct_warning': bool(np.isfinite(pilot_sd) and pilot_sd > 4.),
            'advice': 'If pilot paired-seed SD exceeds roughly 4 percentage points, consider more M0/M1 seeds before optional arms.'},
        'H1_message_benefit': {
            'status': ('not_evaluated' if h1 is None else
                       'null_bound_below_delta' if h1['ci_high'] < delta_pct else
                       'observed_mean_ge_delta_and_ci_excludes_zero' if
                       h1['mean'] >= delta_pct and h1['ci_low'] > 0 else 'inconclusive'),
            'M1_p1_vs_M0': h1,
            'caution': 'A confidence interval excluding zero is weaker than its entire range exceeding delta.'
        },
        'H2_fragility': {
            'status': 'not_evaluated' if comparison_half is None else
                      'M1_advantage_remains_confidently_ge_delta_at_p05' if
                      comparison_half['ci_low'] >= delta_pct else
                      'M1_advantage_confidently_below_delta_at_p05' if
                      comparison_half['ci_high'] < delta_pct else 'inconclusive',
            'M1_loss_evidence': fragility,
            'caution': 'The pre-specified fragility criterion concerns M1 vs M0 at p<=0.5.'
        },
        'H2_robust_training': {
            'high_delivery_M1_RND_benefit': 'evidence' if rnd_high is not None else
                                        ('inconclusive' if message_use else 'not_evaluated'),
            'comparisons_at_p1_p09': message_use,
            'robustness_by_delivery': (robustness.to_dict('records') if not robustness.empty else []),
            'M1_minus_M1_RND_loss_pct': (robust_difference.to_dict('records')
                                         if not robust_difference.empty else []),
            'M1_RND_vs_M0_at_every_delivery': rnd_threshold,
            'caution': 'Low relative loss alone does NOT establish message use. '
                       'M1-RND must also provide a meaningful benefit vs M0 at high delivery.'
        },
        'H3_exploratory_synchrony': {
            'M1_p1_vs_M0_signed_correlation_difference': h3,
            'caution': 'Interpret with valid-pair counts; report |r| to detect anti-phase/strong coupling.'
        },
    }
    return notes


def write_delivery_plot(seed_df: pd.DataFrame, target: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    core = seed_df[(seed_df.arm.isin(['M1', 'M1-RND'])) &
                   (seed_df.hold_last == False) & (seed_df.force_mask == False)]
    for arm in ('M1', 'M1-RND'):
        group = core[core.arm == arm]
        if group.empty:
            continue
        xs, means, lower, upper = [], [], [], []
        for p, g in group.groupby('p'):
            m, lo, hi = by_seed_ci(g.primary)
            xs.append(float(p)); means.append(m); lower.append(lo); upper.append(hi)
        ax.plot(xs, means, marker='o', label=arm)
        if all(np.isfinite(lower)) and all(np.isfinite(upper)):
            ax.fill_between(xs, lower, upper, alpha=.15)
    baseline = seed_df[seed_df.condition == 'M0']
    if len(baseline):
        mean, lo, hi = by_seed_ci(baseline.primary)
        ax.axhline(mean, linestyle='--', label='M0 (no messages)')
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axhspan(lo, hi, alpha=.08)
    ax.set(xlabel='Message delivery probability p',
           ylabel='Average daily district peak / R0 (lower is better)',
           title='Delivery sweep: independently trained seeds')
    ax.set_xlim(-.025, 1.025)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def write_robustness_plot(robustness: pd.DataFrame, target: Path) -> None:
    if robustness.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    for arm, group in robustness.groupby('arm'):
        g = group.sort_values('p')
        ax.plot(g.p, g['mean'], marker='o', label=arm)
        if g.ci_low.notna().all() and g.ci_high.notna().all():
            ax.fill_between(g.p.to_numpy(dtype=float),
                            g.ci_low.to_numpy(dtype=float), g.ci_high.to_numpy(dtype=float), alpha=.15)
    ax.axhline(0, linestyle=':', linewidth=1)
    ax.set(xlabel='Message delivery probability p',
           ylabel='Primary-KPI loss vs own p=1 (%)',
           title='Robustness (positive = worse than perfect delivery)')
    ax.set_xlim(-.025, 1.025)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def write_synchrony_plot(seed_df: pd.DataFrame, target: Path) -> None:
    core = seed_df[(seed_df.condition.isin(['M0', 'M1 p=1.0', 'M1-RND p=1.0']))]
    if core.empty or core.synchrony.notna().sum() == 0:
        return
    order = ['M0', 'M1 p=1.0', 'M1-RND p=1.0']
    labels, means, errors = [], [], []
    for label in order:
        group = core[(core.condition == label) & core.synchrony.notna()]
        if group.empty:
            continue
        mean, low, high = by_seed_ci(group.synchrony)
        labels.append(label)
        means.append(mean)
        errors.append((high - mean) if np.isfinite(high) else 0)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.bar(labels, means, yerr=errors, capsize=4)
    ax.axhline(0, linestyle=':', linewidth=1)
    ax.set(ylabel='Mean pairwise Pearson r (realized battery energy)',
           title='Battery-power synchrony across trained seeds')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def write_extension_plot(seed_df: pd.DataFrame, target: Path) -> None:
    """Feedback checks: hold-last M1 and forced-mask M1-RND vs standard."""
    selections = [
        ('M1', False, False, 'M1 default'),
        ('M1', True, False, 'M1 hold-last'),
        ('M1-RND', False, False, 'M1-RND default'),
        ('M1-RND', False, True, 'M1-RND forced mask'),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    drew = 0
    for arm, hold, force, label in selections:
        g = seed_df[(seed_df.arm == arm) & (seed_df.hold_last == hold)
                    & (seed_df.force_mask == force)]
        if g.empty:
            continue
        values = [(float(p), by_seed_ci(s.primary)[0]) for p, s in g.groupby('p')]
        if not values:
            continue
        values.sort()
        ax.plot([x for x, _ in values], [y for _, y in values],
                marker='o', linestyle='--' if hold or force else '-', label=label)
        drew += 1
    if not drew:
        plt.close(fig)
        return
    base = seed_df[seed_df.condition == 'M0']
    if len(base):
        ax.axhline(float(base.primary.mean()), linestyle=':', label='M0')
    ax.set(xlabel='Message delivery probability p',
           ylabel='Average daily district peak / R0 (lower is better)',
           title='Evaluation-only message-recovery and mask checks')
    ax.set_xlim(-.025, 1.025)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--input', type=Path, default=None)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--delta', type=float, default=5.0, help='minimum meaningful peak improvement, percent')
    args = p.parse_args()
    root = Path('results') / 'part_c' / args.split
    path = args.input or root / 'replicates.csv'
    out = args.out or root / 'analysis'
    if not path.is_file():
        raise FileNotFoundError(f'{path} not found. Run scripts.part_c_experiments first.')
    df = pd.read_csv(path)
    if df['split'].nunique() != 1 or df['split'].iloc[0] != args.split:
        raise ValueError('input contains a different split or mixed splits')
    # Never silently proceed with duplicate runs or nonmatching evaluation seeds.
    if df['env_seed'].nunique() != 1:
        raise ValueError('data mix multiple outage seeds; cannot compare as paired')
    if df['steps'].nunique() != 1:
        raise ValueError('evaluations have different horizon lengths')
    for (arm, seed), g in df.groupby(['arm', 'seed']):
        if g['checkpoint_sha256'].nunique() != 1:
            raise ValueError(f'mixed checkpoint versions for {arm}, seed {seed}')

    per_seed = seed_summary(df)
    comparisons = paired_comparisons(per_seed, delta_pct=args.delta)
    robustness = robustness_table(per_seed)
    robust_difference = robustness_arm_comparison(per_seed)
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / 'per_seed.csv', index=False)
    comparisons.to_csv(out / 'paired_vs_M0.csv', index=False)
    robustness.to_csv(out / 'robustness.csv', index=False)
    robust_difference.to_csv(out / 'robustness_arm_difference.csv', index=False)
    interpretations = hypothesis_summary(comparisons, robustness, robust_difference, args.delta)
    (out / 'hypotheses.json').write_text(json.dumps(interpretations, indent=2,
                                      allow_nan=True) + '\n', encoding='utf-8')
    write_delivery_plot(per_seed, out / 'delivery_sweep.png')
    write_robustness_plot(robustness, out / 'robustness.png')
    write_synchrony_plot(per_seed, out / 'synchrony.png')
    write_extension_plot(per_seed, out / 'message_extensions.png')

    n_seeds = int(per_seed['seed'].nunique())
    print(f'Wrote Part C analysis: {out}; {n_seeds} distinct training seed IDs')
    print('H1:', interpretations['H1_message_benefit']['status'])
    print('H2 high-delivery message use:',
          interpretations['H2_robust_training']['high_delivery_M1_RND_benefit'])
    if n_seeds < 5:
        print('WARNING: fewer than 5 seeds; confidence intervals are preliminary.')
    if interpretations['pilot_precision']['above_professor_4pct_warning']:
        print('WARNING: pilot paired-seed spread >4 percentage points; consider more M0/M1 seeds.')


if __name__ == '__main__':
    main()
