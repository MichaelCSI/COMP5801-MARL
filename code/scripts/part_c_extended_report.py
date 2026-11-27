import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path('results/part_c')
CONDITIONS = {'M0':'M0', 'M1':'M1 p=1.0', 'M1-RND':'M1-RND p=1.0', 'I0':'I0', 'M-FULL':'M-FULL'}
ARMS = list(CONDITIONS)
SEEDS = set(range(5))
METRICS = ['primary', 'cost_rel_r0', 'synchrony', 'mean_abs_correlation']
COMPARISONS = [('I0','M0'), ('M-FULL','M0'), ('M-FULL','M1')]

def interval(values):
    x = np.asarray(values, dtype=float)
    assert len(x) == 5 and np.isfinite(x).all()
    m = x.mean()
    h = t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m-h, m+h

def load(split):
    old = pd.read_csv(ROOT / f'{split}_combined/replicates.csv')
    new = pd.read_csv(ROOT / f'extended/{split}/replicates.csv')
    assert len(old) == 950 and len(new) == 100, 'Unexpected input counts'
    for d in [old,new]:
        assert set(d['split']) == {split}
        assert not d.duplicated(['split','arm','condition','seed','replicate']).any()
    d = pd.concat([old,new], ignore_index=True)
    d = d[d.apply(lambda r: CONDITIONS.get(r.arm) == r.condition, axis=1)].copy()
    assert len(d) == 250
    assert set(d['env_seed']) == ({7} if split=='validation' else {0})
    assert set(d['steps']) == {359} and set(d['n_agents']) == {6}
    for arm, g in d.groupby('arm'):
        assert set(g.seed) == SEEDS and len(g) == 50, arm
        for _, h in g.groupby('seed'):
            assert set(h.replicate) == set(range(10)) and len(h) == 10
            assert h.checkpoint_sha256.nunique() == 1
    assert set(d.arm) == set(ARMS)
    assert np.isfinite(d[METRICS].to_numpy()).all()
    return d.groupby(['arm','seed'], as_index=False)[METRICS].mean()

def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)

def report(split):
    out = ROOT / f'extended/{split}/analysis'
    d = load(split); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out/'per_seed.csv', index=False)
    summary = d.groupby('arm')[METRICS].mean().reindex(ARMS)
    summary['peak_reduction_pct'] = 100*(1-summary.primary)
    summary.to_csv(out/'summary_by_arm.csv')
    rows = []
    for a,b in COMPARISONS:
        x = d[d.arm==a].set_index('seed').sort_index()
        y = d[d.arm==b].set_index('seed').sort_index()
        for metric in METRICS:
            if metric in ('primary','cost_rel_r0'):
                delta = 100*(y[metric]-x[metric])/y[metric]
                unit = 'percentage_improvement'
            else:
                delta = y[metric]-x[metric]
                unit = 'correlation_difference'
            m,l,h = interval(delta)
            rows.append(dict(arm=a, reference=b, metric=metric, unit=unit,
                             n=5, mean=m, ci_low=l, ci_high=h,
                             ci_excludes_zero=(l>0 or h<0)))
    paired = pd.DataFrame(rows)
    paired.to_csv(out/'paired_comparisons.csv', index=False)
    # Absolute KPI comparison; seed means and marginal pointwise t intervals.
    fig, ax = plt.subplots(figsize=(8,4.5))
    for i,a in enumerate(ARMS):
        g=d[d.arm==a].primary.to_numpy(); m,l,h=interval(g)
        ax.scatter(np.full(5,i),g,alpha=.45)
        ax.errorbar(i,m,yerr=[[m-l],[h-m]],fmt='o',capsize=5,color='black')
    ax.axhline(1,ls='--',color='gray',label='R0')
    ax.set(xticks=range(5),xticklabels=ARMS,ylabel='Peak relative to R0 (lower is better)',
           title=f'Five-controller comparison — {split} (95% seed-level CI)')
    ax.legend(); ax.grid(axis='y',alpha=.2)
    save(fig,out/'01_controller_comparison.png')
    # Paired improvement plot.
    q=paired[paired.metric=='primary'].reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(8,4))
    for i,r in q.iterrows():
        ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],
                    fmt='o',capsize=5,color='black')
    ax.axvline(0,ls='--',color='gray')
    ax.set(yticks=range(len(q)),yticklabels=[f'{r.arm} vs {r.reference}' for r in q.itertuples()],
           xlabel='Paired improvement (%) — positive is better',title=f'Exploratory comparisons — {split} (95% CI)')
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    save(fig,out/'02_paired_comparisons.png')
    # Peak/cost trade-off using individual training seeds.
    fig,ax=plt.subplots(figsize=(7,5))
    for a in ARMS:
        g=d[d.arm==a]; sc=ax.scatter(g.primary,g.cost_rel_r0,alpha=.6,label=a)
        ax.scatter(g.primary.mean(),g.cost_rel_r0.mean(),marker='*',s=170,color=sc.get_facecolors()[0])
    ax.set(xlabel='Peak relative to R0',ylabel='Cost relative to R0',title=f'Peak–cost trade-off — {split}')
    ax.legend(); ax.grid(alpha=.2)
    save(fig,out/'03_peak_cost_tradeoff.png')
    # Signed and absolute correlations shown side by side.
    fig,ax=plt.subplots(figsize=(8,4.5))
    for metric, offset,label in [('synchrony',-.13,'Signed r'),('mean_abs_correlation',.13,'Absolute |r|')]:
        stats=[interval(d[d.arm==a][metric]) for a in ARMS]
        m,l,h=np.array(stats).T
        ax.errorbar(np.arange(5)+offset,m,yerr=[m-l,h-m],fmt='o',capsize=4,label=label)
    ax.axhline(0,ls='--',color='gray');ax.set(xticks=range(5),xticklabels=ARMS,
        ylabel='Mean pairwise correlation',title=f'Battery-energy coupling — {split} (95% CI)')
    ax.legend();ax.grid(axis='y',alpha=.2)
    save(fig,out/'04_battery_correlation.png')
    print(summary.round(5).to_string())
    print('\nPaired primary comparisons (unadjusted exploratory intervals):')
    print(q.round(5).to_string(index=False));print(f'Wrote {out}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    report(p.parse_args().split)
