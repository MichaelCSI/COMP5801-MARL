from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/part_c/predictability'))
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.input
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    expected = {(i, j) for i in range(6) for j in range(6) if i != j}
    
    for seed in range(5):
        df = pd.read_csv(args.input / f'M0_s{seed}.csv')
        assert len(df) == 30 and set(zip(df.receiver, df.peer)) == expected
        assert (df.n_train == 251).all() and (df.n_test == 108).all()
        assert np.isfinite(df[['test_r2','test_mae','test_soc_sd']]).all().all()
        frames.append(df.assign(seed=seed))
        
    data = pd.concat(frames, ignore_index=True)
    summary = data.groupby('seed').agg(
        mean_r2=('test_r2','mean'), median_r2=('test_r2','median'),
        mean_mae=('test_mae','mean'), mean_soc_sd=('test_soc_sd','mean'),
        negative_r2=('test_r2',lambda x: (x < 0).sum()),
        r2_at_least_05=('test_r2',lambda x: (x >= .5).sum()))
    
    summary.to_csv(out / 'predictability_summary_by_seed.csv')
    print(summary.round(4).to_string())
    print('\nOverall mean R2:', data.test_r2.mean())
    print('Overall mean MAE:', data.test_mae.mean())
    print('Negative R2:', (data.test_r2 < 0).sum(), '/', len(data))
    matrix = data.pivot_table(index='receiver',columns='peer',values='test_r2').reindex(index=range(6),columns=range(6))
    matrix.to_csv(out / 'predictability_pair_mean_r2.csv')

    fig, ax = plt.subplots(figsize=(7.2, 6))
    bound = float(np.nanmax(np.abs(matrix.to_numpy())))
    im = ax.imshow(matrix.to_numpy(), cmap='RdBu', norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
    
    for i in range(6):
        for j in range(6):
            value = matrix.iloc[i,j]
            ax.text(j,i,'—' if i==j else f'{value:.2f}',ha='center',va='center',fontsize=11, color='white' if np.isfinite(value) and abs(value) > bound*.65 else 'black')
            
    ax.set(xticks=range(6), yticks=range(6), xticklabels=range(1,7),yticklabels=range(1,7),xlabel='Peer building (prediction target)',ylabel='Receiving building (local observation)',title='Peer SoC predictability: mean R² across five M0 seeds')
    fig.colorbar(im, ax=ax, label='Mean evaluation R²')
    fig.tight_layout()
    fig.savefig(out / '01_peer_soc_r2_heatmap.png',dpi=300,bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    values = [data.loc[data.seed==s,'test_r2'].to_numpy() for s in range(5)]
    ax.boxplot(values, positions=range(5), widths=.5, showfliers=False)
    rng = np.random.default_rng(0)
    
    for s, v in enumerate(values):
        ax.scatter(s+rng.uniform(-.13,.13,len(v)),v,s=17,alpha=.55,label='Directed peer pair' if s==0 else None)
        
    ax.scatter(range(5),summary.mean_r2,marker='D',s=48,label='Seed mean R²',zorder=4)
    ax.axhline(0,linestyle='--',color='gray',linewidth=1,label='R² = 0')
    ax.set(xticks=range(5),xlabel='Independent M0 training seed',ylabel='Evaluation R²',title='Peer-SoC predictability across training seeds')
    ax.grid(axis='y',alpha=.2)
    ax.legend(loc='lower left',fontsize=9)
    fig.tight_layout()
    fig.savefig(out / '02_peer_soc_r2_distribution.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print('\nSaved summaries and plots to',out)

if __name__ == '__main__':
    main()
