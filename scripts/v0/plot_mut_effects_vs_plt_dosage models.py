#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import EJ2_SERIES
        
        
def plot(df, axes, target, max_err, col='M', color='black', alpha=1, label=None):
    lims = np.exp([-8, 6])
    # err_cols = ['W_err', '{}_err'.format(col)]
    cols = ['W', col]# + err_cols
    if col in df:
        df = df[cols].dropna(subset=['W', col])
        # errs = df[err_cols].fillna(max_err)
        x, y = np.exp(df['W']), np.exp(df[col])
        
        axes.scatter(x, y, c=color, alpha=alpha, s=15, lw=0, label=label)
        # dx, dy = errs['W_err'], errs[err_cols[-1]]
        # axes.errorbar(x, y, xerr=dx, yerr=dy, lw=0.5, alpha=alpha, ecolor=color, fmt='none')    
        
        axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--', alpha=alpha)
        axes.set(xlabel='{}-WT branching events'.format(target),
                ylabel='{}-Mutant branching events'.format(target),
                xscale='log', yscale='log',
                xlim=lims, ylim=lims
                )
        axes.grid(alpha=0.2)
    
    

if __name__ == '__main__':
    max_err = 0
    cols = ['EJ2 variant', 'EJ2', 'J2', 'PLT3', 'PLT7', 'Season']
    backgrounds = 'WHM'
    n = len(backgrounds)
    print('Plotting effects of mutations across backgrounds')

    data = [('data', pd.read_csv('data/genotype_means.csv', index_col=0), 'log_mean'),
            ('pairwise', pd.read_csv('results/model_predictions.genotypes.dxe.csv', index_col=0), 'coef'),
            ('threeway', pd.read_csv('results/model_predictions.genotypes.d3.csv', index_col=0), 'coef'),
            ('multilinear', pd.read_csv('results/data.predictions.multilinear2.csv', index_col=0), 'pred'),
            # ('biophysical', pd.read_csv('results/genotypes_season_estimates.biophysical_model.csv', index_col=0), 'coef')
            ]
    nrows = len(data)

    fig, subplots = plt.subplots(nrows, 6, figsize=(3 * 6, 2.75 * nrows), sharex=True, sharey=True)

    for axes_row, (label, gt_data, cname) in zip(subplots, data):
        gt_data['ej2_bc'] = [x[0] for x in gt_data['EJ2']]
        
        axes_sets = (axes_row[:3], axes_row[3:])    
        for target, other, axes_list in zip(['EJ2', 'J2'], ['J2', 'ej2_bc'], axes_sets):
            scols = [c for c in cols[1:] if c != target]
            gt_data['label'] = ['{}{}{}-{}'.format(*x) for x in gt_data[scols].values]

            for background, axes in zip(backgrounds, axes_list):
                data = gt_data.loc[gt_data[other] == background, :]
                df = pd.pivot_table(data, columns=target, values=cname, index='label')

                if target == 'EJ2':
                    for variant in EJ2_SERIES:
                        plot(df, axes, target=target, max_err=max_err, col='M{}'.format(variant), color='black', label='Homozygous')    
                        plot(df, axes, target=target, max_err=max_err, col='H{}'.format(variant), color='grey', label='Heterozygous')
                else:
                    plot(df, axes, target=target, max_err=max_err, col='M', color='black', label='Homozygous')    
                    plot(df, axes, target=target, max_err=max_err, col='H', color='grey', label='Heterozygous')
                axes.set(title='{} {}-{} background'.format(label, other, background))

                idx = (data['PLT7'] == 'W') & (data['J2'] == 'W') & (data['PLT3'] == 'M')
                sdata = data.loc[idx, :]
                
                if sdata.shape[0] > 0:
                    df = pd.pivot_table(sdata, columns=target, values=cname, index='label')
                    for variant in EJ2_SERIES:
                        if 'M{}'.format(variant) in df.columns and 'W' in df.columns:
                            plot(df, axes, target=target, max_err=max_err, col='M{}'.format(variant), color='red', label='EJ2-PLT3')    

    # subplots[0, 0].legend(loc=4)

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/mut_effects_plts_bcs.models'
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    