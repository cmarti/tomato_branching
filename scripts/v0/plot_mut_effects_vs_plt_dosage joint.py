#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import EJ2_SERIES
        
        
def plot(df, axes, target, max_err, col='M', color='black', alpha=1, label=None):
    lims = (-8, 5)
    err_cols = ['W_err', '{}_err'.format(col)]
    cols = ['W', col] + err_cols
    df = df[cols].dropna(subset=['W', col])
    errs = df[err_cols].fillna(max_err)
    x, y = df['W'], df[col]
    
    axes.scatter(x, y, c=color, alpha=alpha, s=15, lw=0, label=label)
    dx, dy = errs['W_err'], errs[err_cols[-1]]
    # axes.errorbar(x, y, xerr=dx, yerr=dy, lw=0.5, alpha=alpha, ecolor=color, fmt='none')    
    
    axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--', alpha=alpha)
    axes.set(xlabel='{}-WT'.format(target),
            ylabel='{}-Mutant'.format(target),
            # xscale='log', yscale='log',
            xlim=lims, ylim=lims
            )
    axes.grid(alpha=0.2)
    

if __name__ == '__main__':
    cols = ['EJ2 variant', 'EJ2', 'J2', 'PLT3', 'PLT7', 'Season']
    print('Plotting effects of mutations across backgrounds')
    gt_data = pd.read_csv('data/genotype_means.csv', index_col=0)
    max_err = np.nanmax(gt_data['log_std_err'].values)
    j2_backgrounds = 'WHM'
    
    # Init figure
    fig, subplots = plt.subplots(1, 3, figsize=(3 * 3, 3), sharex=True, sharey=True)
    
    target = 'EJ2'
    scols = [c for c in cols[1:] if c != target]
    gt_data['label'] = ['{}{}{}-{}'.format(*x) for x in gt_data[scols].values]
    show_legend = True
    for j2_background, axes in zip(j2_backgrounds, subplots):
        data = gt_data.loc[gt_data['J2'] == j2_background, :]
        df = pd.pivot_table(data, columns=target, values='log_mean', index='label')
        df = df.join(pd.pivot_table(data, columns=target, values='log_std_err', index='label'), rsuffix='_err')

        for allele in EJ2_SERIES:
            plot(df, axes, target='EJ2({})'.format(allele), max_err=max_err, col='M{}'.format(allele), color='black', label='Homozygous')    
            plot(df, axes, target='EJ2({})'.format(allele), max_err=max_err, col='H{}'.format(allele), color='grey', label='Heterozygous')    
            if show_legend:
                axes.legend(loc=4)
                show_legend = False
        axes.set(title='J2-{} background'.format(j2_background),
                 xlabel='EJ2-WT', ylabel='EJ2-Mutant')

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/mut_effects_plt_bcs_joint'
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    