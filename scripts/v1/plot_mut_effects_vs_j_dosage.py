#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import EJ2_SERIES
        
        
def plot(df, axes, target, max_err, col='M', ref='W', color='black', alpha=1, label=None):
    lims = (-8, 6)
    err_cols = ['{}_err'.format(ref), '{}_err'.format(col)]
    cols = [ref, col] + err_cols
    df = df[cols].dropna(subset=[ref, col])
    errs = df[err_cols].fillna(max_err)
    x, y = df[ref], df[col]
    
    axes.scatter(x, y, c=color, alpha=alpha, s=15, lw=0, label=label)
    dx, dy = errs['{}_err'.format(ref)], errs[err_cols[-1]]
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
    backgrounds = 'WHM'
    n = len(backgrounds)
    # Init figure
    fig, subplots = plt.subplots(2, n, figsize=(3 * 3, 2.75 * 2), sharex=True, sharey=True)
    
    for target, other, axes_row in zip(['PLT3', 'PLT7'], ['PLT7', 'PLT3'], subplots):
        scols = [c for c in cols[1:] if c != target]
        gt_data['label'] = ['{}{}{}-{}'.format(*x) for x in gt_data[scols].values]

        for plt_background, axes in zip(backgrounds, axes_row):
            data = gt_data.loc[gt_data[other] == plt_background, :]
            df = pd.pivot_table(data, columns=target, values='log_mean', index='label')
            df = df.join(pd.pivot_table(data, columns=target, values='log_std_err', index='label'), rsuffix='_err')
            if plt_background != 'M':
                plot(df, axes, target=target, max_err=max_err, col='M', ref='W', color='black', label='Homozygous')    
            plot(df, axes, target=target, max_err=max_err, col='H', ref='W', color='grey', label='Heterozygous')    
            axes.legend(loc=4)
            axes.set(title='{}-{} background'.format(other, plt_background))

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/mut_effects_js_bcs'.format()
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    