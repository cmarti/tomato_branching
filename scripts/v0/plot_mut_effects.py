#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import EJ2_SERIES
        
        
def plot(df, axes, target, max_err, col='M'):
    lims = (-8, 8)
    err_cols = ['W_err', '{}_err'.format(col)]
    cols = ['W', col] + err_cols
    df = df[cols].dropna(subset=['W', col])
    errs = df[err_cols].fillna(max_err)
    x, y = df['W'], df[col]
    
    axes.scatter(x, y, c='black', alpha=0.5, s=15, lw=0)
    dx, dy = errs['W_err'], errs[err_cols[-1]]
    axes.errorbar(x, y, xerr=dx, yerr=dy, lw=0.5, alpha=0.3, ecolor='black', fmt='none')    
    
    axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--', alpha=0.5)
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
    print(gt_data)
    
    # Init figure
    fig, subplots = plt.subplots(3, 3, figsize=(3 * 3, 2.75 * 3), sharex=True, sharey=True)
    
    for target, axes in zip(['J2', 'PLT3', 'PLT7'], subplots[0]):
        scols = [c for c in cols[1:] if c != target]
        gt_data['label'] = ['{}{}{}-{}'.format(*x) for x in gt_data[scols].values]
        df = pd.pivot_table(gt_data, columns=target, values='log_mean', index='label')
        df = df.join(pd.pivot_table(gt_data, columns=target, values='log_std_err', index='label'), rsuffix='_err')
        # idx = [x.split('-')[0][-2:] == 'WW' for x in df.index]
        # df = df.loc[idx, :]
        plot(df, axes, target, max_err)
        
    target = 'EJ2'
    scols = [c for c in cols[1:] if c != target]
    gt_data['label'] = ['{}{}{}-{}'.format(*x) for x in gt_data[scols].values]
    df = pd.pivot_table(gt_data, columns=target, values='log_mean', index='label')
    df = df.join(pd.pivot_table(gt_data, columns=target, values='log_std_err', index='label'), rsuffix='_err')
    for allele, axes in zip(EJ2_SERIES, subplots[1:].flatten()):
        plot(df, axes, target='EJ2({})'.format(allele), max_err=max_err, col='M{}'.format(allele))    

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/mut_effects'
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    