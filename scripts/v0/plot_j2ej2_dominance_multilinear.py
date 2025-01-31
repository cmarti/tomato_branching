#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import EJ2_SERIES
        
        
def plot(df, axes, target, max_err, col='M', color='black', alpha=1, label=None):
    lims = (-8, 6)
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
    lims = (-0.25, 7)
    fig, subplots = plt.subplots(2, 2, figsize=(2.75 * 2, 2.75 * 2), sharex=True, sharey=True)

    gt_data = pd.read_csv('results/js_predictions.multilinear.csv', index_col=0)
    gt_data['j2'] = [x[0] for x in gt_data.index.values]
    gt_data['ej2'] = [x[1:] for x in gt_data.index.values]
    for i, axes in enumerate(subplots[0]):
        data = gt_data.pivot(index='j2', columns='ej2', values='log_branches_{}'.format(i+1)).loc[['W', 'H', 'M']]
        for variant in EJ2_SERIES:
            w, h, m = data['W'], data['H{}'.format(variant)], data['M{}'.format(variant)]
            axes.scatter(h, m, s=10, c='black', lw=0)    
            axes.scatter(w, h, s=10, c='grey', lw=0) 
            axes.plot(h, m, c='black', lw=1)
            axes.plot(w, h, c='grey', lw=1)

        axes.plot(lims, lims, linestyle='--', lw=0.5, c='grey')
        axes.set(ylabel='EJ2 Mutant phenotype',
                xlabel='J2/EJ2 background phenotype', xlim=lims, ylim=lims)
        axes.grid(alpha=0.2)
    
    gt_data = pd.read_csv('results/plts_predictions.multilinear.csv', index_col=0)
    gt_data['plt3'] = [x[0] for x in gt_data.index.values]
    gt_data['plt7'] = [x[1:] for x in gt_data.index.values]

    for i, axes in enumerate(subplots[1]):
        data = gt_data.pivot(index='plt3', columns='plt7', values='log_branches_{}'.format(i+1)).loc[['W', 'H', 'M']]
        w, h, m = data['W'], data['H'], data['M']
        axes.scatter(h, m, s=10, c='black', lw=0)    
        axes.scatter(w, h, s=10, c='grey', lw=0) 
        axes.plot(h, m, c='black', lw=1)
        axes.plot(w, h, c='grey', lw=1)

        axes.plot(lims, lims, linestyle='--', lw=0.5, c='grey')
        axes.set(ylabel='PLT7 Mutant phenotype',
                xlabel='PLT3/PLT7 background phenotype', xlim=lims, ylim=lims)
        axes.grid(alpha=0.2)

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/dominance.multilinear'
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    

