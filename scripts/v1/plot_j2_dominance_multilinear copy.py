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
    print('Plotting effects of mutations across backgrounds')
    gt_data = pd.read_csv('results/js_predictions.multilinear.csv', index_col=0)
    gt_data['j2'] = [x[0] for x in gt_data.index.values]
    gt_data['ej2'] = [x[1:] for x in gt_data.index.values]

    lims = (-0.25, 6)
    fig, subplots = plt.subplots(1, 2, figsize=(3 * 2, 2.75 ), sharex=True, sharey=True)
    
    axes = subplots[0]
    data = gt_data.pivot(index='j2', columns='ej2', values='log_branches').loc[['W', 'H', 'M']]
    cols = []
    for variant in EJ2_SERIES:
        data['M{}'.format(variant)] -= data.loc['W', 'M{}'.format(variant)]
        data['H{}'.format(variant)] -= data.loc['W', 'H{}'.format(variant)]
        # data['D{}'.format(variant)] = data['H{}'.format(variant)] / data['M{}'.format(variant)]
        # cols.append('D{}'.format(variant))

        axes.scatter(data['W'], data['M{}'.format(variant)], s=5, c='black', lw=0)    
        axes.plot(data['W'], data['M{}'.format(variant)], c='black', lw=1)
        axes.scatter(data['W'], data['H{}'.format(variant)], s=5, c='grey', lw=0) 
        axes.plot(data['W'], data['H{}'.format(variant)], c='grey', lw=1)

    axes.set(ylabel='EJ2 Mutant phenotype',
             xlabel='J2 background phenotype', xlim=lims, ylim=lims)
    axes.grid(alpha=0.2)

    # print(data)
    # print(data[cols])

    data = gt_data.pivot(index='ej2', columns='j2', values='log_branches')
    # data['M'] -= data['W']
    # data['H'] -= data['W']
    # data['D'] = data['H'] / data['M']
    # print(data)
    # print(data['D'].mean())
    
    axes = subplots[1]
    # axes.scatter(data['H'], data['M'], s=5, c='black', lw=0)    
    axes.scatter(data['W'], data['M'], s=5, c='black', lw=0)    
    axes.scatter(data['W'], data['H'], s=5, c='grey', lw=0)    
    for x, y, z, label in zip(data['W'], data['H'], data['M'], data.index.values):
        axes.text(x, y, label, color='grey', fontsize=7)
        axes.text(x, z, label, color='black', fontsize=7)
        # axes.text(y, z, label, color='black', fontsize=7)
    axes.set(xlabel='J2 Heterozygous effect',
             ylabel='J2 Homozygous effect', xlim=lims, ylim=lims)
    axes.grid(alpha=0.2)

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/dominance.multilinear'
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    