#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import EJ2_SERIES
        
        
def plot(df, axes, target, max_err, col='M', color='black', alpha=1, label=None):
    lims = (-8, 5)
    # err_cols = ['W_err', '{}_err'.format(col)]
    cols = ['W', col]# + err_cols
    df = df[cols].dropna(subset=['W', col])
    # errs = df[err_cols].fillna(max_err)
    x, y = df['W'], df[col]
    
    axes.scatter(x, y, c=color, alpha=alpha, s=15, lw=0, label=label)
    # dx, dy = errs['W_err'], errs[err_cols[-1]]
    # axes.errorbar(x, y, xerr=dx, yerr=dy, lw=0.5, alpha=alpha, ecolor=color, fmt='none')    
    
    axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--', alpha=alpha)
    axes.set(xlabel='{}-WT'.format(target),
            ylabel='{}-Mutant'.format(target),
            # xscale='log', yscale='log',
            xlim=lims, ylim=lims
            )
    axes.grid(alpha=0.2)


def get_split_gt(gt):
    return({'PLT3': gt[0], 'PLT7': gt[1],
            'J2': gt[2], 'EJ2': gt[3:]})
    

if __name__ == '__main__':
    model = 'biophysical_model'

    cols = ['EJ2 variant', 'EJ2', 'J2', 'PLT3', 'PLT7', 'Season']
    print('Plotting effects of mutations across backgrounds')
    gt_data = pd.read_csv('results/genotypes_season_estimates.{}.csv'.format(model), index_col=0)
    # gt_data = gt_data.join(pd.DataFrame([get_split_gt(x) for x in gt_data.index], index=gt_data.index))
    j2_backgrounds = 'WHM'
    
    # Init figure
    fig, subplots = plt.subplots(6, 3, figsize=(3 * 3, 2.75 * 6), sharex=True, sharey=True)
    
    target = 'EJ2'
    scols = [c for c in cols[1:] if c != target]
    gt_data['label'] = ['{}{}{}-{}'.format(*x) for x in gt_data[scols].values]
    show_legend = True
    for j2_background, axes_col in zip(j2_backgrounds, subplots.T):
        data = gt_data.loc[gt_data['J2'] == j2_background, :]
        df = pd.pivot_table(data, columns=target, values='coef', index='label')
        # df = df.join(pd.pivot_table(data, columns=target, values='std err', index='label'), rsuffix='_err')
        for allele, axes in zip(EJ2_SERIES, axes_col):
            plot(df, axes, target='EJ2({})'.format(allele), max_err=0, col='M{}'.format(allele), color='black', label='Homozygous')    
            plot(df, axes, target='EJ2({})'.format(allele), max_err=0, col='H{}'.format(allele), color='grey', label='Heterozygous')    
            if show_legend:
                axes.legend(loc=4)
                show_legend = False
            axes.set(title='J2-{} background'.format(j2_background))

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/mut_effects_plt_bcs_{}'.format(model)
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    

    # Init figure
    fig, subplots = plt.subplots(1, 3, figsize=(3 * 3, 3), sharex=True, sharey=True)
    
    target = 'EJ2'
    scols = [c for c in cols[1:] if c != target]
    gt_data['label'] = ['{}{}{}-{}'.format(*x) for x in gt_data[scols].values]
    show_legend = True
    for j2_background, axes in zip(j2_backgrounds, subplots):
        data = gt_data.loc[gt_data['J2'] == j2_background, :]
        df = pd.pivot_table(data, columns=target, values='coef', index='label')
        # df = df.join(pd.pivot_table(data, columns=target, values='std err', index='label'), rsuffix='_err')

        for allele in EJ2_SERIES:
            plot(df, axes, target='EJ2({})'.format(allele), max_err=0, col='M{}'.format(allele), color='black', label='Homozygous')    
            plot(df, axes, target='EJ2({})'.format(allele), max_err=0, col='H{}'.format(allele), color='grey', label='Heterozygous')    
            if show_legend:
                axes.legend(loc=4)
                show_legend = False
        axes.set(title='J2-{} background'.format(j2_background),
                 xlabel='EJ2-WT', ylabel='EJ2-Mutant')

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/mut_effects_plt_bcs_joint_{}'.format(model)
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))