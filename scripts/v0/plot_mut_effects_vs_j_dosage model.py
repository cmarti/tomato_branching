#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import EJ2_SERIES
        
        
def plot(df, axes, target, max_err, col='M', color='black', alpha=1, label=None):
    lims = (-8, 6)
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
    
    

if __name__ == '__main__':
    model = 'd3'

    max_err = 0
    cols = ['EJ2 variant', 'EJ2', 'J2', 'PLT3', 'PLT7', 'Season']
    print('Plotting effects of mutations across backgrounds')
    # gt_data = pd.read_csv('results/genotypes_season_estimates.{}.csv'.format(model), index_col=0)
    gt_data = pd.read_csv('results/model_predictions.genotypes.{}.csv'.format(model), index_col=0)
    d = pd.read_csv('data/nb_model.predictions.csv', index_col=0)
    gt_data['nb_pred'] = d['yhat']
    print(gt_data)
    print(d)
    exit()
    model = 'nb2d'

    # max_err = np.nanmax(gt_data['std err'].values)
    backgrounds = 'WHM'
    n = len(backgrounds)
    # Init figure
    fig, subplots = plt.subplots(2, n, figsize=(3 * 3, 2.75 * 2), sharex=True, sharey=True)
    
    for target, other, axes_row in zip(['PLT3', 'PLT7'], ['PLT7', 'PLT3'], subplots):
        scols = [c for c in cols[1:] if c != target]
        gt_data['label'] = ['{}{}{}-{}'.format(*x) for x in gt_data[scols].values]

        for plt_background, axes in zip(backgrounds, axes_row):
            data = gt_data.loc[gt_data[other] == plt_background, :]
            df = pd.pivot_table(data, columns=target, values='nb_pred', index='label')
            # df = df.join(pd.pivot_table(data, columns=target, values='std err', index='label'), rsuffix='_err')
            if plt_background != 'M':
                plot(df, axes, target=target, max_err=max_err, col='M', color='black', label='Homozygous')    
            plot(df, axes, target=target, max_err=max_err, col='H', color='grey', label='Heterozygous')    
            axes.legend(loc=4)
            axes.set(title='{}-{} background'.format(other, plt_background))

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/mut_effects_js_bcs.{}'.format(model)
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    