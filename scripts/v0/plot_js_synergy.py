#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import G_SITES_SIMPLE, ALLELES, EJ2_SERIES_LABELS
                        
                        
if __name__ == '__main__':
    model_label = 'dxe'

    # Load genotype estimates
    gts = pd.read_csv('results/genotypes_estimates.{}.csv'.format(model_label), index_col=0)

    # Load estimates
    estimates = pd.read_csv('results/test_j2_effects_across_ej2_backgrounds.{}.csv'.format(model_label), index_col=0)
    estimates['background'] = [x.split('_')[2] for x in estimates.index]
    estimates['mutation'] = [x.split('_')[0][:-1] for x in estimates.index]
    estimates['gt'] = [x.split('_')[0][-1] for x in estimates.index]
    estimates['ej2_background'] = [x[3] for x in estimates['background']]
    estimates = estimates.join(gts, on='background', rsuffix='_bc')
    
    data = estimates.loc[np.isin(estimates['background'], ['WWWW', 'WWWH', 'WWWM']), :]
    data['x'] = [0, 1, 2, 0, 1, 2]

    fig, axes = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
    palette = {'H': 'dimgrey', 'M': 'black'}
    axes.axhline(0, lw=0.5, linestyle='--', color='grey')
    for gt, df in data.groupby('gt'):
        axes.errorbar(df['x'], df['coef'], yerr=df['std err'],
                    fmt='o', capsize=2., color=palette[gt], label=gt)
    axes.set(xlabel='EJ2 Background', ylabel='J2 effect',
             xlim=(-0.5, 2.5), xticks=[0, 1, 2], xticklabels=['$wt$', '$ej2^{pro}/+$', '$ej2^{pro}$'])
    axes.legend(loc=2, fontsize=8)
    axes.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig('plots/j2_effects_across_ej2_means.{}.png'.format(model_label), dpi=300)

    print(df)
    exit()

    fig, subplots = plt.subplots(1, 3, figsize=(9, 3 * 1), sharex=True, sharey=True)
    palette = {'H': 'dimgrey', 'M': 'black'}
    
    for axes, ej2 in zip(subplots, ['W', 'H', 'M']):
        data = estimates.loc[estimates['ej2_background'] == ej2, :]
        axes.axhline(0, lw=0.5, linestyle='--', color='grey')
        for gt, df in data.groupby('gt'):
            axes.errorbar(df['coef_bc'], df['coef'],
                        xerr=df['std err_bc'], yerr=df['std err'],
                        fmt='o', capsize=2., color=palette[gt], label=gt)
            
        axes.set(xlabel='EJ2{} Background phenotype'.format(ej2), ylabel='J2 effect')
        axes.legend(loc=1, fontsize=8)
        axes.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig('plots/j2_effects_across_ej2.{}.png'.format(model_label), dpi=300)