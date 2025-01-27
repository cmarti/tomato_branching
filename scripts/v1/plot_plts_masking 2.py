#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import G_SITES_SIMPLE, ALLELES, EJ2_SERIES_LABELS
                        
                        
if __name__ == '__main__':
    model_label = 'd3'

    # Load genotype estimates
    gts = pd.read_csv('results/genotypes_estimates.{}.csv'.format(model_label), index_col=0)

    # Load estimates
    estimates = pd.read_csv('results/test_plts_effects_across_backgrounds.{}.csv'.format(model_label), index_col=0)
    estimates['background'] = [x.split('_')[-1] for x in estimates.index]
    estimates['mutation'] = [x.split('_')[0][:-1] for x in estimates.index]
    estimates['gt'] = [x.split('_')[0][-1] for x in estimates.index]
    estimates = estimates.join(gts, on='background', rsuffix='_bc')
    
    # estimates = estimates.loc[[x[:4] == 'WWWM' or x[:4] == 'WWWW' for x in estimates['background']], :]
    # print(estimates)

    fig, subplots = plt.subplots(1, 2, figsize=(2 * 3, 1 * 3), sharex=True, sharey=True)
    subplots = subplots.flatten()
    palette = {'H': 'dimgrey', 'M': 'black'}

    for i, (variant, axes) in enumerate(zip(['PLT3', 'PLT7'], subplots)):
        data = estimates.loc[estimates['mutation'] == variant, :]

        axes.axhline(0, lw=0.5, linestyle='--', color='grey')
        for gt, df in data.groupby('gt'):
            # axes.scatter(df['coef_bc'], df['coef'], color=palette[gt], label=gt,
            #              s=10, alpha=0.5, lw=0)
            axes.errorbar(df['coef_bc'], df['coef'],
                        xerr=df['std err_bc'], yerr=df['std err'],
                        ms=3, alpha=0.5, elinewidth=0.75, capthick=0.75,
                        fmt='o', capsize=2., color=palette[gt], label=gt)
            
        axes.set(xlabel='J2/EJ2 Background phenotype',
                 ylabel='{} effect'.format(variant))
        axes.legend(loc=1, fontsize=8)
        axes.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig('plots/plts_effects.{}.png'.format(model_label), dpi=300)
    