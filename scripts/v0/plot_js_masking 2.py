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
    print(gts)

    # Load estimates
    estimates = pd.read_csv('results/test_j2_effects_across_backgrounds.{}.csv'.format(model_label), index_col=0)
    estimates['background'] = [x.split('_')[-1] for x in estimates.index]
    estimates['mutation'] = [x.split('_')[0][:-1] for x in estimates.index]
    estimates['gt'] = [x.split('_')[0][-1] for x in estimates.index]
    estimates['j2_background'] = [x[2] for x in estimates['background']]
    estimates = estimates.join(gts, on='background', rsuffix='_bc')

    for j2, est in estimates.groupby('j2_background'):

        fig, subplots = plt.subplots(2, 3, figsize=(3 * 3, 3 * 2), sharex=True, sharey=True)
        subplots = subplots.flatten()
        palette = {'H': 'dimgrey', 'M': 'black'}

        for i, (variant, axes) in enumerate(zip(EJ2_SERIES_LABELS, subplots)):
            data = est.loc[est['mutation'] == variant, :]

            axes.axhline(0, lw=0.5, linestyle='--', color='grey')
            for gt, df in data.groupby('gt'):
                axes.errorbar(df['coef_bc'], df['coef'],
                            xerr=df['std err_bc'], yerr=df['std err'],
                            fmt='o', capsize=2., color=palette[gt], label=gt)
                
            axes.set(xlabel='PLT3/PLT7 Background phenotype' if i > 2 else '',
                     ylabel='{} effect in $j2$'.format(variant) if j2 == 'M' else '{} effect'.format(variant))
            axes.legend(loc=1, fontsize=8)
            axes.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig('plots/ej2_effects_j2{}.{}.png'.format(j2, model_label), dpi=300)
        