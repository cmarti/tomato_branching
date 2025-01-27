#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import G_SITES_SIMPLE, ALLELES, EJ2_SERIES_LABELS
                        
                        
if __name__ == '__main__':
    model_label = 'd3'
    
    order = ['J2'] + EJ2_SERIES_LABELS
    hue_order = ['j2', 'plt3/j2', 'plt3/plt7h']
    colors = ['lightgrey', 'dimgrey', 'black']

    # Load estimates
    estimates = pd.read_csv('results/test_masking.{}.csv'.format(model_label), index_col=0)
    estimates['background'] = [x.split('_')[0] for x in estimates.index]
    estimates['site'] = [x.split('_')[1] for x in estimates.index]
    estimates = estimates.loc[np.isin(estimates['background'], hue_order), :]
    estimates = estimates.loc[np.isin(estimates['site'], order), :]
    print(estimates)
    estimates['x1'] = [order.index(x) for x in estimates['site']]
    estimates['x2'] = [0.2 * (hue_order.index(x)-1) for x in estimates['background']]
    estimates['x'] = estimates['x1'] + estimates['x2']

    fig, axes = plt.subplots(1, 1, figsize=(4.2, 3))

    axes.axhline(0, lw=0.5, linestyle='--', color='grey')
    for background, color in zip(hue_order, colors):
        df = estimates.loc[estimates['background'] == background, :]
        axes.errorbar(df['x'], df['coef'], yerr=2 * df['std err'], fmt='o', capsize=2.,
                      color=color, label=background)
        
    axes.set(xlabel='', xticks=np.arange(len(order)), xticklabels=order,
             ylabel='Homozygous mutational effect', xlim=(-0.5, len(order)-0.5))
    axes.legend(loc=(0.17, 1.05), fontsize=8, title='Background', ncols=3)
    axes.grid(alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig('plots/masking_plt3j2.{}.png'.format(model_label), dpi=300)
    fig.savefig('plots/masking_plt3j2.{}.pdf'.format(model_label), dpi=300)
    