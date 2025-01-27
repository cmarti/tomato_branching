#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import G_SITES_SIMPLE, ALLELES, EJ2_SERIES_LABELS
                        
                        
if __name__ == '__main__':
    model_label = 'd3'
    
    order = ['wt', 'j2'] + [x.lower() for x in EJ2_SERIES_LABELS]
    order += [x.lower() + '/j2' for x in EJ2_SERIES_LABELS]
    hue_order = ['PLT7', 'PLT3']
    colors = ['dimgrey', 'black']

    # Load estimates
    estimates = pd.read_csv('results/test_masking_plts.{}.csv'.format(model_label), index_col=0)
    estimates['background'] = [x.split('_')[0] for x in estimates.index]
    estimates = estimates.loc[np.isin(estimates['background'], order), :]
    estimates['site'] = [x.split('_')[1] for x in estimates.index]
    estimates['x1'] = [order.index(x) for x in estimates['background']]
    estimates['x2'] = [0.2 * (hue_order.index(x)-1) for x in estimates['site']]
    estimates['x'] = estimates['x1'] + estimates['x2']
    print(estimates)

    fig, axes = plt.subplots(1, 1, figsize=(5.5, 3.5))

    axes.axhline(0, lw=0.5, linestyle='--', color='grey')
    for site, color in zip(hue_order, colors):
        df = estimates.loc[estimates['site'] == site, :]
        axes.errorbar(df['x'], df['coef'], yerr=2 * df['std err'], fmt='o', capsize=2.,
                      color=color, label=site)
        
    axes.set(xlabel='Genetic background', xticks=np.arange(len(order)), xticklabels=order,
             ylabel='Homozygous mutational effect')
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
    axes.legend(loc=3, fontsize=8, title='Locus', ncols=3)
    axes.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig('plots/masking_js.{}.png'.format(model_label), dpi=300)
    fig.savefig('plots/masking_js.{}.pdf'.format(model_label), dpi=300)
    