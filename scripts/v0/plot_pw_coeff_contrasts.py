#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import G_SITES_SIMPLE, ALLELES, EJ2_SERIES_LABELS
                        

if __name__ == '__main__':
    # Load contrasts
    contrasts = pd.read_csv('results/model_epistatic_coefficients_contrasts.csv', index_col=0)
    contrasts['label'] = ['{}-{}'.format(a2, a1) for a1, a2 in zip(contrasts['a1'], contrasts['a2'])]

    rows = ['J2', 'PLT3', 'PLT7'][::-1]
    cols = ['EJ2(1)-EJ2(4)', 'EJ2(3)-EJ2(1)', 'EJ2(3)-EJ2(4)', 
            'EJ2(8)-EJ2(1)', 'EJ2(8)-EJ2(3)', 'EJ2(8)-EJ2(4)',
            'EJ2(7)-EJ2(1)', 'EJ2(7)-EJ2(3)', 'EJ2(7)-EJ2(4)', 'EJ2(7)-EJ2(8)',
            'EJ2(6)-EJ2(1)', 'EJ2(6)-EJ2(3)', 'EJ2(6)-EJ2(4)', 'EJ2(6)-EJ2(7)', 'EJ2(6)-EJ2(8)']

    coeffs = pd.pivot_table(contrasts, index='label', columns='background', values='coef')[rows].T[cols]
    pvalues = pd.pivot_table(contrasts, index='label', columns='background', values='P>|z|')[rows].T[cols]
    labels = np.full(pvalues.shape, '')
    labels[pvalues.values < 0.05] = '*'

    fig, axes = plt.subplots(1, 1, figsize=(7, 2.5))
    sns.heatmap(coeffs, vmin=-1, vmax=1, cmap='coolwarm', ax=axes, annot=labels, fmt='',
                cbar_kws={'label': '$\epsilon_{S_3, S_1} - \epsilon_{S_3, S_2}$'})
    axes.set(ylabel='Site 3 ($S_3$)', xlabel='Site 1 ($S_1$) - Site 2 ($S_2$)')
    sns.despine(top=False, right=False)
    fig.tight_layout()
    fig.savefig('figures/delta_epistatic_coeffs.png', dpi=300)
    fig.savefig('figures/delta_epistatic_coeffs.pdf')


    