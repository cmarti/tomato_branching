#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import combinations, product
from scripts.settings import G_SITES_SIMPLE, ALLELES, EJ2_SERIES_LABELS, EJ2_SERIES
                        
                        
def get_site_allele_labels(sites, reverse=False):
    alleles = ALLELES.copy()
    if reverse:
        alleles = alleles[::-1]
    labels = np.hstack([['{}-{}'.format(site, gt) for gt in alleles] for site in sites])
    return(labels)


def get_site_pairs_allele_labels(sites, reverse=False):
    alleles = ALLELES.copy()
    if reverse:
        alleles = alleles[::-1]
    labels = np.hstack([['{}-{} {}-{}'.format(site1, a1) for gt in alleles] for site in sites])
    return(labels)


if __name__ == '__main__':
    model_label = 'd3'
    
    # Load predictions
    estimates = pd.read_csv('results/genotypes_estimates.{}.csv'.format(model_label),
                            index_col=0)
    
    series_cols = get_site_allele_labels(EJ2_SERIES_LABELS, reverse=False)
    cols = ['PLT7-W', 'PLT7-H', 'PLT7-M']
    
    df = []
    rows = []

    combs = [('J2', 'PLT3', 'W', 'W'), ('J2', 'PLT3', 'W', 'H'), ('J2', 'PLT3', 'W', 'M'),
             ('J2', 'PLT3', 'M', 'W'), ('J2', 'PLT3', 'M', 'H'), ('J2', 'PLT3', 'M', 'M'),
             ('EJ2', 'PLT3', 'M6', 'W'), ('EJ2', 'PLT3', 'M6', 'H'), ('EJ2', 'PLT3', 'M6', 'M'),
             ('EJ2', 'PLT3', 'M8', 'W'), ('EJ2', 'PLT3', 'M8', 'H'), ('EJ2', 'PLT3', 'M8', 'M')][::-1]

    for site1, site2, a1, a2  in combs:
        comb_label = '{}-{} {}-{}'.format(site1, a1, site2, a2)
        rows.append(comb_label)
        
        for a3 in ALLELES:
            gt = dict(zip(G_SITES_SIMPLE + ['EJ2'], 'W' * 4))
            gt.update({site1: a1, site2: a2, 'PLT7': a3})
            gt = ''.join([gt[x] for x in G_SITES_SIMPLE + ['EJ2']])
            record = {'v1': comb_label,
                      'v2': '{}-{}'.format('PLT7', a3),
                      'gt' : gt}
            df.append(record)

    df = pd.DataFrame(df).join(estimates, on='gt')
    values = pd.pivot_table(df, index='v1', columns='v2', values='coef')[cols].loc[rows, :]
    values.loc[['J2-W PLT3-M', 'J2-M PLT3-M', 'EJ2-M6 PLT3-M', 'EJ2-M8 PLT3-M'], 'PLT7-M'] = np.nan
    print(values)
    
    fig, axes = plt.subplots(1, 1, figsize=(2.8, 4))
    axes.set_facecolor('lightgrey')
    lw=0.75
    heatmap_kwargs = {'cmap': 'Blues', 'vmin': -3, 'vmax': 3,
                      'cbar_kws': {'label': 'Expected branching events', 'shrink': 0.8}}
    sns.heatmap(values, ax=axes, **heatmap_kwargs)
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    axes.hlines([0, 3, 6, 9, 12, 15, 18, 21, 24], xmin=xlim[0], xmax=xlim[1], lw=0.5, color='black')
    axes.vlines([0, 3, 6, 9, 12, 15, 18], ymin=ylim[0], ymax=ylim[1], lw=lw, color='black')
    sns.despine(ax=axes, top=False, right=False)
    axes.set(xlabel=None, ylabel=None)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    fig.tight_layout()
    
    # Adjust cbar    
    for spine in fig.axes[-1].spines.values():
        spine.set_visible(True)
    fig.axes[-1].set_yticks(np.log([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]))
    fig.axes[-1].set_yticklabels([r'$\frac{1}{16}$', r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', 1, 2, 4, 8, 16])
    
    # Save figure
    fig.savefig('figures/triple_mutant_predictions_plt.{}.png'.format(model_label), dpi=300)
    fig.savefig('figures/triple_mutant_predictions_plt.{}.pdf'.format(model_label))
    