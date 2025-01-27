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
    
    df = []
    rows = []

    combs = [('PLT3', 'J2', 'W', 'W'), ('PLT3', 'J2', 'W', 'H'), ('PLT3', 'J2', 'W', 'M'),
             ('PLT3', 'J2', 'M', 'W'), ('PLT3', 'J2', 'M', 'H'), ('PLT3', 'J2', 'M', 'M'),
             ('PLT7', 'J2', 'M', 'W'), ('PLT7', 'J2', 'M', 'H'), ('PLT7', 'J2', 'M', 'M')][::-1]

    for site1, site2, a1, a2  in combs:
        comb_label = '{}-{} {}-{}'.format(site1, a1, site2, a2)
        rows.append(comb_label)
        for variant, label in zip(EJ2_SERIES, EJ2_SERIES_LABELS):
            for a3 in ALLELES:
                gt = dict(zip(G_SITES_SIMPLE + ['EJ2'], 'W' * 4))
                gt.update({site1: a1, site2: a2, 'EJ2': 'W' if a3 == 'W' else '{}{}'.format(a3, variant)})
                gt = ''.join([gt[x] for x in G_SITES_SIMPLE + ['EJ2']])
                record = {'v1': comb_label,
                          'v2': '{}-{}'.format(label, a3),
                          'gt' : gt}
                df.append(record)
    df = pd.DataFrame(df).join(estimates, on='gt')
    values = pd.pivot_table(df, index='v1', columns='v2', values='coef')[series_cols].loc[rows, :]
    
    fig, axes = plt.subplots(1, 1, figsize=(6.5, 3))
    axes.set_facecolor('lightgrey')
    lw=0.75
    heatmap_kwargs = {'cmap': 'Blues', 'vmin': -3, 'vmax': 3,
                      'cbar_kws': {'label': 'Expected branching events'}}
    sns.heatmap(values, ax=axes, **heatmap_kwargs)
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    axes.hlines([0, 3, 6, 9, 12, 15, 18, 21, 24], xmin=xlim[0], xmax=xlim[1], lw=0.5, color='black')
    axes.vlines([0, 3, 6, 9, 12, 15, 18], ymin=ylim[0], ymax=ylim[1], lw=lw, color='black')
    sns.despine(ax=axes, top=False, right=False)
    axes.set(xlabel=None, ylabel=None)
    fig.tight_layout()
    
    # Adjust cbar    
    for spine in fig.axes[-1].spines.values():
        spine.set_visible(True)
    fig.axes[-1].set_yticks(np.log([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]))
    fig.axes[-1].set_yticklabels([r'$\frac{1}{16}$', r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', 1, 2, 4, 8, 16])
    
    # Save figure
    fig.savefig('figures/triple_mutant_predictions.{}.png'.format(model_label), dpi=300)
    fig.savefig('figures/triple_mutant_predictions.{}.pdf'.format(model_label))
    