#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.utils import get_double_mutant_plant_data
from scripts.settings import G_SITES_SIMPLE, ALLELES, EJ2_SERIES_LABELS, EJ2_SERIES
                        
                        
def get_site_allele_labels(sites, reverse=False):
    alleles = ALLELES.copy()
    if reverse:
        alleles = alleles[::-1]
    labels = np.hstack([['{}-{}'.format(site, gt) for gt in alleles] for site in sites])
    return(labels)


if __name__ == '__main__':
    model_label = 'dxe'
    
    # Load predictions
    estimates = pd.read_csv('results/genotypes_estimates.{}.csv'.format(model_label),
                            index_col=0)
    print(estimates)
    series_cols = get_site_allele_labels(EJ2_SERIES_LABELS, reverse=False)
    
    plant_data = get_double_mutant_plant_data().drop('Season', axis=1)
    plant_data['gt'] = [''.join(x) for x in plant_data[G_SITES_SIMPLE + ['EJ2']].values]
    plant_data = plant_data.join(estimates, on='gt').drop_duplicates()
    values = pd.pivot_table(plant_data, index='v1', columns='v2', values='coef')
    plant_data.columns = ['variant1', 'variant2', 'PLT3', 'PLT7', 'J2', 'EJ2', 'gt', 'log_branching_events', 'stderr', 'ci_95_lower', 'ci_95_upper']
    plant_data ['exp_branching_events'] = np.exp(plant_data['log_branching_events'])
    cols = ['variant1', 'variant2', 'PLT3', 'PLT7', 'J2', 'EJ2', 'log_branching_events', 'ci_95_lower', 'ci_95_upper', 'exp_branching_events']
    plant_data[cols].to_csv('results/double_mutants_predictions.csv', index=False)
    print(plant_data[cols])
    
    # Define plotting order
    null_rows = get_site_allele_labels(['J2', 'PLT7', 'PLT3'], reverse=True)
    series_rows = get_site_allele_labels(['PLT7', 'PLT3', 'J2'], reverse=True)
    
    null_cols = get_site_allele_labels(['J2', 'PLT7'], reverse=False)
    series_cols = get_site_allele_labels(EJ2_SERIES_LABELS, reverse=False)
    
    # Select dataframe subsets to make plots
    pred_nulls = values.loc[null_rows, :][null_cols]
    pred_nulls.loc['PLT3-M', 'PLT7-M'] = np.nan
    pred_series = values.loc[series_rows, :][series_cols]

    # Make joint heatmaps
    ws = np.array([2, 8.])
    fig, subplots = plt.subplots(1, 2, figsize=(8.5, 3), width_ratios=ws / ws.sum())
    heatmap_kwargs = {'cmap': 'Blues', 'vmin': -3, 'vmax': 3,
                      'cbar_kws': {'label': 'Expected branching events'}}
    
    # Make heatmap for null mutants
    axes = subplots[0]
    lw = 0.75
    sns.heatmap(pred_nulls, ax=axes, cbar=False, **heatmap_kwargs)
    axes.set(ylabel=None, xlabel=None, yticks=axes.get_yticks()[3:])
    axes.plot((0, 9), (6, 6), lw=lw, color='black')
    axes.plot((0, 3), (3, 3), lw=lw, color='black')
    axes.plot((3, 3), (3, 9), lw=lw, color='black')
    axes.plot((0, 0), (3, 9), lw=lw, color='black')
    axes.plot((5.98, 5.98), (6, 9), lw=lw, color='black')
    sns.despine(ax=axes, top=True, left=True, right=True)

    # Make heatmap for pairwise mutants
    axes = subplots[1]
    sns.heatmap(pred_series, ax=axes, **heatmap_kwargs)
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    axes.hlines([0, 3, 6, 9], xmin=xlim[0], xmax=xlim[1], lw=0.5, color='black')
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
    fname = 'figures/double_mutant_predictions.{}'.format(model_label)
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))

    