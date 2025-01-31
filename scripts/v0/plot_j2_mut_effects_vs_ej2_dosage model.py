#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import EJ2_SERIES, SEASONS
from scripts.utils import plot_mut_scatter
    

if __name__ == '__main__':
    target = 'J2'
    backgrounds = ['WW', 'HW', 'HH', 'MW', 'MH']
    backgrounds = [{'PLT3': x[0], 'PLT7': x[1]} for x in backgrounds]
    n = len(backgrounds)
    
    print('Plotting effects of mutations across backgrounds')
    model = None
    gt_data = pd.read_csv('data/genotype_means.csv', index_col=0)
    print(gt_data)
    fig, subplots = plt.subplots(1, n, figsize=(3 * n, 3), sharex=True, sharey=True)
    for background, axes in zip(backgrounds, subplots):
        plot_mut_scatter(axes, gt_data, target, background, show_err=False, alpha=1, model=model)
    # subplots[-1].legend(loc=4)
    fig.tight_layout()
    fig.savefig('figures/j2_mut_effects_ej2_bcs.data.png', dpi=300)

    print('Plotting effects of mutations across backgrounds')
    model = 'dxe'
    gt_data = pd.read_csv('results/genotypes_season_estimates.{}.csv'.format(model), index_col=0)
    fig, subplots = plt.subplots(1, n, figsize=(3 * n, 3), sharex=True, sharey=True)
    for background, axes in zip(backgrounds, subplots):
        plot_mut_scatter(axes, gt_data, target, background, show_err=False, alpha=1, model=model)
    # subplots[-1].legend(loc=4)
    fig.tight_layout()
    fig.savefig('figures/j2_mut_effects_ej2_bcs.{}.png'.format(model), dpi=300)
    