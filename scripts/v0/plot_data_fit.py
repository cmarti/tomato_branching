#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
                        

if __name__ == '__main__':
    model_label = 'dxe'
    
    print('Plotting model predictions')

    # Load predictions at genotype and plant levels
    plant_data = pd.read_csv('results/model_predictions.{}.csv'.format(model_label), index_col=0)
    gt_data = pd.read_csv('results/model_predictions.genotypes.{}.csv'.format(model_label), index_col=0)
    phi_pred = pd.read_csv('results/predictive_distribution.{}.csv'.format(model_label), index_col=0)

    # Init figure
    fig, subplots = plt.subplots(1, 2, figsize=(6.5, 3.25))

    # Plot plant level scatterplot
    axes = subplots[0]
    axes.plot(phi_pred['y'], phi_pred['y'], c='grey', alpha=0.5, lw=2)
    for q in [0.01, 0.25, 0.05, 0.1, 0.15, 0.2, 0.25]:
        lower = phi_pred['y_{}'.format(q)]
        upper = phi_pred['y_{}'.format(1-q)]
        axes.fill_between(phi_pred['y'], lower, upper, color='grey', alpha=0.1, lw=0)
    axes.scatter(plant_data['mean'], plant_data['obs_mean'], c='black', lw=0, alpha=0.1, s=2 * plant_data['influorescences'])
    axes.set(xscale='log',
             ylabel='Average plant branching events', 
             xlabel='Expected plant branching events',
             xlim=(0.01, 75), ylim=(-2.5, 65))
    axes.grid(alpha=0.2)

    # Plot genotype level predictions
    axes = subplots[1]
    axes.scatter(gt_data['mean'], gt_data['obs_mean'], c='black', lw=0, alpha=0.2, s=3 * gt_data['n_plants'])
    lims = (1e-2, 2e2)
    axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--', alpha=0.5)
    axes.set(xlabel='Predicted genotype branching events',
             ylabel='Average genotype branching events',
             yscale='log', xscale='log',
             xlim=lims, ylim=lims)

    # Add R2 to plot
    log_gt_data = np.log(gt_data[['mean', 'obs_mean']].replace(0., np.nan).dropna())
    r2 = pearsonr(log_gt_data['mean'], log_gt_data['obs_mean'])[0] ** 2
    axes.text(0.95, 0.05, r'R$^2$=' + '{:.2f}'.format(r2), transform=axes.transAxes, ha='right', va='bottom')
    axes.grid(alpha=0.2)

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/model_fit.{}'.format(model_label)
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    