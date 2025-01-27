#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
                        

if __name__ == '__main__':
    print('Plotting model predictions')
    data = pd.read_csv('results/basic_models_predictions.csv', index_col=0)
    labels = {'ols_linear': 'Linear least squares',
              'poisson_log': 'Multiplicative Poisson'}
    n = len(labels)

    # Init figure
    fig, subplots = plt.subplots(1, 3, figsize=(3.2 * 3, 3.2))

    lims = (1e-2, 1e2)
    lims = (-10, 150)
    ylim = (-5, 65)
    xlabel = 'Predicted branching events\nper inflorescence'
    ylabel = 'Average branching events\nper inflorescence'
    alpha = 0.2
    
    # OLS Linear scale
    model = 'ols_linear'
    label = 'Linear least squares'
    axes = subplots[0]
    axes.scatter(data[model], data['obs'], c='black', lw=0, alpha=alpha, s=2 * data['n_plants'])
    axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--', alpha=0.5)
    axes.set(xlabel=xlabel, ylabel=ylabel, 
             ylim=ylim, xlim=(-5, 10), title=label)
    gt_data = data[[model, 'obs']].dropna()
    r2 = pearsonr(gt_data[model], gt_data['obs'])[0] ** 2
    axes.text(0.05, 0.95, r'R$^2$=' + '{:.2f}'.format(r2), transform=axes.transAxes, ha='left', va='top')
    axes.grid(alpha=0.2)
    
    # Poisson-log Linear scale
    model = 'poisson_log'
    label = 'Multiplicative Poisson'
    axes = subplots[1]
    axes.scatter(data[model], data['obs'], c='black', lw=0, alpha=alpha, s=2 * data['n_plants'])
    axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--', alpha=0.5)
    axes.set(xlabel=xlabel, ylabel=ylabel, 
             ylim=ylim, xlim=(-5, 30), title=label,
             xticks=[0, 10, 20, 30])
    gt_data = data[[model, 'obs']].dropna()
    r2 = pearsonr(gt_data[model], gt_data['obs'])[0] ** 2
    axes.text(0.05, 0.95, r'R$^2$=' + '{:.2f}'.format(r2), transform=axes.transAxes, ha='left', va='top')
    axes.grid(alpha=0.2)
    
    # Poisson-log log scale
    axes = subplots[2]
    lims = (2e-2, 150)
    axes.scatter(data[model], data['obs'], c='black', lw=0, alpha=alpha, s=2 * data['n_plants'])
    axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--', alpha=0.5)
    axes.set(xlabel=xlabel, ylabel=ylabel, 
             yscale='log', xscale='log',
             xlim=lims, ylim=lims, title=label,
            #  xticks=[1e-1, 1e0, 1e1, 1e2]
             )
    log_gt_data = np.log(data[[model, 'obs']].replace(0., np.nan)).dropna()
    r2 = pearsonr(log_gt_data[model], log_gt_data['obs'])[0] ** 2
    axes.text(0.05, 0.95, r'R$^2$=' + '{:.2f}'.format(r2), transform=axes.transAxes, ha='left', va='top')
    axes.grid(alpha=0.2)

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/obs_models_fit_linear_talk'
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    