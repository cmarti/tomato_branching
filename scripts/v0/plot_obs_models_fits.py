#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
                        

if __name__ == '__main__':
    print('Plotting model predictions')
    data = pd.read_csv('results/basic_models_predictions.csv', index_col=0)
    labels = {'ols_linear': 'Linear least squares',
              'ols_log': 'Log least squares',
              'poisson_linear': 'Poisson linear',
              'poisson_log': 'Poisson log',
              'nb_log': 'Negative Binomial'}
    n = len(labels)
    print(data)

    # Init figure
    fig, subplots = plt.subplots(2, n, figsize=(3 * n, 3 * 2))

    lims1 = (-5, 70)
    lims2 = (2e-2, 150)
    for axes1, axes2, (model, label) in zip(subplots[0], subplots[1], labels.items()):
        
        # Linear scale
        axes1.scatter(data[model], data['obs'], c='black', lw=0, alpha=0.2, s=2 * data['n_plants'])
        axes1.plot(lims1, lims1, lw=0.5, c='grey', linestyle='--', alpha=0.5)
        axes1.set(xlabel='Predicted branching events',
                 ylabel='Observed branching events' if model == 'ols_linear' else '',
                 xlim=(-5, 20) if 'linear' in model or 'ols' in model else (-5, 40), ylim=lims1, title=label)
        log_gt_data = data[[model, 'obs']].dropna()
        r2 = pearsonr(log_gt_data[model], log_gt_data['obs'])[0] ** 2
        axes1.text(0.05, 0.95, r'R$^2$=' + '{:.2f}'.format(r2), transform=axes1.transAxes, ha='left', va='top')
        axes1.grid(alpha=0.2)
        
        # Log scale
        axes2.scatter(data[model], data['obs'], c='black', lw=0, alpha=0.2, s=2 * data['n_plants'])
        axes2.plot(lims2, lims2, lw=0.5, c='grey', linestyle='--', alpha=0.5)
        axes2.set(xlabel='Predicted branching events',
                 ylabel='Observed branching events' if model == 'ols_linear' else '',
                 yscale='log', xscale='log',
                 xlim=lims2, ylim=lims2, title=label)
        log_gt_data = np.log(data[[model, 'obs']].replace(0., np.nan)).dropna()
        r2 = pearsonr(log_gt_data[model], log_gt_data['obs'])[0] ** 2
        axes2.text(0.05, 0.95, r'R$^2$=' + '{:.2f}'.format(r2), transform=axes2.transAxes, ha='left', va='top')
        axes2.grid(alpha=0.2)

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/obs_models_fit_linear'
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
    