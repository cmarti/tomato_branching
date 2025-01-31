#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import SEASONS
                        

if __name__ == '__main__':
    model_label = 'pairwise'
    
    print('Plotting predictions of a season using all other seasons')
    data = pd.read_csv('results/leave_season_out_results.{}.csv'.format(model_label), index_col=0)

    fig, subplots = plt.subplots(1, 4, figsize=(11, 3.25), sharex=True, sharey=True)
    ylabel = True
    for axes, season in zip(subplots, SEASONS):
        print('\tSeason: {}'.format(season))
        season_data = data.loc[data['Season'] == season, :]
        x, y = season_data['mean'], season_data['obs_mean']
        axes.scatter(x, y, c='black', lw=0, alpha=0.2, s=3 * season_data['n_plants'])
        lims = (1e-2, 2e2)
        axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--', alpha=0.5)
        axes.set(xlabel='Predicted branching events',
                 ylabel='Average genotype branching events' if ylabel else '',
                 yscale='log', xscale='log',
                 xlim=lims, ylim=lims, title='{} (n={})'.format(season, season_data.shape[0]))
        axes.grid(alpha=0.2)
        ylabel = False
        
        log_season_data = np.log(season_data[['mean', 'obs_mean']].replace(0., np.nan).dropna())
        x, y = log_season_data['mean'], log_season_data['obs_mean']
        r2 = pearsonr(x, y)[0] ** 2
        axes.text(0.95, 0.05, r'R$^2$=' + '{:.2f}'.format(r2),
                  transform=axes.transAxes, ha='right', va='bottom')

    # Re-arrange and save figure
    fig.tight_layout()
    fname = 'figures/leave_season_out.{}'.format(model_label)
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.pdf'.format(fname))
