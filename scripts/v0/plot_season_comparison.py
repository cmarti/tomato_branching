#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import SEASONS
    

def pairplot(mean, err=None):
    lim = (-5, 5)
    bins = np.linspace(lim[0], lim[1], 25)
    
    fig, subplots = plt.subplots(4, 4, figsize=(11, 10))
    
    for i, season_i in enumerate(SEASONS):
        for j, season_j in enumerate(SEASONS):
            axes = subplots[i, j]
            if i == j:
                sns.histplot(mean[season_i].dropna(), bins=bins, color='grey', lw=0.5, edgecolor='black', ax=axes)
                axes.set(xlabel='', ylabel='', xlim=lim)
            else:
                m = mean[[season_j, season_i]].dropna()
                x, y = m[season_j], m[season_i]
                axes.scatter(x, y, c='black', alpha=0.5, s=15, lw=0)
                
                if err is not None:
                    max_err = err[[season_j, season_i]].max().max()
                    e = err[[season_j, season_i]].dropna().reindex(m.index).fillna(max_err) * 2
                    dx, dy = e[season_j], e[season_i]
                    axes.errorbar(x, y, xerr=dx, yerr=dy, lw=0.5, alpha=0.3, ecolor='black', fmt='none')
                axes.plot(lim, lim, linestyle='--', c='grey', lw=0.5)
                axes.set(xlim=lim, ylim=lim)
            
            axes.grid(alpha=0.2)
            if j == 0:
                axes.set(ylabel=season_i)
            else:
                axes.set(yticklabels=[])
            if i == 3:
                axes.set(xlabel=season_j)
            else:
                axes.set(xticklabels=[])
    fig.tight_layout()
    return(fig)
                    

if __name__ == '__main__':
    # Define aux variabbles
    cols = ['PLT3', 'PLT7', 'J2', 'EJ2']
    
    print('Comparing observed phenotypes across seasons')
    data = pd.read_csv('data/genotype_means.csv', index_col=0)
    data['label'] = [''.join([str(x) for x in y]) for y in data[cols].values]
    mean = np.log(pd.pivot_table(data, index='label', columns='Season', values='obs_mean').reset_index()[SEASONS])
    err = pd.pivot_table(data, index=cols, columns='Season', values='log_std_err').reset_index()
    fig = pairplot(mean, err[SEASONS] / 3.)
    fig.savefig('figures/seasons_data.png', dpi=300)
    fig.savefig('figures/seasons_data.pdf')
    
    model = 'dxe'
    print('Comparing predicted phenotypes across seasons under {} model'.format(model))
    data = pd.read_csv('results/model_predictions.genotypes.{}.csv'.format(model), index_col=0)
    mean = pd.pivot_table(data, index=cols, columns='Season', values='coef').reset_index()
    err = pd.pivot_table(data, index=cols, columns='Season', values='std err').reset_index()
    fig = pairplot(mean, err)
    fig.savefig('figures/seasons_pred.png', dpi=300)
    fig.savefig('figures/seasons_pred.pdf')
    
    print('Comparing predicted phenotypes across seasons under multilinear model')
    data = pd.read_csv('results/data.predictions.multilinear2.csv', index_col=0)
    mean = pd.pivot_table(data, index=cols, columns='Season', values='pred').reset_index()
    fig = pairplot(mean)
    fig.savefig('figures/seasons_pred.multilinear.png', dpi=300)
    fig.savefig('figures/seasons_pred.multilinear.pdf')
    