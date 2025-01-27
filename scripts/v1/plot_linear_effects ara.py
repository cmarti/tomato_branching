#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator
from scripts.settings import EJ2_SERIES, LIMS, FIG_WIDTH


def set_log_ticks(axes):
    major_locator = LogLocator(base=10.0, numticks=10)
    minor_locator = LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)

    axes.xaxis.set_major_locator(major_locator)
    axes.xaxis.set_minor_locator(minor_locator)
    axes.yaxis.set_major_locator(major_locator)
    axes.yaxis.set_minor_locator(minor_locator)


def set_aspect(axes, xlabel=None, ylabel=None):
    axes.set(
        aspect="equal",
        xscale='log',
        yscale='log',
        xlim=LIMS,
        ylim=LIMS,
        ylabel=ylabel,
        xlabel=xlabel,
    )
    set_log_ticks(axes)
    axes.grid(alpha=0.2, lw=0.3)
    axes.axline((1, 1), (2, 2), lw=0.3, c='grey', linestyle=':', alpha=1)
    
        
def plot(df, axes, target, col='M', ref='W', color='black', alpha=1, label=None):
    df = np.exp(df.dropna(subset=[ref, col]))
    x, y = df[ref], df[col]
    
    dx = np.abs(df[['{}_lower'.format(ref), '{}_upper'.format(ref)]].T - x)
    dy = np.abs(df[['{}_lower'.format(col), '{}_upper'.format(col)]].T - y)
    
    axes.errorbar(x, y, xerr=dx, yerr=dy, lw=0.15, alpha=alpha, ecolor=color, fmt='none')    
    axes.scatter(x, y, c=color, alpha=alpha, s=2.5, lw=0.2, label=label, edgecolor='white')
    set_aspect(axes)
    

def add_model_line(axes, theta, gt):
    point1 = (np.exp(theta.loc['WW', 'v1']), np.exp(theta.loc[gt, 'v1']))
    point2 = (np.exp(theta.loc['WW', 'v2']), np.exp(theta.loc[gt, 'v2']))
    axes.axline(point1, point2, linestyle='--', lw=0.5, c='black', label='Model')


if __name__ == '__main__':
    field = 'Env1 Frel median'
    # Load data
    print('Plotting effects of mutations across backgrounds')
    gt_data = pd.read_csv('data/arabinose.csv', index_col=0)
    print(gt_data.columns)
    gt_data['s1'] = [x.split('.')[0] for x in gt_data.index]
    gt_data['s2'] = [x.split('.')[1] for x in gt_data.index]
    df = pd.pivot_table(gt_data, columns='s1', values=field, index='s2')
    lims = gt_data[field].min()-0.1, gt_data[field].max() + 0.1
    mutations = [x for x in df.columns if x != 'WT']
    
    # Init figure
    fig, subplots = plt.subplots(4, 9, figsize=(FIG_WIDTH, 0.6 * FIG_WIDTH), sharex=True, sharey=True)
    subplots = subplots.flatten()

    for mutation, axes in zip(mutations, subplots):
        axes.scatter(df['WT'], df[mutation], c='black', s=2.5, lw=0.2, edgecolor='white')
        axes.axline((1, 1), (2, 2), lw=0.3, c='grey', linestyle='--', alpha=1)
        axes.set(aspect='equal', xlim=lims, ylim=lims)

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0.05, h_pad=0.15)
    fname = 'figures/linear_effects_arabinose'.format()
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.svg'.format(fname))
    