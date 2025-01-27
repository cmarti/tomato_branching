#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator
from scripts.settings import EJ2_SERIES, LIMS

        
        
def plot(df, axes, target, col='M', ref='W', color='black', alpha=1, label=None):
    df = np.exp(df.dropna(subset=[ref, col]))
    x, y = df[ref], df[col]
    
    dx = np.abs(df[['{}_lower'.format(ref), '{}_upper'.format(ref)]].T - x)
    dy = np.abs(df[['{}_lower'.format(col), '{}_upper'.format(col)]].T - y)
    
    axes.errorbar(x, y, xerr=dx, yerr=dy, lw=0.3, alpha=alpha, ecolor=color, fmt='none')    
    axes.scatter(x, y, c=color, alpha=alpha, s=4.5, lw=0.2, label=label, edgecolor='white')
    
    axes.axline((1, 1), (2, 2), lw=0.5, c='grey', linestyle='--', alpha=alpha)
    
    axes.set(xlabel='', ylabel='',
             xlim=LIMS, ylim=LIMS,
             xscale='log', yscale='log',
             aspect='equal',
             )
    
    major_locator = LogLocator(base=10.0, numticks=10)
    minor_locator = LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)

    axes.xaxis.set_major_locator(major_locator)
    axes.xaxis.set_minor_locator(minor_locator)
    axes.yaxis.set_major_locator(major_locator)
    axes.yaxis.set_minor_locator(minor_locator)

    # Show gridlines
    axes.grid(alpha=0.2, lw=0.3)
    

def add_model_line(axes, theta, gt):
    point1 = (np.exp(theta.loc['WW', 'v1']), np.exp(theta.loc[gt, 'v1']))
    point2 = (np.exp(theta.loc['WW', 'v2']), np.exp(theta.loc[gt, 'v2']))
    axes.axline(point1, point2, linestyle='--', lw=0.75, c='black', label='Model')


if __name__ == '__main__':
    cols = ['EJ2 variant', 'EJ2', 'J2', 'PLT3', 'PLT7', 'Season']
    print('Plotting effects of mutations across backgrounds')
    gt_data = pd.read_csv('results/genotype_means_estimates.csv', index_col=0)
    gt_data = gt_data.loc[gt_data['stderr'] < 10, :]
    theta1 = pd.read_csv('results/bilinear_model1.theta1.csv', index_col=0)
    theta2 = pd.read_csv('results/bilinear_model1.theta2.csv', index_col=0)
    theta2['gt'] = [x[:2] for x in theta2.index]
    theta2av = theta2.groupby('gt').mean()
    
    # Init figure
    fig, subplots = plt.subplots(2, 3, figsize=(2.3 * 3, 2.5 * 2), sharex=True, sharey=True)
    
    # PLT3-PLT7 effects
    cols1 = ['PLT3', 'PLT7']
    cols2 = ['J2', 'EJ2', 'Season']
    gt_data['mutants'] = ['{}{}'.format(*x) for x in gt_data[cols1].values]
    gt_data['background'] = ['{}{}-{}'.format(*x) for x in gt_data[cols2].values]
    
    df = pd.pivot_table(gt_data, columns='mutants', values='estimate', index='background')
    df = df.join(pd.pivot_table(gt_data, columns='mutants', values='lower', index='background'), rsuffix='_lower')
    df = df.join(pd.pivot_table(gt_data, columns='mutants', values='upper', index='background'), rsuffix='_upper')
    
    axes = subplots[0, 0]
    plot(df, axes, target='Mutant', col='HW', ref='WW', color='black')
    add_model_line(axes, theta1, gt='HW')
    axes.set_title('$plt3/+$')
    axes.legend(loc=2)
    
    axes = subplots[0, 1]
    plot(df, axes, target='$plt3$', col='MW', ref='WW', color='black')    
    add_model_line(axes, theta1, gt='MW')
    axes.set_ylabel('')
    axes.set_title('$plt3$')
    
    axes = subplots[0, 2]
    plot(df, axes, target='$plt3\ plt7/+$', col='MH', ref='WW', color='black')    
    add_model_line(axes, theta1, gt='MH')
    axes.set_ylabel('')
    axes.set_title('$plt3\ plt7/+$')
    
    
    # PLT3-PLT7 effects
    cols1 = ['J2', 'EJ2']
    cols2 = ['PLT3', 'PLT7', 'Season']
    gt_data['mutants'] = ['{}{}'.format(*x) for x in gt_data[cols1].values]
    gt_data['background'] = ['{}{}-{}'.format(*x) for x in gt_data[cols2].values]
    
    df = pd.pivot_table(gt_data, columns='mutants', values='estimate', index='background')
    df = df.join(pd.pivot_table(gt_data, columns='mutants', values='lower', index='background'), rsuffix='_lower')
    df = df.join(pd.pivot_table(gt_data, columns='mutants', values='upper', index='background'), rsuffix='_upper')
    
    axes = subplots[1, 0]
    plot(df, axes, target='Mutant', col='MW', ref='WW', color='black')
    add_model_line(axes, theta2, gt='MW')
    axes.set_title('$j2$')
    
    axes = subplots[1, 1]
    for allele in EJ2_SERIES:
        plot(df, axes, target='', col='MH{}'.format(allele), ref='WW', color='black')    
        add_model_line(axes, theta2, gt='MH{}'.format(allele))
    axes.set_title('$j2\ EJ2^{pro}/+$')
    
    axes = subplots[1, 2]
    for allele in EJ2_SERIES:
        plot(df, axes, target='', col='MM{}'.format(allele), ref='WW', color='black')    
        add_model_line(axes, theta2, gt='MM{}'.format(allele))
    axes.set_title('$j2\ EJ2^{pro}$')


    subplots[0, 1].set_xlabel('Background branching events in J2/EJ2 combinations', fontsize=11)
    subplots[1, 1].set_xlabel('Background branching events in PLT3/PLT7 combinations', fontsize=11)
    fig.supylabel('Mutant branching events', ha='center', x=0.05, y=0.55, fontsize=11)

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0.05, h_pad=2.0)
    fname = 'figures/linear_effects'.format()
    fig.savefig('{}.png'.format(fname), dpi=300)
    fig.savefig('{}.svg'.format(fname))
    