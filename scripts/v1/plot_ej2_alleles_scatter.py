#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.settings import EJ2_SERIES

if __name__ == "__main__":
    theta2 = pd.read_csv("results/bilinear_model1.theta2.csv", index_col=0)
    theta2["j2"] = [x[0] for x in theta2.index]
    theta2["ej2"] = [x[1:] for x in theta2.index]
    theta2 = theta2.loc[theta2['ej2'] != 'W', :]
    theta2['allele'] = [x[-1] for x in theta2['ej2']]
    theta2['gt'] = [x[0] + y for x, y in zip(theta2['ej2'], theta2['j2'])]
    theta2 = pd.pivot_table(theta2, index='gt', columns='allele', values='param')
    
    fig, subplots = plt.subplots(6, 6, figsize=(8, 8), sharex=True, sharey=True)
    lims = -3.25, 3.25
    ticks = np.arange(-3, 4)
    s = 5
    
    for i, allele1 in enumerate(EJ2_SERIES):
        for j, allele2 in enumerate(EJ2_SERIES):
            axes = subplots[i, j]
            if i == j:
                continue
            axes.scatter(theta2[allele2], theta2[allele1], s=s, c="black", label='$j2$')
            axes.grid(alpha=0.2)
            axes.axline((0, 0), (1, 1), linestyle="--", c='lightgrey', lw=0.75)
            axes.axvline(0, linestyle="--", c='lightgrey', lw=0.75)
            axes.axhline(0, linestyle="--", c='lightgrey', lw=0.75)
            axes.set(xlim=lims, ylim=lims)
    
    for i, allele in enumerate(theta2.columns):
        subplots[-1, i].set_xlabel('Allele {}'.format(allele))
        subplots[i, 0].set_ylabel('Allele {}'.format(allele))

    fig.suptitle('$EJ2^{pro}$ alleles', ha='center')
    fig.tight_layout()
    fig.savefig('figures/ej2_alleles_scatter.png', dpi=300)
    
