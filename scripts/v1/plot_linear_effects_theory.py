#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.stats import pearsonr
from scripts.settings import FIG_WIDTH
from scripts.utils import set_aspect
from scripts.models.multilinear_model import MultilinearModel


if __name__ == "__main__":
    np.random.seed(0)
    # Init figure
    fig, axes = plt.subplots(
        1,
        1,
        figsize=(FIG_WIDTH * 0.28, 0.28 * FIG_WIDTH),
    )


    x = np.linspace(-4, 4, 100)

    y1 = x
    y2 = 1.5 + x

    phi = np.linspace(-6, 6, 100)
    x3 = -4 + 8 * np.exp(phi) / (1 + np.exp(phi))
    y3 = -4 + 8 * np.exp(phi + 2) / (1 + np.exp(phi + 2))

    x4 = np.linspace(-3.5, 3.5, 10)
    y4 = np.random.uniform(low=-4, high=4, size=10)
    lims = (np.exp(-4.1), np.exp(4.1))
    lw = 1

    axes.axline((0, 0), (1, 1), c='grey', lw=lw, label='No effect', linestyle='--')
    axes.axline((1, 3), (4, 12), c='black', lw=lw, label='Multiplicative effect')
    # axes.plot(np.exp(x), np.exp(y2), c='black', lw=lw, label='Multiplicative effect')
    axes.plot(np.exp(x3), np.exp(y3), c='darkred', lw=lw, label='Global epistasis')
    axes.scatter(np.exp(x4), np.exp(y4), c='darkblue', lw=lw, label='Idiosyncratic epistasis',
                 marker='+', s=10)
    axes.set(xscale='log',
             yscale='log', 
             xlabel='Wildtype phenotype',
             ylabel='Mutant phenotype',
             aspect='equal',
             xlim=lims, ylim=lims)
    axes.legend(loc=4, fontsize=5)

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0)

    fname = "figures/linear_effects_theory".format()
    fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.svg".format(fname))
    fig.savefig("{}.pdf".format(fname))
