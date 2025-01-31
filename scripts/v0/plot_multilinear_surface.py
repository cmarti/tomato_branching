#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    fig, subplots = plt.subplots(1, 2, figsize=(3.75 * 2, 3))

    axes = subplots[0]
    paralog1 = np.linspace(-1.5, 0.75, 100)
    paralog2 = np.linspace(-1.5, 0.75, 100)
    x, y = np.meshgrid(paralog2, paralog1)
    f = np.exp(x + y) - 1
    print(f.min(), f.max())

    labels = np.array([0.5, 1, 2, 4, 8, 16, 32])
    ticks = np.log(labels)
    im = axes.contourf(x, y, f, levels=50, cmap='viridis')
    plt.colorbar(im, label='Expected branching events')
    axes.set(xlabel='Paralog 1 dosage',
             ylabel='Paralog 2 dosage')
    fig.axes[-1].set(yticks=ticks, yticklabels=labels)


    # Between paralogs interaction
    axes = subplots[1]
    c = 1 / 4.5
    plts = np.linspace(-3, 2.5, 100)
    js = np.linspace(-3, 2.5, 100)

    x, y = np.meshgrid(plts, js)
    f = np.exp(x) + np.exp(y) - c * np.exp(x + y)  - 1
    print(f.min(), f.max())
    labels = np.array([0.5, 1, 2, 4, 8, 16, 32])
    ticks = np.log(labels)
    im = axes.contourf(x, y, f, levels=50, cmap='viridis')
    plt.colorbar(im, label='Expected branching events')

    axes.set(xlabel='PLT3/PLT7 dosage',
             ylabel='J2/EJ2 dosage')
    fig.axes[-1].set(yticks=ticks, yticklabels=labels)


    fig.tight_layout()
    fig.savefig('plots/surface_multilinear.png', dpi=300)