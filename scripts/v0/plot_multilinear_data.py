#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.stats import pearsonr
from scripts.settings import SEASONS

if __name__ == "__main__":
    # Load raw data
    data = pd.read_csv("results/genotype_means_estimates.csv", index_col=0)
    data = data.loc[data["stderr"] < 10, :]
    data.index = [
        "{}{}{}{}-{}".format(*x)
        for x in data[["PLT3", "PLT7", "J2", "EJ2", "Season"]].values
    ]

    phenotypes = data["estimate"].to_dict()
    data["plts"] = [
        phenotypes.get("{}{}WW-{}".format(*x), np.nan)
        for x in data[["PLT3", "PLT7", "Season"]].values
    ]
    data["js"] = [
        phenotypes.get("WW{}{}-{}".format(*x), np.nan)
        for x in data[["J2", "EJ2", "Season"]].values
    ]
    data = data.dropna()

    # theta = [-2.7784035, np.exp(-0.3377388), np.exp(-1.008262), 0.04332217]
    theta = [
        np.exp(-1.0544941),
        # np.exp(-0.9955431),
        # np.exp(-1.631103),
        1,
        1,
        0.01857191,
    ]
    # b = [-4.2162,  7.6368]
    theta = [2.459512, np.exp(0.368562), np.exp(0.720668), np.exp(1.019448)]
    theta = [0, 1, 1, 1]
    print(theta)

    x = np.linspace(-2, 1, 30)
    y = np.linspace(-2, 1, 30)
    xs, ys = np.meshgrid(x, y)

    zs = theta[0] + theta[1] * xs + theta[2] * ys - theta[3] * xs * ys
    # zs = b[0] + b[1]  * np.exp(zs) / (1 + np.exp(zs))
    xs = theta[0] + theta[1] * xs
    ys = theta[0] + theta[2] * ys

    # # fig, axes = plt.subplots(1, 1, figsize=(3.5, 3.5))
    # fig = plt.figure(figsize=(4.5, 3.5))
    # axes = fig.add_subplot(111)

    # vmin, vmax = np.log(0.125), np.log(64)
    # # axes.contourf(np.exp(x), np.exp(y), zs, cmap='Blues', vmin=vmin, vmax=vmax, levels=50)
    # x, y, z = data["plts"], data["js"], data["estimate"]
    # # axes.axvline(np.exp(phenotypes['WWWW-Summer 22']), zorder=-1, c='grey', lw=0.75)
    # # axes.axvline(np.exp(phenotypes['MWWW-Summer 22']), zorder=-1, c='grey', lw=0.75)

    # # axes.axhline(np.exp(phenotypes['WWWW-Summer 23']), zorder=-1, c='grey', lw=0.75)
    # # axes.axhline(np.exp(phenotypes['WWMM6-Summer 23']), zorder=-1, c='grey', lw=0.75)

    # sc = axes.scatter(np.exp(x), np.exp(y), lw=0.5, edgecolors='black',
    #                   s=15, c=z, vmin=vmin, vmax=vmax, cmap='Blues')
    # axes.set(xlabel='PLT3 PLT7 branching events',
    #          ylabel='$EJ2^{pro}$ J2 branching events',
    #          xscale='log', yscale='log',
    #          aspect='equal',
    #         #  xlim=(None, np.exp(vmax)), ylim=(None, np.exp(vmax)),
    #          )
    # axes.grid(alpha=0.2, lw=0.5)

    # plt.colorbar(sc, label='branching events', shrink=0.8)
    # ax = fig.axes[-1]
    # ticklabels = np.array([0.125, 0.25, 0.5, 1, 2., 4, 8, 16., 32., 64])
    # ax.set_yticks(np.log(ticklabels))
    # ax.set_yticklabels(ticklabels)

    # fig.tight_layout()
    # fig.savefig('figures/surface_pred.png', dpi=300)
    # exit()

    fig = plt.figure()
    gs = GridSpec(1, 2)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    # x, y, z = data["plts"], data["js"], data["log_mean"]
    # x, y, z = np.exp(x), np.exp(y), np.exp(z)
    # ax.scatter(x, y, z, lw=0, c="black")
    ax.set(
        ylabel="PLT3/PLT7 phenotype",
        xlabel="J2/EJ2 phenotype",
        zlabel="PLT3/PLT7/J2/EJ2 phenotype",
    )
    print(
        theta[2] / theta[3],
        theta[1] / theta[3],
        theta[0] + theta[1] * theta[2] / theta[3] ** 2,
    )
    print(zs.max())

    # xs = np.exp(xs)
    # ys = np.exp(ys)
    # zs = np.exp(zs)

    ax.plot_wireframe(ys, xs, np.exp(zs), color="darkred", lw=0.5)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Transparent
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Transparent
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Transparent
    ax.grid(False)

    ax = fig.add_subplot(gs[0, 1], projection="3d")
    ax.plot_wireframe(ys, xs, np.exp(zs), color="darkred", lw=0.5)
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Transparent
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Transparent
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Transparent
    plt.show()
    plt.savefig("plots/surface_intermediate_model.png", dpi=300)
    # plt.show()

    # for angle in range(0, 360 + 181):
    #     ax.view_init(20, angle, 0)
    #     plt.draw()
    #     plt.pause(.01)

    # angles = list(range(180, 240))
    # for angle in (angles + angles[::-1]) * 3:
    #     ax.view_init(20, angle, 0)
    #     plt.draw()
    #     plt.pause(.01)

    # fig.tight_layout()
    # fig.savefig('plots/3d_scatter.png', dpi=300)
