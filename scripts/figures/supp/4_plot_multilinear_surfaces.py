#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from scripts.settings import FIG_WIDTH
from scripts.utils import set_aspect


def plot(df, axes, x, y, color="black", alpha=1, label=None):
    cols = ["{}_lower".format(y), "{}_upper".format(y)]
    x = df["{}_pred".format(x)]
    y = df["{}_pred".format(y)]
    r2 = pearsonr(np.log(x), np.log(y))[0] ** 2
    dy = np.abs(df[cols].T - y)
    axes.errorbar(x, y, yerr=dy, lw=0.1, alpha=alpha, ecolor=color, fmt="none")
    axes.scatter(
        x,
        y,
        c=color,
        alpha=alpha,
        s=2,
        lw=0.1,
        label=label,
        edgecolor="white",
    )
    axes.text(
        0.05,
        0.95,
        "R$^2$=" + "{:.2f}".format(r2),
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    set_aspect(axes)


if __name__ == "__main__":
    plt.rcParams["axes.labelsize"] = 6
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6

    # Load data
    print("Loading data")
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    stderr = (gt_data["saturated_upper"] - gt_data["saturated_lower"]) / 2
    gt_data = gt_data.loc[stderr < 10, :]
    gt_data["x"] = [
        "{}_{}_W_W_{}".format(plt3, plt7, season)
        for plt3, plt7, season in gt_data[["PLT3", "PLT7", "Season"]].values
    ]
    gt_data["x"] = gt_data.reindex(gt_data["x"].values)["saturated_pred"].values
    gt_data["y"] = [
        "W_W_{}_{}_{}".format(j2, ej2, season)
        for j2, ej2, season in gt_data[["J2", "EJ2", "Season"]].values
    ]
    gt_data["y"] = gt_data.reindex(gt_data["y"].values)["saturated_pred"].values
    gt_data["z"] = gt_data["saturated_pred"]

    # df = np.exp(gt_data[["x", "y", "z"]].dropna())
    df = gt_data[["x", "y", "z", "hierarchical_pred"]].dropna()

    print("Loading hierarchical model")
    xs = np.load("results/multilinear.xs.npy")
    ys = np.load("results/multilinear.ys.npy")
    zs = np.load("results/multilinear.zs.npy")

    xs = xs / np.log(10)
    ys = ys / np.log(10)

    # Init figure
    print("Plotting surface for the multilinear layer with observations")
    fig = plt.figure(
        figsize=(FIG_WIDTH * 1.5, 0.375 * FIG_WIDTH),
    )
    gs = GridSpec(1, 3)

    subplots = [
        fig.add_subplot(gs[0, 0], projection="3d"),
        fig.add_subplot(gs[0, 1], projection="3d"),
        fig.add_subplot(gs[0, 2], projection="3d"),
    ]

    for axes in subplots:
        df1 = df.loc[df["z"] < df["hierarchical_pred"], :]
        df2 = df.loc[df["z"] > df["hierarchical_pred"], :]

        axes.scatter(
            df1["y"] / np.log(10),
            df1["x"] / np.log(10),
            np.exp(df1["z"]),
            s=5,
            c="black",
            lw=0,
            zorder=0,
        )
        axes.plot_wireframe(
            ys, xs, np.exp(zs), color="darkred", lw=0.5, zorder=-10, alpha=0.5
        )
        axes.scatter(
            df2["y"] / np.log(10),
            df2["x"] / np.log(10),
            np.exp(df2["z"]),
            s=5,
            c="black",
            lw=0,
            zorder=10,
        )

        for axis in [axes.xaxis, axes.yaxis, axes.zaxis]:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Transparent
            # axis.set_tick_params(pad=0.5)  # Adjust X-axis tick label spacing

        axes.grid(False)
        axes.set(
            zlim=(None, 25),
            ylabel="branching events\nin $EJ2\ J2$ background",
            xlabel="branching events\nin $PLT3\ PLT7$ background",
            zlabel="branching events",
            xticks=[-2, -1, 0, 1],
            yticks=[-2, -1, 0, 1],
            xticklabels=["$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^{1}$"],
            yticklabels=["$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^{1}$"],
        )
    subplots[1].view_init(elev=21, azim=-103)
    subplots[0].view_init(elev=26, azim=120)

    fig.subplots_adjust(wspace=0.5)

    print("Rendering")
    fname = "figures/FigureS8a".format()
    fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.svg".format(fname))
    print("Done")
