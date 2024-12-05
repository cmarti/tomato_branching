#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import FIG_WIDTH
from scripts.utils import set_aspect


def plot(df, axes, x, color="black", alpha=0.1, label=None):
    sdf = df[[x, "obs_mean"]]
    sdf = sdf.loc[np.all(sdf > 0, 1), :]

    x, y = sdf[x], sdf["obs_mean"]
    axes.scatter(
        x,
        y,
        c=color,
        alpha=alpha,
        s=2.5,
        lw=0,
        label=label,
    )

    r2 = pearsonr(np.log(x), np.log(y))[0] ** 2
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
    axes.grid(alpha=0)


if __name__ == "__main__":
    # Init figure
    fig, subplots = plt.subplots(
        2, 2, figsize=(0.7 * FIG_WIDTH, 0.7 * FIG_WIDTH)
    )

    print("Load genotype estimates data")
    gt_data = pd.read_csv("results/plant_predictions.csv")

    axes = subplots[0][0]
    plot(gt_data, axes, x="poisson_linear")
    axes.set(
        xlabel="Predicted plant branching events",
        ylabel="Plant average branching events",
        title="Identity link function",
        ylim=(0.01, 400),
        xlim=(0.01, 400),
    )
    axes.text(
        -0.4,
        1.05,
        "A",
        fontsize=12,
        fontweight="bold",
        transform=axes.transAxes,
    )

    axes = subplots[0][1]
    plot(gt_data, axes, x="poisson_log")
    axes.set(
        xlabel="Predicted plant branching events",
        ylabel="Plant average branching events",
        title="log link function",
        ylim=(0.01, 400),
        xlim=(0.01, 400),
    )
    axes.text(
        -0.4,
        1.05,
        "B",
        fontsize=12,
        fontweight="bold",
        transform=axes.transAxes,
    )

    # Load predictions at genotype and plant levels
    plant_data = pd.read_csv("data/plant_data.csv", index_col=0)
    cols = ["PLT3", "PLT7", "J2", "EJ2", "Season"]
    gt_data = (
        plant_data.loc[plant_data["influorescences"] == 10, :]
        .groupby(cols)["obs_mean"]
        .agg(("mean", "var"))
        .reset_index()
    )

    # Plot genotype level predictions
    axes = subplots[1][0]
    lims = (-2, 1e2)
    axes.scatter(
        plant_data["obs_mean"],
        plant_data["variance"],
        c="black",
        lw=0,
        alpha=0.2,
        s=2,
    )
    axes.plot(lims, lims, lw=0.5, c="grey", linestyle="--", alpha=0.5)
    axes.set(
        xlabel="Plant average\nbranching events",
        ylabel="Plant variance\nin branching events",
        xlim=lims,
        ylim=lims,
        aspect="equal",
    )
    axes.text(
        -0.4,
        1.05,
        "C",
        fontsize=12,
        fontweight="bold",
        transform=axes.transAxes,
    )

    # Plot genotype level predictions
    axes = subplots[1][1]
    lims = (-2, 1e2)
    axes.scatter(
        gt_data["mean"],
        gt_data["var"],
        c="black",
        lw=0,
        alpha=0.2,
        s=2,
    )
    axes.plot(lims, lims, lw=0.5, c="grey", linestyle="--", alpha=0.5)
    axes.set(
        xlabel="Genotype-season average\nbranching events",
        ylabel="Genotype-season variance\nin branching events",
        xlim=lims,
        ylim=lims,
        aspect="equal",
    )
    axes.text(
        -0.4,
        1.05,
        "D",
        fontsize=12,
        fontweight="bold",
        transform=axes.transAxes,
    )

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0, h_pad=0)
    fname = "figures/FigureS4"
    # fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.pdf".format(fname))
