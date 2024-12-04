#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import FIG_WIDTH


if __name__ == "__main__":
    # Load predictions at genotype and plant levels
    plant_data = pd.read_csv("data/plant_data.csv", index_col=0)
    cols = ["PLT3", "PLT7", "J2", "EJ2", "Season"]
    gt_data = (
        plant_data.loc[plant_data["influorescences"] == 10, :]
        .groupby(cols)["obs_mean"]
        .agg(("mean", "var"))
        .reset_index()
    )

    # Init figure
    fig, subplots = plt.subplots(
        1, 2, figsize=(0.7 * FIG_WIDTH, 0.35 * FIG_WIDTH)
    )

    # Plot genotype level predictions
    axes = subplots[0]
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
        "A",
        fontsize=12,
        fontweight="bold",
        transform=axes.transAxes,
    )

    # Plot genotype level predictions
    axes = subplots[1]
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
        "B",
        fontsize=12,
        fontweight="bold",
        transform=axes.transAxes,
    )

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0)
    fname = "figures/FigureS4"
    # fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.pdf".format(fname))
