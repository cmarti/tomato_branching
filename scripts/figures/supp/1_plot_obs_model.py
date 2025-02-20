#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    set_aspect(axes)
    axes.grid(alpha=0)


if __name__ == "__main__":
    # Init figure
    fig, subplots = plt.subplots(
        2, 2, figsize=(0.7 * FIG_WIDTH, 0.7 * FIG_WIDTH)
    )

    # Load predictions at genotype and plant levels
    plant_data = pd.read_csv("data/plant_data.csv", index_col=0)
    gt_data = pd.read_csv("data/gt_data.csv", index_col=0)
    plant_pred = pd.read_csv("results/plant_predictions.csv", index_col=0)
    deviance = pd.read_csv("results/models.deviance.csv", index_col=0)[
        "deviance"
    ]

    print("Plotting mean vs variance at the plant level")
    axes = subplots[0][0]
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
        "a",
        fontsize=12,
        fontweight="bold",
        transform=axes.transAxes,
    )

    print("Plotting mean vs variance at the genotype-season level")
    axes = subplots[0][1]
    lims = (-2, 1e2)
    axes.scatter(
        gt_data["obs_mean"],
        gt_data["variance"],
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
        "b",
        fontsize=12,
        fontweight="bold",
        transform=axes.transAxes,
    )

    print(
        "Plotting predictions of an additive Negative binomial model with identity-link function"
    )
    axes = subplots[1][0]
    plot(plant_pred, axes, x="nb_linear")
    axes.set(
        xlabel="Predicted plant branching events",
        ylabel="Plant average branching events",
        title="Identity link function",
        ylim=(0.01, 400),
        xlim=(0.01, 400),
    )
    dev = deviance.loc["negative binomial linear"]
    axes.text(
        0.05,
        0.95,
        "Deviance explained = {:.2f}%".format(dev),
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )

    axes.text(
        -0.4,
        1.05,
        "c",
        fontsize=12,
        fontweight="bold",
        transform=axes.transAxes,
    )

    print(
        "Plotting predictions of an additive Negative binomial model with log-link function"
    )
    axes = subplots[1][1]
    plot(plant_pred, axes, x="nb_log")
    axes.set(
        xlabel="Predicted plant branching events",
        ylabel="Plant average branching events",
        title="log link function",
        ylim=(0.01, 400),
        xlim=(0.01, 400),
    )
    dev = deviance.loc["negative binomial log"]
    axes.text(
        0.05,
        0.95,
        "Deviance explained = {:.2f}%".format(dev),
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    axes.text(
        -0.4,
        1.05,
        "d",
        fontsize=12,
        fontweight="bold",
        transform=axes.transAxes,
    )

    # Re-arrange and save figure
    print("Rendering")
    fig.tight_layout(w_pad=0, h_pad=0)
    fname = "figures/FigureS4"
    fig.savefig("{}.pdf".format(fname))
    print("Done")
