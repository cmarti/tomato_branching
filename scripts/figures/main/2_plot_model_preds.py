#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import FIG_WIDTH
from scripts.utils import set_aspect


if __name__ == "__main__":
    # Load data
    print("Plotting effects of mutations across backgrounds")
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    stderr = (gt_data["saturated_upper"] - gt_data["saturated_lower"]) / 2
    gt_data = gt_data.loc[stderr < 10, :]
    seasons = gt_data["Season"].values
    gt_data = np.exp(gt_data.iloc[:, 8:])

    # Init figure
    fig, axes = plt.subplots(
        1,
        1,
        figsize=(FIG_WIDTH * 0.3, 0.3 * FIG_WIDTH),
    )

    cols = ["saturated_lower", "saturated_upper"]
    x = gt_data["multilinear_pred"]
    y = gt_data["saturated_pred"]
    r2 = pearsonr(np.log(x), np.log(y))[0] ** 2
    dy = np.abs(gt_data[cols].T - y)
    color = "black"
    alpha = 1
    axes.errorbar(x, y, yerr=dy, lw=0.1, alpha=alpha, ecolor=color, fmt="none")
    axes.scatter(
        x,
        y,
        c=color,
        alpha=alpha,
        s=2,
        lw=0.1,
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

    axes.set(xlim=(1e-2, 1e2), ylim=(1e-2, 1e2), title="Model predictions")
    axes.set_xlabel("Predicted branching events")
    axes.set_ylabel("branching events")

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0)

    fname = "figures/Figure4F".format()
    fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.svg".format(fname))
