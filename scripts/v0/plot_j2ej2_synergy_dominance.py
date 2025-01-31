#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import LogLocator
from scripts.settings import FIG_WIDTH, LIMS


def set_log_ticks(axes):
    major_locator = LogLocator(base=10.0, numticks=10)
    minor_locator = LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)

    axes.xaxis.set_major_locator(major_locator)
    axes.xaxis.set_minor_locator(minor_locator)
    axes.yaxis.set_major_locator(major_locator)
    axes.yaxis.set_minor_locator(minor_locator)


def set_aspect(axes, xlabel, ylabel):
    axes.set(
        aspect="equal",
        xscale='log',
        yscale='log',
        xlim=LIMS,
        ylim=LIMS,
        ylabel=ylabel,
        xlabel=xlabel,
    )
    set_log_ticks(axes)
    axes.grid(alpha=0.2)
    axes.axline((1, 1), (2, 2), lw=0.5, c='grey', linestyle=':', alpha=1)
    

if __name__ == "__main__":
    pred = pd.read_csv("results/bilinear_model1.j2ej2_pred.csv", index_col=0)
    pred["j2"] = [x[0] for x in pred.index]
    pred["ej2"] = [x[1:] for x in pred.index]
    pred = pd.pivot_table(pred, index="ej2", columns="j2", values="pred")
    print(np.log(pred))

    fig, subplots = plt.subplots(
        1, 3, figsize=(0.6 * FIG_WIDTH, 0.3 * FIG_WIDTH), sharex=True, sharey=True
    )
    s = 2.5

    df = pred.loc[[x.startswith("H") for x in pred.index], :]
    axes = subplots[1]
    axes.scatter(df["W"], df["M"], s=s, c="black", label="$j2$")
    axes.scatter(df["W"], df["H"], s=s, c="grey", label="$j2/+$")
    set_aspect(axes, ylabel="$EJ2^{pro}/+\ j2$ phenotype", xlabel="$EJ2^{pro}/+$ phenotype")
    axes.legend(loc=4)
    
    logdf = np.log(df)
    dy1 = np.exp((logdf["H"] - logdf["W"]).mean())
    dy2 = np.exp((logdf["M"] - logdf["W"]).mean())
    axes.axline((1, dy1), (2, 2 * dy1), linestyle="--", c="grey", lw=0.75)
    axes.axline((1, dy2), (2, 2 * dy2), linestyle="--", c="black", lw=0.75)

    df = pred.loc[[x.startswith("M") for x in pred.index], :]
    axes = subplots[2]
    axes.scatter(df["W"], df["H"], s=s, c="grey", label="$j2/+$")
    axes.scatter(df["W"], df["M"], s=s, c="black", label="$j2$")
    set_aspect(axes, ylabel="$EJ2^{pro}\ j2$ phenotype", xlabel="$EJ2^{pro}$ phenotype",)
    
    logdf = np.log(df)
    dy1 = np.exp((logdf["H"] - logdf["W"]).mean())
    dy2 = np.exp((logdf["M"] - logdf["W"]).mean())
    axes.axline((1, dy1), (2, 2 * dy1), linestyle="--", c="grey", lw=0.75)
    axes.axline((1, dy2), (2, 2 * dy2), linestyle="--", c="black", lw=0.75)

    axes = subplots[0]
    df = pred[["H", "W", "M"]].T
    rows0 = ["W"]
    rows1 = ["H{}".format(i) for i in [1, 3, 4, 6, 7, 8]]
    rows2 = ["M{}".format(i) for i in [1, 3, 4, 6, 7, 8]]
    x = (df[rows1].values).flatten()
    y = (df[rows2].values).flatten()
    axes.scatter(x, y, s=s, c="black", label="$j2$")
    set_aspect(axes, xlabel="$EJ2^{pro}/+$ phenotype",
        ylabel="$EJ2^{pro}$ phenotype",)
    axes.axline(np.exp([-2, -2]), np.exp([0, 2]), linestyle="--", c="black", lw=0.75)

    fig.tight_layout(w_pad=0.5)
    fig.savefig("figures/j2ej2_synergy_dominance.png", dpi=300)
    fig.savefig("figures/j2ej2_synergy_dominance.svg", dpi=300)
