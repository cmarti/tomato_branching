#!/usr/bin/env python
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.settings import FIG_WIDTH


def plot_epistatic_square(axes, pred, gts, labels, dx2=-0.2, dx3=-0.2):
    df = pred.loc[gts, :]
    x = df["x"].values
    logy = df["multilinear_pred"]
    y = np.exp(logy).values
    axes.scatter(
        x,
        y,
        c="black",
        zorder=10,
        s=7,
        lw=0.2,
        edgecolor="white",
    )
    cmap = cm.get_cmap("binary")
    idxs = [[0, 1], [0, 2], [1, 3], [2, 3]]
    c1, c2 = cmap(0.4), cmap(0.7)
    colors = [c1, c2, c2, c1]
    for idx, c in zip(idxs, colors):
        axes.plot(x[idx], y[idx], lw=0.75, c=c)

    expected = np.exp(logy[0] + (logy[1] - logy[0]) + (logy[2] - logy[0]))
    axes.plot(x[[1, 3]], [y[1], expected], lw=0.5, c=c2, linestyle="--", alpha=0.8)
    axes.plot(x[[2, 3]], [y[2], expected], lw=0.5, c=c1, linestyle="--", alpha=0.8)
    axes.scatter(x[-1], expected, c="grey", s=7, lw=0.2, edgecolor="white")

    axes.text(x[1] + 0.2, y[1] * 0.9, labels[1], va="top", ha="left", fontsize=6)
    axes.text(x[2] + dx2, y[2] * 1.1, labels[2], va="bottom", ha="right", fontsize=6)
    axes.text(x[3] + dx3, y[3] * 1.1, labels[3], va="bottom", ha="left", fontsize=6)

    axes.set(
        yscale="log",
        ylabel="branching events",
        xlabel="Mutational distance\nto wild-type",
        xticks=np.arange(8),
        ylim=lims,
        xlim=(-0.5, 7.5),
        title="",
    )


if __name__ == "__main__":
    # Init figure
    fig, subplots = plt.subplots(
        2,
        2,
        figsize=(FIG_WIDTH * 0.4, 0.4 * FIG_WIDTH),
        sharex=True,
        sharey=True,
    )

    encoding = {"W": 0, "H": 1, "M": 2}
    pred = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    pred["genotype"] = ["".join(x) for x in pred[["s1", "s2"]].values]
    pred = pred.groupby("genotype")[["multilinear_pred"]].mean()
    pred["x"] = [np.sum([encoding[a] for a in x[:4]]) for x in pred.index]

    lims = 0.05, 60

    axes = subplots[0][0]
    gts = [
        "WWWW",
        "WWWM8",
        "WWMW",
        "WWMM8",
    ]
    labels = ["", "$EJ2^{pro8}$", "$j2$", "$EJ2^{pro8}j2$"]
    plot_epistatic_square(axes, pred, gts, labels)
    axes.set(xlabel="", title="Paralogs synergy")

    axes = subplots[1][0]
    gts = [
        "WWWW",
        "WHWW",
        "MWWW",
        "MHWW",
    ]
    labels = ["", "$plt7/+$", "$plt3$", "$plt3\ plt7/+$"]
    plot_epistatic_square(axes, pred, gts, labels)

    axes = subplots[0][1]
    gts = [
        "WWWW",
        "WWMW",
        "MWWW",
        "MWMW",
    ]
    labels = ["", "$j2$", "$plt3$", "$plt3\ j2$"]
    plot_epistatic_square(axes, pred, gts, labels, dx3=0.3)
    axes.set(xlabel="", ylabel="", title="Epistatic masking")

    axes = subplots[1][1]
    gts = [
        "WWWW",
        "WWMM8",
        "MHWW",
        "MHMM8",
    ]
    labels = ["", "$EJ2^{pro8}j2$", "$plt3\ plt7/+$", "$EJ2^{pro8}j2\ plt3\ plt7/+$"]
    plot_epistatic_square(axes, pred, gts, labels, dx2=0.7, dx3=-6.5)
    axes.set(ylabel="")

    fname = "figures/Figure4H".format()
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.svg".format(fname))
