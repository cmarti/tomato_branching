#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy.stats import pearsonr
from scripts.settings import FIG_WIDTH
from scripts.utils import set_aspect
from scripts.models.multilinear_model import MultilinearModel


if __name__ == "__main__":
    # Init figure
    fig, subplots = plt.subplots(
        2,
        1,
        figsize=(FIG_WIDTH * 0.25, 0.435 * FIG_WIDTH),
        sharex=True,
        sharey=True,
    )

    encoding = {"W": 0, "H": 1, "M": 2}
    pred = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    lims = 0.05, 50

    axes = subplots[0]
    gts = [
        "W_W_W_W_Summer 23",
        "M_W_W_W_Summer 23",
        "W_W_M_W_Summer 23",
        "M_W_M_W_Summer 23",
    ]
    df = pred.loc[gts, ["multilinear_pred"]]
    df["gt"] = ["WWWW", "MWWW", "WWMW", "MWMW"]
    df["x"] = [0, 2, 2, 4]
    df.set_index("gt", inplace=True)

    dy = np.exp(
        df.loc["MWWW", "multilinear_pred"] - df.loc["WWWW", "multilinear_pred"]
    )
    print("plt3 effect is {:.2f} times in wild-type background".format(dy))
    dy = np.exp(
        df.loc["MWMW", "multilinear_pred"] - df.loc["WWMW", "multilinear_pred"]
    )
    print("plt3 effect is {:.2f} times in j2 background".format(dy))

    axes.scatter(
        df["x"],
        np.exp(df["multilinear_pred"]),
        c="black",
        zorder=10,
        s=7,
        lw=0.2,
        edgecolor="white",
    )
    c = {"W": 0, "H": 1, "M": 2}
    for s1, s2 in combinations(df.index, 2):
        d = np.sum([np.abs(c[a1] - c[a2]) for a1, a2 in zip(s1, s2)])
        print(s1, s2, d)
        if d in (2,):
            sdf = df.loc[[s1, s2], :]
            axes.plot(
                sdf["x"], np.exp(sdf["multilinear_pred"]), lw=0.75, c="grey"
            )
    axes.set(
        yscale="log",
        ylabel="branching events",
        xlabel="",
        # xticks=np.arange(8),
        # xticklabels=[],
        ylim=lims,
        xlim=(-0.5, 7.5),
        title="Epistatic masking",
        # aspect=4,
    )

    x, y = df.loc["WWMW", ["x", "multilinear_pred"]]
    axes.text(x + 0.2, 1.1 * np.exp(y), "$j2$", va="top", ha="left", fontsize=6)
    x, y = df.loc["MWWW", ["x", "multilinear_pred"]]
    axes.text(
        x - 0.2,
        np.exp(y),
        "$plt3$",
        va="center",
        ha="right",
        fontsize=6,
    )
    x, y = df.loc["MWMW", ["x", "multilinear_pred"]]
    axes.text(
        x + 0.2,
        np.exp(y) * 0.9,
        "$plt3\ j2$",
        va="bottom",
        ha="left",
        fontsize=6,
    )

    axes = subplots[1]
    gts = [
        "W_W_W_W_Summer 23",
        "M_H_W_W_Summer 22",
        "W_W_M_M8_Summer 23",
        "M_H_M_M8_Summer 22",
    ]
    df = pred.loc[gts, ["multilinear_pred"]]
    df["gt"] = ["WWWW", "MHWW", "WWMM", "MHMM"]
    df["x"] = [0, 3, 4, 7]
    df.set_index("gt", inplace=True)

    dy = np.exp(
        df.loc["MHWW", "multilinear_pred"] - df.loc["WWWW", "multilinear_pred"]
    )
    print(
        "plt3 plt7/+ effect is {:.2f} times in wild-type background".format(dy)
    )
    dy = np.exp(
        df.loc["MHMM", "multilinear_pred"] - df.loc["WWMM", "multilinear_pred"]
    )
    print(
        "plt3 plt7/+ effect is {:.2f} times in EJ2pro8 j2 background".format(dy)
    )

    axes.scatter(
        df["x"],
        np.exp(df["multilinear_pred"]),
        c="black",
        zorder=10,
        s=7,
        lw=0.2,
        edgecolor="white",
    )
    c = {"W": 0, "H": 1, "M": 2}
    for s1, s2 in combinations(df.index, 2):
        d = np.sum([np.abs(c[a1] - c[a2]) for a1, a2 in zip(s1, s2)])
        print(s1, s2, d)
        if d in (3, 4):
            sdf = df.loc[[s1, s2], :]
            axes.plot(
                sdf["x"], np.exp(sdf["multilinear_pred"]), lw=0.75, c="grey"
            )
    axes.set(
        yscale="log",
        ylabel="branching events",
        xlabel="Mutational distance\nto wild-type",
        xticks=np.arange(8),
        ylim=lims,
        xlim=(-0.5, 7.5),
        # aspect=4,
    )

    x, y = df.loc["WWMM", ["x", "multilinear_pred"]]
    axes.text(x, np.exp(y), "$EJ2^{pro8}\ j2$", va="top", ha="left", fontsize=6)
    x, y = df.loc["MHWW", ["x", "multilinear_pred"]]
    # axes.text(
    #     x - 0.2,
    #     np.exp(y),
    #     "$plt3\ plt7/+$",
    #     va="center",
    #     ha="right",
    #     fontsize=6,
    # )
    x, y = df.loc["MHMM", ["x", "multilinear_pred"]]
    axes.text(
        x - 0.2,
        np.exp(y) * 0.9,
        "$plt3\ plt7/+\ EJ2^{pro8}\ j2$",
        va="bottom",
        ha="right",
        fontsize=6,
    )

    fname = "figures/masking_example".format()
    fig.tight_layout(h_pad=0.5)
    fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.svg".format(fname))
    fig.savefig("{}.pdf".format(fname))
