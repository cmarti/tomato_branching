#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from scripts.settings import FIG_WIDTH, EJ2_SERIES


def load(fpath, ref=0):
    df = pd.read_csv(fpath, index_col=0)
    c = {"W": 0, "H": 1, "M": 2}
    df["x"] = [c[x[ref]] + 0.3 * c[x[ref - 1]] for x in df.index]
    return df


def plot(pred, axes, y="y"):
    axes.scatter(
        pred["x"],
        np.exp(pred[y]),
        c="black",
        zorder=10,
        s=7,
        lw=0.2,
        edgecolor="white",
    )
    c = {"W": 0, "H": 1, "M": 2}
    for s1, s2 in combinations(pred.index, 2):
        d = np.sum([np.abs(c[a1] - c[a2]) for a1, a2 in zip(s1[:2], s2[:2])])
        if d == 1:
            df = pred.loc[[s1, s2], :]
            axes.plot(df["x"], np.exp(df[y]), lw=0.75, c="grey")

    axes.set_ylim(0.05, 20)
    axes.set_xlim(-0.3, 4.4)


if __name__ == "__main__":
    pred = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    # print(
    #     pred.loc[
    #         pred["saturated_upper"] < pred["bilinear_pred"] - 1,
    #         ["bilinear_pred", "saturated_pred"],
    #     ]
    # )
    # exit()

    encoding = {"W": 0, "H": 1, "M": 2}

    pred_plt = (
        pred.loc[pred["s2"] == "WW", :]
        .drop_duplicates(["PLT3", "PLT7", "J2", "EJ2"])
        .drop("Season", axis=1)
        .set_index("s1")
    )
    pred_plt["x"] = [
        encoding[x] + encoding[y] for x, y in pred_plt[["PLT7", "PLT3"]].values
    ]

    pred_j2 = (
        pred.loc[pred["s1"] == "WW", :]
        .drop_duplicates(["PLT3", "PLT7", "J2", "EJ2"])
        .drop("Season", axis=1)
        .set_index("s2")
    )
    pred_j2["x"] = [
        encoding[x[0]] + encoding[y[0]]
        for x, y in pred_j2[["EJ2", "J2"]].values
    ]

    fig, subplots = plt.subplots(
        1, 7, figsize=(FIG_WIDTH, 1.8), sharex=True, sharey=True
    )

    print("Plotting PLT3 PLT7 interaction")
    axes = subplots[0]
    plot(pred_plt, axes, y="bilinear_pred")
    axes.set(
        yscale="log",
        ylabel="Estimated branching events",
        xlabel="Hamming distance\nto $PLT3\ PLT7$",
        xticks=[0, 1, 2, 3, 4],
    )
    axes.grid(alpha=0.2, lw=0.3)

    x, y = pred_plt.loc["WM", ["x", "bilinear_pred"]]
    axes.text(x, np.exp(y), "$plt7$", va="top", ha="left", fontsize=6)
    x, y = pred_plt.loc["MW", ["x", "bilinear_pred"]]
    axes.text(x, np.exp(y), "$plt3$", va="bottom", ha="right", fontsize=6)

    print("Plotting EJ2 J2 interaction")
    labels = [
        "$EJ2^{pro3}$",
        "$EJ2^{pro4}$",
        "$EJ2^{pro1}$",
        "$EJ2^{pro7}$",
        "$EJ2^{pro8}$",
        "$EJ2^{pro6}$",
    ]

    for allele, axes, label in zip(EJ2_SERIES, subplots[1:], labels):
        idx = [
            x.endswith("W") or x.endswith(str(allele)) for x in pred_j2["EJ2"]
        ]
        df = pred_j2.loc[idx, :]
        plot(df, axes, y="bilinear_pred")
        axes.set(
            yscale="log",
            ylabel=None,
            xlabel="Hamming distance\nto $EJ2\ J2$",
            ylim=(0.035, 40),
            xlim=(-0.5, 4.4),
        )
        # axes.text(
        #     0.05,
        #     0.95,
        #     label,
        #     fontsize=6,
        #     transform=axes.transAxes,
        #     ha="left",
        #     va="top",
        # )
        axes.grid(alpha=0.2, lw=0.3)
        try:
            x, y = df.loc["WM{}".format(allele), ["x", "bilinear_pred"]]
            axes.text(
                x + 0.1, np.exp(y), label, va="top", ha="left", fontsize=6
            )
        except KeyError:
            pass
        x, y = df.loc["MW", ["x", "bilinear_pred"]]
        axes.text(x, np.exp(y), "$j2$", va="bottom", ha="right", fontsize=6)
        x, y = df.loc["MM{}".format(allele), ["x", "bilinear_pred"]]
        axes.text(
            x, np.exp(y), label + " $j2$", va="bottom", ha="right", fontsize=6
        )

    fig.tight_layout(w_pad=0.75)
    fig.savefig("figures/paralogs_synergy.png", dpi=300)
    fig.savefig("figures/paralogs_synergy.svg", dpi=300)
