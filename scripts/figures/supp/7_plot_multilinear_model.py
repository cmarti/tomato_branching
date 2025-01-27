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
    fig = plt.figure(
        figsize=(FIG_WIDTH * 1.1, 0.55 * FIG_WIDTH),
    )
    gs = GridSpec(
        2,
        8,
        figure=fig,
        width_ratios=[0.5] * 6 + [1.2, 0.05],
        height_ratios=[1, 0.5],
    )

    encoding = {"W": 0, "H": 1, "M": 2}
    pred = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    lims = 0.035, 30

    ax_joint = fig.add_subplot(gs[0, 6])
    subplots_js = [fig.add_subplot(gs[0, i]) for i in range(6)]
    ax_plt = fig.add_subplot(gs[1, 6])  # , sharex=ax_joint)
    cbar_axes = fig.add_subplot(gs[0, 7])

    print("Plotting across paralogs interaction surface")
    axes = ax_joint
    model = torch.load("results/multilinear.pkl")
    wt = model.beta[0].detach().item()
    f1 = np.log(10 ** np.linspace(-2, np.log10(30), 100))
    f2 = np.log(10 ** np.linspace(-2, np.log10(30), 100))

    x, y = np.meshgrid(f1 - wt, f2 - wt)
    z = (
        model.bilinear_function(torch.Tensor(x), torch.Tensor(y), model.beta)
        .detach()
        .numpy()
    )
    x, y = np.meshgrid(f1, f2)
    z = z / np.log(10)

    im = axes.contourf(
        np.exp(x),
        np.exp(y),
        z,
        cmap="Blues",
        levels=200,
        vmin=-2,
        vmax=np.log10(30),
        linewidths=0,
        linestyles=None,
        rasterized=True,
        antialiased=False,
    )
    cs = axes.contour(
        np.exp(x),
        np.exp(y),
        z,
        colors="white",
        levels=15,  # [-1.5, -1, -0.5, 0.0, 0.5, 1, 1.5],
        vmin=0.5,
        linewidths=0.5,
        linestyles="dashed",
    )

    axes.set(
        yscale="log",
        xscale="log",
        xlabel="",
        ylabel="",
        # aspect="equal",
        xlim=lims,
        ylim=lims,
        yticklabels=[],
        xticklabels=[],
        title="Epistatic masking",
    )
    plt.colorbar(im, cax=cbar_axes, label="branching events")
    cbar_axes.set_ylim((-2, np.log10(30)))
    cbar_axes.set_yticks([-2, -1, 0, 1])
    cbar_axes.set_yticklabels(["$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^1$"])

    print("Plotting EJ2 J2 interaction")
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
    labels = [
        "$EJ2^{pro3}$",
        "$EJ2^{pro4}$",
        "$EJ2^{pro1}$",
        "$EJ2^{pro7}$",
        "$EJ2^{pro8}$",
        "$EJ2^{pro6}$",
    ]
    series = [3, 4, 1, 7, 8, 6]
    for axes, label, allele in zip(subplots_js, labels, series):
        idx = [
            x.endswith("W") or x.endswith(str(allele)) for x in pred_j2["EJ2"]
        ]
        df = pred_j2.loc[idx, :]
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
            d = np.sum(
                [np.abs(c[a1] - c[a2]) for a1, a2 in zip(s1[:2], s2[:2])]
            )
            if d == 1:
                sdf = df.loc[[s1, s2], :]
                axes.plot(
                    sdf["x"], np.exp(sdf["multilinear_pred"]), lw=0.75, c="grey"
                )
        axes.set(
            yscale="log",
            xticks=[0, 1, 2, 3, 4],
            ylim=lims,
            xlim=(-0.5, 4.5),
            # aspect=4,
        )
        if series != 3:
            axes.set(yticklabels=[])
        try:
            x, y = df.loc["WM{}".format(allele), ["x", "multilinear_pred"]]
            axes.text(
                x + 0.1, np.exp(y), label, va="top", ha="left", fontsize=6
            )
        except KeyError:
            pass
        x, y = df.loc["MW", ["x", "multilinear_pred"]]
        axes.text(x, np.exp(y), "$j2$", va="bottom", ha="right", fontsize=6)
        x, y = df.loc["MM{}".format(allele), ["x", "multilinear_pred"]]
        axes.text(
            x, np.exp(y), label + " $j2$", va="bottom", ha="right", fontsize=6
        )
    subplots_js[0].set(
        ylabel="branching events in\n$PLT3\ PLT7$ background",
    )
    subplots_js[0].set(yticklabels=["", "", "$10^{-1}$", "$10^{0}$", "$10^1$"])
    fig.suptitle(
        "MADS synergy",
        x=0.4,
        y=0.895,
        fontsize=8,
    )
    fig.supxlabel(
        "Mutational distance to wild-type",
        x=0.40,
        y=0.375,
        fontsize=7,
        ha="center",
    )

    print("Plotting PLT3 PLT7 interaction")
    # ax_plt.set(yticks=[])
    axes = ax_plt
    pred_plt = (
        pred.loc[pred["s2"] == "WW", :]
        .drop_duplicates(["PLT3", "PLT7", "J2", "EJ2"])
        .drop("Season", axis=1)
        .set_index("s1")
    )
    pred_plt["x"] = [
        encoding[x] + encoding[y] for x, y in pred_plt[["PLT7", "PLT3"]].values
    ]
    axes.scatter(
        np.exp(pred_plt["multilinear_pred"]),
        pred_plt["x"],
        c="black",
        zorder=10,
        s=7,
        lw=0.2,
        edgecolor="white",
    )
    c = {"W": 0, "H": 1, "M": 2}
    for s1, s2 in combinations(pred_plt.index, 2):
        d = np.sum([np.abs(c[a1] - c[a2]) for a1, a2 in zip(s1[:2], s2[:2])])
        if d == 1:
            df = pred_plt.loc[[s1, s2], :]
            axes.plot(
                np.exp(df["multilinear_pred"]), df["x"], lw=0.75, c="grey"
            )
    axes.set(
        xscale="log",
        # ylabel="Hamming distance\nto $PLT3\ PLT7$",
        yticks=[0, 1, 2, 3, 4],
        xlim=lims,
        ylim=(-0.5, 4.5),
        # aspect=0.25,
        xlabel="branching events in\n$EJ2\ J2$ background",
    )

    x, y = pred_plt.loc["WM", ["x", "multilinear_pred"]]
    axes.text(np.exp(y), x, "$plt7$", va="bottom", ha="right", fontsize=6)
    x, y = pred_plt.loc["MW", ["x", "multilinear_pred"]]
    axes.text(np.exp(y), x, "$plt3$", va="top", ha="left", fontsize=6)
    x, y = pred_plt.loc["HM", ["x", "multilinear_pred"]]
    axes.text(
        np.exp(y), x, "$plt3/+\ plt7$", va="bottom", ha="right", fontsize=6
    )
    x, y = pred_plt.loc["MH", ["x", "multilinear_pred"]]
    axes.text(
        np.exp(y), x, "$plt3\ plt7/+$", va="bottom", ha="center", fontsize=6
    )

    ax_joint.set(xticklabels=[], yticklabels=[])
    ax_plt.set(xticklabels=["", "", "$10^{-1}$", "$10^{0}$", "$10^1$"])
    ax = ax_plt.twinx()
    ax.set_ylabel("PLT Synergy", fontsize=8)
    ax.set_yticks([])

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0, h_pad=-0.25)

    fname = "figures/multilinear_model_supp"
    fig.savefig("{}.png".format(fname), dpi=600)
    fig.savefig("{}.svg".format(fname))
    fig.savefig("{}.pdf".format(fname))
