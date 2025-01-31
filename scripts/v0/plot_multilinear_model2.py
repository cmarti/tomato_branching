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
        figsize=(FIG_WIDTH * 0.55, 0.45 * FIG_WIDTH),
    )
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[0.5, 1, 0.05],
        height_ratios=[1, 0.5],
    )

    encoding = {"W": 0, "H": 1, "M": 2}
    pred = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    lims = 0.035, 25

    ax_joint = fig.add_subplot(gs[0, 1])
    ax_js = fig.add_subplot(gs[0, 0])  # , sharey=ax_joint)
    ax_plt = fig.add_subplot(gs[1, 1])  # , sharex=ax_joint)
    cbar_axes = fig.add_subplot(gs[0, 2])

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
        linewidths=0.3,
        linestyles="dashed",
    )

    gts = [
        "W_W_W_W_Summer 23",
        "M_H_W_W_Summer 22",
        "W_W_M_M8_Summer 23",
        "M_H_M_M8_Summer 22",
        "M_W_W_W_Summer 23",
        "W_W_M_W_Summer 23",
        "M_W_M_W_Summer 23",
    ]
    j2plt3 = pred.loc[gts, :]
    x = [
        "{}_{}_W_W_{}".format(*x)
        for x in j2plt3[["PLT3", "PLT7", "Season"]].values
    ]
    j2plt3["x"] = pred.reindex(x)["multilinear_pred"].values
    y = [
        "W_W_{}_{}_{}".format(*x)
        for x in j2plt3[["J2", "EJ2", "Season"]].values
    ]
    j2plt3["y"] = pred.reindex(y)["multilinear_pred"].values
    axes.scatter(
        np.exp(j2plt3["x"]),
        np.exp(j2plt3["y"]),
        c=j2plt3["multilinear_pred"],
        cmap="Blues",
        vmin=-2 * np.log(10),
        vmax=np.log(30),
        s=10,
        edgecolors="black",
        lw=0.5,
        zorder=10,
    )
    coords = np.exp(j2plt3[["x", "y"]].values)
    axes.plot(coords[[0, 1], 0], coords[[0, 1], 1], lw=0.5, c="black")
    axes.plot(coords[[0, 2], 0], coords[[0, 2], 1], lw=0.5, c="black")
    axes.plot(coords[[1, 3], 0], coords[[1, 3], 1], lw=0.5, c="black")
    axes.plot(coords[[2, 3], 0], coords[[2, 3], 1], lw=0.5, c="black")
    axes.plot(coords[[4, 6], 0], coords[[4, 6], 1], lw=0.5, c="black")
    axes.plot(coords[[5, 6], 0], coords[[5, 6], 1], lw=0.5, c="black")

    axes.text(
        coords[3, 0] * 0.9,
        coords[3, 1] * 0.9,
        "$plt3\ plt7/+$\n$EJ2^{pro8}\ j2$",
        ha="right",
        va="top",
        color="white",
        fontsize=6,
    )
    axes.text(
        coords[6, 0] * 0.9,
        coords[6, 1] * 1.1,
        "$plt3\ j2$",
        ha="right",
        va="bottom",
        color="white",
        fontsize=6,
    )

    # # cs.set_linestyle('solid')
    # def fmt(x):
    #     return "{}".format(x)

    # axes.clabel(cs, levels=[0], inline=True, fontsize=6, fmt=fmt)

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
    axes = ax_js
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
    label = "$EJ2^{pro8}$"
    idx = [x.endswith("W") or x.endswith(str("8")) for x in pred_j2["EJ2"]]
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
        d = np.sum([np.abs(c[a1] - c[a2]) for a1, a2 in zip(s1[:2], s2[:2])])
        if d == 1:
            sdf = df.loc[[s1, s2], :]
            axes.plot(
                sdf["x"], np.exp(sdf["multilinear_pred"]), lw=0.75, c="grey"
            )
    axes.set(
        yscale="log",
        ylabel="branching events in\n$PLT3\ PLT7$ background",
        # xlabel="Hamming distance\nto $PLT3\ PLT7$",
        xticks=[0, 1, 2, 3, 4],
        ylim=lims,
        xlim=(-0.5, 4.5),
        # aspect=4,
    )
    x, y = df.loc["WW", ["x", "multilinear_pred"]]
    axes.text(x, np.exp(y)*0.9, "$wt$", va="top", ha="left", fontsize=6)
    try:
        x, y = df.loc["WM8", ["x", "multilinear_pred"]]
        axes.text(x + 0.1, np.exp(y), label, va="top", ha="left", fontsize=6)
    except KeyError:
        pass
    x, y = df.loc["MW", ["x", "multilinear_pred"]]
    axes.text(x, np.exp(y), "$j2$", va="bottom", ha="right", fontsize=6)
    x, y = df.loc["MM8", ["x", "multilinear_pred"]]
    axes.text(
        x, np.exp(y), label + " $j2$", va="bottom", ha="right", fontsize=6
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
    x, y = pred_plt.loc["WW", ["x", "multilinear_pred"]]
    axes.text(np.exp(y)*0.9, x, "$wt$", va="bottom", ha="right", fontsize=6)
    
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
    ax_js.set(yticklabels=["", "", "$10^{-1}$", "$10^{0}$", "$10^1$"])
    ax = ax_plt.twinx()
    ax.set_ylabel("PLT Synergy", fontsize=8)
    ax.set_yticks([])
    ax_js.set_title("MADS synergy")

    gts = [
        "W_W_W_W_Summer 23",
        "M_W_W_W_Summer 22",
        "M_H_W_W_Summer 22",
        "W_W_W_W_Summer 23",
        "W_W_M_W_Summer 22",
        "W_W_M_M8_Summer 23",
    ]
    gts_values = np.exp(pred.loc[gts, "multilinear_pred"].values)
    print(gts_values)

    lw = 0.3
    alpha = 1

    for v in gts_values[:3]:
        ax_plt.axvline(v, lw=lw, c="black", linestyle="--", alpha=alpha)
        ax_joint.axvline(v, lw=lw, c="black", linestyle="--", alpha=alpha)

    for v in gts_values[3:]:
        ax_js.axhline(v, lw=lw, c="black", linestyle="--", alpha=alpha)
        ax_joint.axhline(v, lw=lw, c="black", linestyle="--", alpha=alpha)

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0, h_pad=-0.25)
    ax_plt.text(
        -0.37,
        0.5,
        "Mutational\nDistance\nto\nwild-type",
        va="center",
        ha="center",
        transform=ax_plt.transAxes,
        fontsize=7,
    )

    fname = "figures/multilinear_model".format()
    fig.savefig("{}.png".format(fname), dpi=600)
    fig.savefig("{}.svg".format(fname))
    fig.savefig("{}.pdf".format(fname))
