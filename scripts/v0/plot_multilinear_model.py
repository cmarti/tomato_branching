#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.stats import pearsonr
from scripts.settings import FIG_WIDTH
from scripts.utils import set_aspect
from scripts.models.multilinear_model import MultilinearModel


def plot(df, axes, x, y, color="black", alpha=1, label=None):
    cols = ["{}_lower".format(y), "{}_upper".format(y)]
    x = df["{}_pred".format(x)]
    y = df["{}_pred".format(y)]
    r2 = pearsonr(np.log(x), np.log(y))[0] ** 2
    dy = np.abs(df[cols].T - y)
    axes.errorbar(x, y, yerr=dy, lw=0.1, alpha=alpha, ecolor=color, fmt="none")
    axes.scatter(
        x,
        y,
        c=color,
        alpha=alpha,
        s=2,
        lw=0.1,
        label=label,
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


if __name__ == "__main__":
    # Load data
    print("Plotting effects of mutations across backgrounds")
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    stderr = (gt_data["saturated_upper"] - gt_data["saturated_lower"]) / 2
    gt_data = gt_data.loc[stderr < 10, :]
    seasons = gt_data["Season"].values
    gt_data = np.exp(gt_data.iloc[:, 8:])
    # print(gt_data)

    gts = [
        "W_W_W_W_Summer 23",
        # "H_W_W_W_Summer 23",
        "M_W_W_W_Summer 23",
        # "M_H_W_W_Fall 22",
    ]
    gts_values = gt_data.loc[gts, "bilinear_pred"]

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

    # Init figure
    fig, subplots = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH * 0.75, 0.31 * FIG_WIDTH),
        width_ratios=[1, 1, 0.05],
    )

    axes = subplots[0]
    model = "bilinear"
    plot(gt_data, axes, model, "saturated")
    axes.set(xlim=(1e-2, 1e2), ylim=(1e-2, 1e2), title="Model predictions")
    axes.set_xlabel("Predicted branching events")
    axes.set_ylabel("Estimated branching events")

    axes = subplots[1]
    cbar_axes = subplots[2]
    im = axes.contourf(
        np.exp(x),
        np.exp(y),
        z,
        cmap="Blues",
        levels=50,
        vmin=-2,
        vmax=np.log10(30),
    )
    cs = axes.contour(
        np.exp(x),
        np.exp(y),
        z,
        colors="white",
        levels=12,
        vmin=0.5,
        linewidths=0.3,
        linestyles="dashed",
    )
    # cs.set_linestyle('solid')
    # axes.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

    axes.set(
        yscale="log",
        xscale="log",
        xlabel="branching events in\n$EJ2\ J2$ background",
        ylabel="branching events in\n$PLT3\ PLT7$ background",
        aspect="equal",
        xlim=(1e-2, 30),
        ylim=(1e-2, 30),
        title="Epistatic masking",
    )
    labels = ["$PLT3$", "$plt3$"]
    for v, label in zip(gts_values, labels):
        axes.axvline(v, lw=0.5, c="black", linestyle="--")
        axes.text(0.9 * v, 0.015, label, ha="right", va="center", fontsize=6)
    plt.colorbar(im, cax=cbar_axes, label="branching events")

    cbar_axes.set_ylim((-2.3, np.log10(30)))
    cbar_axes.set_yticks([-2, -1, 0, 1])
    cbar_axes.set_yticklabels(
        ["$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^{1}$"]
    )

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0)

    fname = "figures/model_preds".format()
    fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.svg".format(fname))
    fig.savefig("{}.pdf".format(fname))
