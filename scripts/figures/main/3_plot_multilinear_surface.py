#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from scripts.models.hierarchical_model import HierarchicalModel
from scripts.settings import FIG_WIDTH
from scripts.utils import set_aspect


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


def get_wt_phenotypes(vmin=0.01, vmax=25):
    return np.log(np.geomspace(vmin, vmax, 25))


def calc_z(f1, f2, model, wt):
    xs_, ys_ = torch.Tensor(f1 - wt), torch.Tensor(f2 - wt)
    zs = model.bilinear_function(xs_, ys_, model.beta).detach().numpy()
    return zs


def plot_line(axes, f1, f2, model, wt, linestyle="--"):
    z = calc_z(f1, f2, model, wt)
    axes.plot(
        f1 / np.log(10),
        f2 / np.log(10),
        zs=np.exp(z),
        c="black",
        linestyle=linestyle,
        lw=0.5,
    )


if __name__ == "__main__":
    # Load data
    print("Plotting surface representing the hierarchical model")
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    stderr = (gt_data["saturated_upper"] - gt_data["saturated_lower"]) / 2
    gt_data = gt_data.loc[stderr < 10, :]
    gt_data["x"] = [
        "{}_{}_W_W_{}".format(plt3, plt7, season)
        for plt3, plt7, season in gt_data[["PLT3", "PLT7", "Season"]].values
    ]
    gt_data["x"] = gt_data.reindex(gt_data["x"].values)["saturated_pred"].values
    gt_data["y"] = [
        "W_W_{}_{}_{}".format(j2, ej2, season)
        for j2, ej2, season in gt_data[["J2", "EJ2", "Season"]].values
    ]
    gt_data["y"] = gt_data.reindex(gt_data["y"].values)["saturated_pred"].values
    gt_data["z"] = gt_data["saturated_pred"]

    # df = np.exp(gt_data[["x", "y", "z"]].dropna())
    df = gt_data[["x", "y", "z", "hierarchical_pred"]].dropna()

    model = torch.load("results/hierarchical.pkl")
    wt = model.beta[0].detach().item()
    f1 = get_wt_phenotypes()
    f2 = get_wt_phenotypes()
    xs, ys = np.meshgrid(f1, f2)
    zs = calc_z(xs, ys, model, wt)

    # Init figure
    fig = plt.figure(
        figsize=(FIG_WIDTH * 0.475, 0.425 * FIG_WIDTH),
    )

    # Plot surface
    axes = fig.add_subplot(projection="3d")
    axes.plot_wireframe(
        ys / np.log(10),
        xs / np.log(10),
        np.exp(zs),
        color="grey",
        lw=0.5,
        zorder=-10,
        alpha=0.35,
    )

    # Plot lines 1
    gts1 = [
        "W_W_W_W_Summer 22",
        "M_W_W_W_Summer 22",
        "M_H_W_W_Summer 22",
    ]
    values = df.loc[gts1, "hierarchical_pred"].values
    for v in values:
        f1_ = get_wt_phenotypes()
        f2_ = np.full_like(f1_, fill_value=v)
        plot_line(axes, f1_, f2_, model, wt, linestyle="--")

    v0 = df.loc["W_W_W_W_Summer 23", "hierarchical_pred"]
    v1 = df.loc["W_W_M_M8_Summer 23", "hierarchical_pred"]
    f1_ = get_wt_phenotypes(np.exp(v0), np.exp(v1))
    for v in [values[0], values[2]]:
        f2_ = np.full_like(f1_, fill_value=v)
        plot_line(axes, f1_, f2_, model, wt, linestyle="-")

    v1 = df.loc["W_W_M_W_Summer 23", "hierarchical_pred"]
    f1_ = get_wt_phenotypes(np.exp(v0), np.exp(v1))
    f2_ = np.full_like(f1_, fill_value=values[1])
    plot_line(axes, f1_, f2_, model, wt, linestyle="-")

    # Plot lines 2
    gts2 = [
        "W_W_W_W_Summer 23",
        "W_W_M_W_Summer 23",
        "W_W_M_M8_Summer 23",
    ]
    values = df.loc[gts2, "hierarchical_pred"].values
    for v in values:
        f2_ = get_wt_phenotypes()
        f1_ = np.full_like(f2_, fill_value=v)
        plot_line(axes, f1_, f2_, model, wt, linestyle="--")

    v0 = df.loc["W_W_W_W_Summer 22", "hierarchical_pred"]
    v1 = df.loc["M_H_W_W_Summer 22", "hierarchical_pred"]
    f2_ = get_wt_phenotypes(np.exp(v0), np.exp(v1))
    for v in [values[0], values[2]]:
        f1_ = np.full_like(f2_, fill_value=v)
        plot_line(axes, f1_, f2_, model, wt, linestyle="-")

    v1 = df.loc["M_W_W_W_Summer 22", "hierarchical_pred"]
    f2_ = get_wt_phenotypes(np.exp(v0), np.exp(v1))
    f1_ = np.full_like(f2_, fill_value=values[1])
    plot_line(axes, f1_, f2_, model, wt, linestyle="-")

    df["gt"] = ["".join(x.split("_")[:-1]) for x in df.index]
    df = df[["gt", "hierarchical_pred"]].drop_duplicates().set_index("gt")
    x = np.array(["{}WW".format(x[:2]) for x in df.index])
    df["x"] = df.loc[x, "hierarchical_pred"].values
    y = np.array(["WW{}".format(x[2:]) for x in df.index])
    df["y"] = df.loc[y, "hierarchical_pred"].values

    # Plot specific genotypes
    gts = np.array(
        [
            "WWWW",
            "MWWW",
            "WWMW",
            "MWMW",
            "MHWW",
            "WWMM8",
            "MHMM8",
            "WWWM8",
            "WHWW",
        ]
    )
    df = df.reindex(gts).dropna()
    axes.scatter(
        df["y"] / np.log(10),
        df["x"] / np.log(10),
        np.exp(df["hierarchical_pred"]),
        s=5,
        alpha=1,
        c="black",
        lw=0,
        zorder=10,
    )
    values = df["hierarchical_pred"]
    wt = values.loc["WWWW"]
    pred1 = values.loc["WWMW"] + (values.loc["WWWM8"] - values.loc["WWWW"])
    pred2 = values.loc["MWWW"] + (values.loc["WHWW"] - values.loc["WWWW"])

    axes.scatter(
        pred1 / np.log(10),
        wt / np.log(10),
        np.exp(pred1),
        s=5,
        alpha=1,
        c="grey",
        lw=0,
        zorder=10,
    )
    axes.scatter(
        wt / np.log(10),
        pred2 / np.log(10),
        np.exp(pred2),
        s=5,
        alpha=1,
        c="grey",
        lw=0,
        zorder=10,
    )

    axes.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Transparent
    axes.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Transparent
    axes.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Transparent

    axes.grid(False)
    axes.set(
        zlim=(None, 25),
        ylabel="branching events\nin $EJ2\ J2$ background",
        xlabel="branching events\nin $PLT3\ PLT7$ background",
        zlabel="branching events",
        xticks=[-2, -1, 0, 1],
        yticks=[-2, -1, 0, 1],
        xticklabels=["$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^{1}$"],
        yticklabels=["$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^{1}$"],
    )
    axes.view_init(elev=22, azim=-111)

    # fig.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=1, right=1)

    # plt.show()
    # print(f"Elevation: {axes.elev}")
    # print(f"Azimuth: {axes.azim}")

    fname = "figures/Figure4G".format()
    fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.svg".format(fname))
