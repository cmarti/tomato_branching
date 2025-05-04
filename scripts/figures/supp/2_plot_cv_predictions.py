#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scripts.settings import FIG_WIDTH, SEASONS
from scripts.utils import set_aspect


def plot(df, axes, x, y, color="black", alpha=1, label=None):
    cols = ["{}_lower".format(y), "{}_upper".format(y)]
    x = df["{}_pred".format(x)]
    y = df["{}_pred".format(y)]
    r2 = pearsonr(np.log(x), np.log(y))[0] ** 2
    dy = np.abs(df[cols].T - y)
    axes.errorbar(x, y, yerr=dy, lw=0.15, alpha=alpha, ecolor=color, fmt="none")
    axes.scatter(
        x,
        y,
        c=color,
        alpha=alpha,
        s=2.5,
        lw=0.2,
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
    return r2


def add_model_label(axes, model_label):
    axes.text(
            0.95,
            0.05,
            model_label,
            transform=axes.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )


if __name__ == "__main__":
    # Load data
    print("Plotting leave-season-out and held out genotypes predictions")
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    gt_data = gt_data.loc[
        (gt_data["saturated_upper"] - gt_data["saturated_lower"]) < np.log(1e3),
        :,
    ]
    seasons = gt_data["Season"].values
    gt_data = np.exp(gt_data.iloc[:, 8:])
    gt = np.array(["_".join(x.split("_")[:-1]) for x in gt_data.index])
    held_out_gts = np.load("results/held_out_gts.npy", allow_pickle=True)
    idx = np.isin(gt, held_out_gts)
    cols = [x for x in gt_data.columns if x.endswith("pred")]

    # Init figure
    fig, subplots = plt.subplots(
        3, 5, figsize=(FIG_WIDTH, 0.65 * FIG_WIDTH), sharex=True, sharey=True
    )

    models = ["additive", "pairwise", "hierarchical"]
    labels = {}
    for model, row in zip(models, subplots):
        
        # Plot held-out genotypes
        r2s = []
        model_label = labels.get(model, model).capitalize()
        for season, axes in zip(SEASONS, row):
            df = gt_data.loc[seasons == season, :]
            r2 = plot(df, axes, model, "saturated")
            r2s.append(r2)
            add_model_label(axes, model_label)
            
        msg = "\t{} average leave-season-out r2 = {:.2f}"
        print(msg.format(model_label, np.mean(r2s)))

        # Plot held-out genotypes
        axes = row[-1]
        df = gt_data.loc[idx, :]
        plot(df, axes, "{}_train".format(model), "saturated")
        add_model_label(axes, model_label)

    # Add titles
    for axes, season in zip(subplots[0], SEASONS):
        axes.set(title=season)
    subplots[0][-1].set_title("10% genotypes")

    # Add axes labels
    fig.supxlabel("Predicted branching events", fontsize=8, x=0.55, y=0.04)
    fig.supylabel(
        "Estimated branching events in held out data",
        fontsize=8,
        x=0.04,
        y=0.55,
    )

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0.05, h_pad=0.15)
    fname = "figures/FigureS5".format()
    fig.savefig("{}.pdf".format(fname))
