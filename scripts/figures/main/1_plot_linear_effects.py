#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.utils import plot_phenotypes_scatter
from scripts.settings import FIG_WIDTH


def add_labels(axes, df, gts, texts, dxs, dys, xs, ys):
    arrowprops = dict(
        color="black",
        shrinkB=0,
        shrinkA=10,
        width=0.05,
        lw=0.3,
        headwidth=1.2,
        headlength=2,
    )
    labels = df.loc[gts, :]
    for x, y, label, dx, dy in zip(labels[xs], labels[ys], texts, dxs, dys):
        xtext = np.exp(x + dx)
        ytext = np.exp(y + dy)
        ha = "center"
        axes.annotate(
            label,
            xy=(np.exp(x), np.exp(y)),
            ha=ha,
            xytext=(xtext, ytext),
            fontsize=5,
            arrowprops=arrowprops,
        )


def get_pivot_df(gt_data, use_plts=True):
    if use_plts:
        cols1 = ["PLT3", "PLT7"]
        cols2 = ["J2", "EJ2", "Season"]
    else:
        cols1 = ["J2", "EJ2"]
        cols2 = ["PLT3", "PLT7", "Season"]
        
    gt_data["mutants"] = ["{}{}".format(*x) for x in gt_data[cols1].values]
    gt_data["background"] = [
        "{}{}-{}".format(*x) for x in gt_data[cols2].values
    ]

    df = pd.pivot_table(
        gt_data, columns="mutants", values="saturated_pred", index="background"
    )
    df = df.join(
        pd.pivot_table(
            gt_data,
            columns="mutants",
            values="saturated_lower",
            index="background",
        ),
        rsuffix="_lower",
    )
    df = df.join(
        pd.pivot_table(
            gt_data,
            columns="mutants",
            values="saturated_upper",
            index="background",
        ),
        rsuffix="_upper",
    )
    return(df)


if __name__ == "__main__":
    np.random.seed(0)

    # Load data
    cols = ["EJ2 variant", "EJ2", "J2", "PLT3", "PLT7", "Season"]
    print("Plotting effects of mutations across backgrounds")
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    gt_data = gt_data.loc[
        (gt_data["saturated_upper"] - gt_data["saturated_lower"]) < np.log(1e3),
        :,
    ]

    # Init figure
    fig, subplots = plt.subplots(
        2,
        3,
        figsize=(0.725 * FIG_WIDTH, 0.425 * FIG_WIDTH),
        sharex=False,
        sharey=True,
    )

    print("\tPlot PLT3 PLT7 effects")
    df = get_pivot_df(gt_data, use_plts=True)

    axes = subplots[0, 0]
    plot_phenotypes_scatter(df, axes, col="HW")
    axes.set_ylabel("branching events in\n$plt3/+\ PLT7$ background")

    dxs = [-0.3, -2.4, 0.4]
    dys = [-2.75, 1.5, -4]
    gts = ["WW-Summer 22", "MW-Fall 22", "MM8-Summer 22"]
    texts = [
        "$EJ2\ J2$\nSummer 2022",
        "$EJ2\ j2$\nFall 2022",
        "$EJ2^{pro8}j2$\nSummer 2022",
    ]
    
    add_labels(axes, df, gts, texts, dxs, dys, xs='WW', ys='HW')

    axes = subplots[0, 1]
    plot_phenotypes_scatter(df, axes, col="MW")
    axes.set_ylabel("branching events in\n$plt3\ PLT7$ background")
    axes.set_xlabel("branching events in $PLT3\ PLT7$ background")

    gts = ["WW-Summer 22", "MW-Fall 22", "MM8-Summer 23"]
    dxs = [-0.75, -2, 0.5]
    dys = [-3.5, 2.25, -4.5]
    texts = [
        "$EJ2\ J2$\nSummer 2022",
        "$EJ2\ j2$\nFall 2022",
        "$EJ2^{pro8}j2$\nSummer 2023",
    ]
    add_labels(axes, df, gts, texts, dxs, dys, xs='WW', ys='MW')

    axes = subplots[0, 2]
    plot_phenotypes_scatter(df, axes, col="MH")
    axes.set_ylabel("branching events in\n$plt3\ plt7/+$ background")

    gts = ["WW-Summer 22", "MW-Fall 22", "MM8-Summer 22"]
    dxs = [-0.75, -0.5, 0.25]
    dys = [-3.5, 2, -3.5]
    texts = [
        "$EJ2\ J2$\nSummer 2022",
        "$EJ2\ j2$ Fall 2022",
        "$EJ2^{pro8}j2$\nSummer 2022",
    ]
    add_labels(axes, df, gts, texts, dxs, dys, xs='WW', ys='MH')

    print("\tPlot EJ2 J2 effects")
    df = get_pivot_df(gt_data, use_plts=False)

    axes = subplots[1, 0]
    plot_phenotypes_scatter(df, axes, col="MW")
    axes.set_ylabel("branching events in\n$EJ2\ j2$ background")

    gts = ["WW-Summer 22", "MW-Fall 22", "MH-Fall 22"]
    texts = [
        "$PLT3\ PLT7$\nSummer 2022",
        "$plt3\ PLT7$\nFall 2022",
        "$plt3\ plt7/+$\nFall 2022",
    ]
    dxs = [-0.5, -0.5, 0.5]
    dys = [-3, 2.25, -3.5]
    add_labels(axes, df, gts, texts, dxs, dys, xs='WW', ys='MW')

    axes = subplots[1, 1]
    plot_phenotypes_scatter(df, axes, col="MH8")
    axes.set_ylabel("branching events in\n$EJ2^{pro8}/+\ j2$ background")
    axes.set_xlabel("branching events in $EJ2\ J2$ background")

    gts = ["WW-Summer 22", "MW-Fall 22", "MH-Fall 22"]
    dxs = [0.8, -2.0, 0.5]
    dys = [-4, 1.6, -3.5]
    texts = [
        "$PLT3\ PLT7$\nSummer 2022",
        "$plt3\ PLT7$\nFall 2022",
        "$plt3\ plt7/+$\nFall 2022",
    ]
    add_labels(axes, df, gts, texts, dxs, dys, xs='WW', ys='MH8')

    axes = subplots[1, 2]
    plot_phenotypes_scatter(df, axes, col="MM8")
    axes.set_ylabel("branching events in\n$EJ2^{pro8}\ j2$ background")

    gts = ["WW-Summer 22", "MW-Summer 23", "MH-Summer 22"]
    dxs = [-0.5, -0.75, 0.5]
    dys = [-4, 1.1, -3.5]
    texts = [
        "$PLT3\ PLT7$\nSummer 2022",
        "$plt3\ PLT7$ Summer 2023",
        "$plt3\ plt7/+$\nSummer 2022",
    ]
    add_labels(axes, df, gts, texts, dxs, dys, xs='WW', ys='MM8')

    for axes in subplots.flatten():
        axes.axline((1, 1), (2, 2), lw=0.3, c="grey", linestyle="--", alpha=0.5)

    # Re-arrange and save figure
    print("Rendering figure")
    fig.tight_layout(w_pad=0.05, h_pad=0)
    fname = "figures/Figure4DE"
    fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.svg".format(fname))
