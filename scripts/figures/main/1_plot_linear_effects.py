#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.utils import plot_phenotypes_scatter, add_model_line
from scripts.settings import EJ2_SERIES, FIG_WIDTH


if __name__ == "__main__":
    # Load data
    cols = ["EJ2 variant", "EJ2", "J2", "PLT3", "PLT7", "Season"]
    print("Plotting effects of mutations across backgrounds")
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    gt_data = gt_data.loc[
        gt_data["saturated_upper"] - gt_data["saturated_lower"] < 10, :
    ]

    theta1 = pd.read_csv("results/multilinear.theta1.csv", index_col=0)
    theta2 = pd.read_csv("results/multilinear.theta2.csv", index_col=0)
    theta2["gt"] = [x[:2] for x in theta2.index]

    # Init figure
    fig, subplots = plt.subplots(
        2,
        3,
        # figsize=(0.725 * FIG_WIDTH, 0.475 * FIG_WIDTH),
        figsize=(0.775 * FIG_WIDTH, 0.5 * FIG_WIDTH),
        sharex=True,
        sharey=True,
    )

    print("\tPlot PLT3 PLT7 effects")
    cols1 = ["PLT3", "PLT7"]
    cols2 = ["J2", "EJ2", "Season"]
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

    axes = subplots[0, 0]
    plot_phenotypes_scatter(df, axes, col="HW", ref="WW", color="black")
    add_model_line(axes, theta1, gt="HW")
    axes.set_ylabel("branching events in\n$plt3/+\ PLT7$ background")
    axes.legend(loc=2)

    axes = subplots[0, 1]
    plot_phenotypes_scatter(df, axes, col="MW", ref="WW", color="black")
    add_model_line(axes, theta1, gt="MW")
    axes.set_ylabel("branching events in\n$plt3\ PLT7$ background")
    axes.set_xlabel("branching events in $PLT3\ PLT7$ background")

    axes = subplots[0, 2]
    plot_phenotypes_scatter(df, axes, col="MH", ref="WW", color="black")
    add_model_line(axes, theta1, gt="MH")
    axes.set_ylabel("branching events in\n$plt3\ PLT7$ background")

    print("\tPlot EJ2 J2 effects")
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

    axes = subplots[1, 0]
    plot_phenotypes_scatter(df, axes, col="MW", ref="WW", color="black")
    add_model_line(axes, theta2, gt="MW")
    axes.set_ylabel("branching events in\n$EJ2\ j2$ background")

    axes = subplots[1, 1]
    for allele in EJ2_SERIES:
        plot_phenotypes_scatter(
            df,
            axes,
            col="MH{}".format(allele),
            ref="WW",
            color="black",
        )
        add_model_line(axes, theta2, gt="MH{}".format(allele))
    axes.set_ylabel("branching events in\n$EJ2^{pro}/+\ j2$ background")
    axes.set_xlabel("branching events in $EJ2\ J2$ background")

    axes = subplots[1, 2]
    for allele in EJ2_SERIES:
        plot_phenotypes_scatter(
            df,
            axes,
            col="MM{}".format(allele),
            ref="WW",
            color="black",
        )
        add_model_line(axes, theta2, gt="MM{}".format(allele))
    axes.set_ylabel("branching events in\n$EJ2^{pro}\ j2$ background")

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0.05, h_pad=0)
    fname = "figures/linear_effects".format()
    fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.svg".format(fname))
