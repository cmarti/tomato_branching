#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.settings import FIG_WIDTH, EJ2_SERIES
from scripts.utils import plot_phenotypes_scatter, add_model_line

if __name__ == "__main__":
    # Load data
    cols = ["EJ2 variant", "EJ2", "J2", "PLT3", "PLT7", "Season"]
    print("Plotting effects of mutations across backgrounds")
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    gt_data = gt_data.loc[
        gt_data["saturated_upper"] - gt_data["saturated_lower"] < 10, :
    ]
    theta = pd.read_csv("results/bilinear_model1.theta2.csv", index_col=0)

    # Init figure
    fig, subplots = plt.subplots(
        2,
        4,
        figsize=(FIG_WIDTH, 0.5 * FIG_WIDTH),
        # sharex=True,
        # sharey=True,
    )
    subplots = subplots.flatten()

    # PLT3-PLT7 effects
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

    genotypes = ["WH", "HW", "WM", "HH", "MW", "HM", "MH", "MM"]
    labels = [
        "$EJ2^{pro}/+\ J2$",
        "$EJ2\ j2/+$",
        "$EJ2^{pro}\ J2$",
        "$EJ2^{pro}/+\ j2/+$",
        "$EJ2\ j2$",
        "$EJ2^{pro}\ j2/+$",
        "$EJ2^{pro}/+\ j2$",
        "$EJ2^{pro}\ j2$",
    ]

    show_legend = True
    for gt, axes, label in zip(genotypes, subplots, labels):
        if gt[-1] != "W":
            for allele in EJ2_SERIES:
                gt_allele = "{}{}".format(gt, allele)
                plot_phenotypes_scatter(
                    df,
                    axes,
                    col=gt_allele,
                    ref="WW",
                    color="black",
                    add_diag=allele == EJ2_SERIES[0],
                )
                add_model_line(axes, theta, gt=gt_allele)

                if show_legend:
                    axes.legend(loc=2)
                    show_legend = False

        else:
            plot_phenotypes_scatter(df, axes, col=gt, ref="WW", color="black")
            add_model_line(axes, theta, gt=gt)

        axes.set(
            xlabel="$EJ2\ J2$\nbranching events",
            ylabel=label + "\nbranching events",
        )

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0.05, h_pad=0.15)
    fname = "figures/FigureS6"
    # fig.savefig("{}.png".format(fname), dpi=300)
    # fig.savefig("{}.svg".format(fname))
    fig.savefig("{}.pdf".format(fname))
