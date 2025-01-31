#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.settings import FIG_WIDTH
from scripts.utils import plot_phenotypes_scatter

if __name__ == "__main__":
    # Load data
    cols = ["EJ2 variant", "EJ2", "J2", "PLT3", "PLT7", "Season"]
    print("Plotting effects of mutations across backgrounds")
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    gt_data = gt_data.loc[
        gt_data["saturated_upper"] - gt_data["saturated_lower"] < 10, :
    ]

    # Init figure
    fig, subplots = plt.subplots(
        2,
        4,
        figsize=(FIG_WIDTH * 0.8, 0.45 * FIG_WIDTH),
    )
    subplots_flat = subplots.flatten()

    # PLT3-PLT7 effects
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

    genotypes = ["WH", "HW", "WM", "HH", "MW", "HM", "MH"]
    labels = [
        "$PLT3\ plt7/+$",
        "$plt3/+\ PLT7$",
        "$PLT3\ plt7$",
        "$plt3/+\ plt7/+$",
        "$plt3\ PLT7$",
        "$plt3/+\ plt7$",
        "$plt3\ plt7/+$",
    ]

    for gt, axes, label in zip(genotypes, subplots_flat, labels):
        plot_phenotypes_scatter(df, axes, col=gt, ref="WW", color="black",
                                add_diag=True, add_svd_line=True)
        axes.text(0.05, 0.95, label, transform=axes.transAxes, fontsize=6,
                      ha='left', va='top')
    
    # empty axes
    sns.despine(ax=subplots_flat[-1], bottom=True, left=True)
    subplots_flat[-1].set(xticks=[], yticks=[])
    subplots_flat[-1].minorticks_off()
    
    for i in range(1, 4):
        subplots[0, i].set(yticklabels=[], ylabel='')
        subplots[1, i].set(yticklabels=[], ylabel='')

    for i in range(4):
        subplots[0, i].set(xticklabels=[], xlabel='')

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0.2, h_pad=0.2)
    fig.subplots_adjust(left=0.1, bottom=0.12)
    fig.supxlabel('branching events in $PLT3\ PLT7$ background',
                  fontsize=8, x=0.55, y=0.025,
                  ha='center', va='center')
    fig.supylabel('branching events in $PLT3\ PLT7$ mutant background',
                  fontsize=8, x=0.025, y=0.52,
                  ha='center', va='center')
    
    fname = "figures/FigureS6"
    fig.savefig("{}.png".format(fname), dpi=300)
    # fig.savefig("{}.svg".format(fname))
    fig.savefig("{}.pdf".format(fname))
