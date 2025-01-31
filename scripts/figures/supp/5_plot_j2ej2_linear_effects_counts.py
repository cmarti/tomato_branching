#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.settings import FIG_WIDTH, EJ2_SERIES, EJ2_SERIES_LABELS
from scripts.utils import plot_phenotypes_scatter

if __name__ == "__main__":
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    
    # Load data
    cols = ["EJ2 variant", "EJ2", "J2", "PLT3", "PLT7", "Season"]
    print("Plotting effects of mutations across backgrounds")
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    gt_data = gt_data.loc[
        gt_data["saturated_upper"] - gt_data["saturated_lower"] < 10, :
    ]
    theta = pd.read_csv("results/bilinear_model1.theta2.csv", index_col=0)

    # PLT3-PLT7 effects
    cols1 = ["J2", "EJ2"]
    cols2 = ["PLT3", "PLT7", "Season"]
    gt_data["mutants"] = ["{}{}".format(*x) for x in gt_data[cols1].values]
    gt_data["background"] = [
        "{}{}-{}".format(*x) for x in gt_data[cols2].values
    ]

    df = np.log(
        pd.pivot_table(
            gt_data, columns="mutants", values="obs_mean", index="background"
        )
    )
    # Init figure
    fig, subplots = plt.subplots(
        8,
        len(EJ2_SERIES),
        figsize=(FIG_WIDTH, FIG_WIDTH * 1.25),
        sharex=True,
        sharey=True,
    )

    genotypes = ["WH", "HW", "WM", "HH", "HM", "MW", "MH", "MM"]
    j2_labels = {'W': r'$J2$', 'H': r'$j2/+$', 'M': r'$j2$'}

    show_legend = True
    for allele, ax_row, allele_label in zip(EJ2_SERIES, subplots.transpose(), EJ2_SERIES_LABELS):
        ej2_labels = {'W': '$EJ2$', 'H': allele_label + '/+', 'M': allele_label}
        
        for gt, axes in zip(genotypes, ax_row):
            label = ej2_labels[gt[1]] + ' ' + j2_labels[gt[0]]
            if gt[-1] != "W":
                gt = "{}{}".format(gt, allele)
            plot_phenotypes_scatter(df, axes, col=gt, ref="WW", color="black",
                                    add_diag=True, add_svd_line=True)
            axes.text(0.05, 0.95, label, transform=axes.transAxes, fontsize=6,
                      ha='left', va='top')

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0.05, h_pad=0.15)
    fig.subplots_adjust(bottom=0.065, left=0.075)
    fig.supxlabel('branching events in $EJ2\ J2$ background',
                  fontsize=8, x=0.55, y=0.025,
                  ha='center', va='center')
    fig.supylabel('branching events in $EJ2\ J2$ mutant background',
                  fontsize=8, x=0.025, y=0.5,
                  ha='center', va='center')
    fname = "figures/FigureS9"
    fig.savefig("{}.png".format(fname), dpi=300)
    # fig.savefig("{}.svg".format(fname))
    fig.savefig("{}.pdf".format(fname))
