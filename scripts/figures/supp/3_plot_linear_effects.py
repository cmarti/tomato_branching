#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.settings import FIG_WIDTH, EJ2_SERIES, EJ2_SERIES_LABELS
from scripts.utils import plot_phenotypes_scatter


def pivot_gt_data(gt_data, cols1, cols2, use_mle=True):
    gt_data["mutants"] = ["{}{}".format(*x) for x in gt_data[cols1].values]
    gt_data["background"] = [
        "{}{}-{}".format(*x) for x in gt_data[cols2].values
    ]

    if use_mle:
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
    else:
        df = np.log(
            pd.pivot_table(
                gt_data, columns="mutants", values="obs_mean", index="background"
            )
        )
    return(df)


if __name__ == "__main__":
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    
    # Load data
    cols = ["EJ2 variant", "EJ2", "J2", "PLT3", "PLT7", "Season"]
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    gt_data = gt_data.loc[
        gt_data["saturated_upper"] - gt_data["saturated_lower"] < 10, :
    ]

    for use_mle, figlabel in [(True, 'S6'), (False, 'S7')]:
        # PLT3-PLT7 effects
        cols1 = ["J2", "EJ2"]
        cols2 = ["PLT3", "PLT7", "Season"]
        df = pivot_gt_data(gt_data, cols1, cols2, use_mle=use_mle)
        
        # Init figure
        print("Plotting effects of mutations across backgrounds")
        fig, subplots = plt.subplots(
            8,
            len(EJ2_SERIES) + 1,
            figsize=(FIG_WIDTH, FIG_WIDTH * 1.175),
            sharex=True,
            sharey=True,
        )

        genotypes = ["WH", "HW", "WM", "HH", "HM", "MW", "MH", "MM"]
        j2_labels = {'W': r'$J2$', 'H': r'$j2/+$', 'M': r'$j2$'}

        show_legend = True
        for allele, ax_row, allele_label in zip(EJ2_SERIES, subplots.transpose(), EJ2_SERIES_LABELS):
            ej2_labels = {'W': '$EJ2$', 'H': allele_label + '/+', 'M': allele_label}
            print('\tPlotting EJ2pro{}-J2'.format(allele))
            for gt, axes in zip(genotypes, ax_row):
                label = ej2_labels[gt[1]] + ' ' + j2_labels[gt[0]]
                if gt[-1] != "W":
                    gt = "{}{}".format(gt, allele)
                plot_phenotypes_scatter(df, axes, col=gt, ref="WW", color="black",
                                        add_diag=True, add_svd_line=True)
                axes.text(0.05, 0.95, label, transform=axes.transAxes, fontsize=6,
                        ha='left', va='top')

        cols1 = ["PLT3", "PLT7"]
        cols2 = ["J2", "EJ2", "Season"]
        genotypes = ["WH", "HW", "WM", "HH", "HM", "MW", "MH", 'MM']
        plt3_labels = {'W': r'$PLT3$', 'H': r'$plt3/+$', 'M': r'$plt3$'}
        plt7_labels = {'W': r'$PLT7$', 'H': r'$plt7/+$', 'M': r'$plt7$'}
        df = pivot_gt_data(gt_data, cols1, cols2, use_mle=use_mle)
        
        print('\tPlotting PLT3-PLT7')
        for gt, axes in zip(genotypes, subplots[:, -1]):
            label = plt3_labels[gt[1]] + ' ' + plt7_labels[gt[0]]
            plot_phenotypes_scatter(df, axes, col=gt, ref="WW", color="black",
                                    add_diag=True, add_svd_line=True)
            axes.text(0.05, 0.95, label, transform=axes.transAxes, fontsize=6,
                        ha='left', va='top')
        subplots[-1, -1].set_xlabel('branching events in\n$PLT3\ PLT7$ background')

        # Re-arrange and save figure
        fig.tight_layout(w_pad=0.05, h_pad=0.15)
        fig.subplots_adjust(bottom=0.065, left=0.075)
        fig.supxlabel('branching events in\n$EJ2\ J2$ background',
                    fontsize=7, x=0.47, y=0.010,
                    ha='center', va='bottom')
        fig.supylabel('branching events in mutant background',
                    fontsize=7, x=0.025, y=0.52,
                    ha='center', va='center')
        fname = "figures/Figure{}".format(figlabel)
        fig.savefig("{}.png".format(fname), dpi=300)
        # fig.savefig("{}.svg".format(fname))
        fig.savefig("{}.pdf".format(fname))
