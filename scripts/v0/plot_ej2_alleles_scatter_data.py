#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator

from scripts.settings import EJ2_SERIES, EJ2_SERIES_LABELS


def plot(df, axes, col="M", ref="W", color="black", alpha=1, label=None):
    df = np.exp(df.dropna(subset=[ref, col]))
    x, y = df[ref], df[col]

    dx = np.abs(df[["{}_lower".format(ref), "{}_upper".format(ref)]].T - x)
    dy = np.abs(df[["{}_lower".format(col), "{}_upper".format(col)]].T - y)

    axes.errorbar(
        x, y, xerr=dx, yerr=dy, lw=0.3, alpha=alpha, ecolor=color, fmt="none"
    )
    axes.scatter(
        x,
        y,
        c=color,
        alpha=alpha,
        s=4.5,
        lw=0.2,
        label=label,
        edgecolor="white",
    )
    lims = 0.0025, 150

    axes.axline((1, 1), (2, 2), lw=0.5, c="grey", linestyle="--", alpha=alpha)
    
    dy = (np.log(y) - np.log(x)).mean()
    axes.axline((1, np.exp(dy)), (2, 2 * np.exp(dy)), lw=1, c=color, linestyle="--", alpha=alpha)

    axes.set(
        xlabel="",
        ylabel="",
        xlim=lims,
        ylim=lims,
        xscale="log",
        yscale="log",
        aspect="equal",
    )

    major_locator = LogLocator(base=10.0, numticks=10)
    minor_locator = LogLocator(
        base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
    )

    axes.xaxis.set_major_locator(major_locator)
    axes.xaxis.set_minor_locator(minor_locator)
    axes.yaxis.set_major_locator(major_locator)
    axes.yaxis.set_minor_locator(minor_locator)

    # Show gridlines
    axes.grid(alpha=0.2, lw=0.3)


if __name__ == "__main__":
    cols = ["EJ2 variant", "EJ2", "J2", "PLT3", "PLT7", "Season"]
    print("Plotting effects of mutations across backgrounds")
    gt_data = pd.read_csv("results/genotype_means_estimates.csv", index_col=0)
    gt_data = gt_data.loc[gt_data["stderr"] < 10, :]
    gt_data = gt_data.loc[gt_data["EJ2"] != "W", :]
    gt_data["ej2_allele"] = [x[-1] for x in gt_data["EJ2"]]
    gt_data["ej2"] = [x[0] for x in gt_data["EJ2"]]
    gt_data["gt"] = [
        "".join(x)
        for x in gt_data[["PLT3", "PLT7", "J2", "ej2", "Season"]].values
    ]

    means = pd.pivot_table(
        gt_data, index="gt", columns="ej2_allele", values="estimate"
    )
    lower = pd.pivot_table(
        gt_data, index="gt", columns="ej2_allele", values="lower"
    )
    lower.columns = ["{}_lower".format(x) for x in lower.columns]
    upper = pd.pivot_table(
        gt_data, index="gt", columns="ej2_allele", values="upper"
    )
    upper.columns = ["{}_upper".format(x) for x in upper.columns]
    data = pd.concat([means, lower, upper], axis=1)
    df1 = data.loc[[x[3] == 'H' for x in data.index], :]
    df2 = data.loc[[x[3] == 'M' for x in data.index], :]

    # theta = pd.read_csv("results/bilinear_model.theta2.csv", index_col=0)[
    #     "param"
    # ].to_dict()
    # theta["EJ2(1)"] = 0.0

    # Init figure
    fig, subplots = plt.subplots(5, 5, figsize=(10, 10))
    print(EJ2_SERIES)
    for i in range(5):
        for j in range(i + 1):
            axes = subplots[i, j]
            a1, a2 = EJ2_SERIES[i + 1], EJ2_SERIES[j]
            plot(
                df1,
                axes,
                col=a1,
                ref=a2,
                color="grey",
                alpha=1,
                label=None,
            )
            plot(
                df2,
                axes,
                col=a1,
                ref=a2,
                color="black",
                alpha=1,
                label=None,
            )
            # allele1, allele2 = EJ2_SERIES_LABELS[i + 1], EJ2_SERIES_LABELS[j]
            # dy = theta[allele1] - theta[allele2]
            # print(allele1, allele2, dy)
            # axes.axline(
            #     np.exp((1, 1.0 + dy)),
            #     np.exp((2, 2.0 + dy)),
            #     lw=1,
            #     c="black",
            #     linestyle="--",
            #     alpha=1,
            # )

        for j in range(i + 1, 5):
            axes = subplots[i, j]
            sns.despine(ax=axes, left=True, bottom=True)
            axes.set_xticks([])
            axes.set_yticks([])
            axes.minorticks_off()

    for i, allele in enumerate(EJ2_SERIES[:-1]):
        subplots[-1, i].set_xlabel("EJ2({}) branching events".format(allele))
    for i, allele in enumerate(EJ2_SERIES[1:]):
        subplots[i, 0].set_ylabel("EJ2({}) branching events".format(allele))

    for i in range(4):
        for axes in subplots[i, :]:
            axes.set_xticklabels([])

    for i in range(1, 5):
        for axes in subplots[:, i]:
            axes.set_yticklabels([])

    fig.suptitle("$EJ2^{pro}$ alleles", ha="center")

    # Re-arrange and save figure
    fig.tight_layout()
    fname = "figures/ej2_alleles_scatter.data"
    fig.savefig("{}.png".format(fname), dpi=300)
