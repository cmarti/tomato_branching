#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator
from scripts.settings import EJ2_SERIES


def plot(df, axes, col="M", ref="W", color="black", alpha=1, label=None):
    df = df.dropna(subset=[ref, col])

    d = df.loc[[not x.startswith("H") for x in df.index]]
    axes.scatter(
        d[ref],
        d[col],
        c=color,
        alpha=alpha,
        s=4.5,
        lw=0.2,
        label=label,
        edgecolor="white",
    )

    d = df.loc[[x.startswith("H") for x in df.index]]
    axes.scatter(
        d[ref],
        d[col],
        c=color,
        alpha=alpha,
        s=4.5,
        lw=0.6,
        label=label,
        edgecolor="red",
    )
    lims = -4, 4

    axes.axline((1, 1), (2, 2), lw=0.5, c="grey", linestyle="--", alpha=alpha)
    axes.set(
        xlabel="",
        ylabel="",
        xlim=lims,
        ylim=lims,
        aspect="equal",
    )

    # major_locator = LogLocator(base=10.0, numticks=10)
    # minor_locator = LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)

    # axes.xaxis.set_major_locator(major_locator)
    # axes.xaxis.set_minor_locator(minor_locator)
    # axes.yaxis.set_major_locator(major_locator)
    # axes.yaxis.set_minor_locator(minor_locator)

    # Show gridlines
    axes.grid(alpha=0.2, lw=0.3)


if __name__ == "__main__":
    theta2 = pd.read_csv("results/bilinear_model.theta2.csv", index_col=0)
    theta2["j2"] = [x[0] for x in theta2.index]
    theta2["ej2"] = [x[1:] for x in theta2.index]
    theta2 = pd.pivot_table(theta2, index="ej2", columns="j2", values="param")
    print(theta2)
    
    
    fig, subplots = plt.subplots(1, 2, figsize=(4.5, 2.5), sharex=True, sharey=True)
    lims = -3.25, 3.25
    ticks = np.arange(-3, 4)
    s = 5

    df = theta2.loc[[x.startswith("H") for x in theta2.index], :]
    axes = subplots[0]
    axes.scatter(df["H"], df["M"], s=s, c="black", label='$j2$')
    axes.scatter(df["W"], df["H"], s=s, c="grey", label='$j2/+$')
    axes.set(aspect="equal", xlim=lims, ylim=lims,
             xticks=ticks, yticks=ticks,
             ylabel="Mutant phenotype",
             xlabel='$EJ2^{pro}/+$ background')
    axes.grid(alpha=0.2)
    axes.axline((0, 0), (1, 1), linestyle="--", c='grey', lw=0.75)
    axes.legend(loc=2, fontsize=9)

    df = theta2.loc[[x.startswith("M") for x in theta2.index], :]
    axes = subplots[1]
    axes.scatter(df["W"], df["H"], s=s, c="grey", label='$j2/+$')
    axes.scatter(df["H"], df["M"], s=s, c="black", label='$j2$')
    axes.set(aspect="equal", xlim=lims, ylim=lims,
             xticks=ticks, yticks=ticks,
             ylabel="",
             xlabel='$EJ2^{pro}$ background')
    axes.grid(alpha=0.2)
    axes.axline((0, 0), (1, 1), linestyle="--", c='grey', lw=0.75)
    fig.tight_layout(w_pad=0.75)
    fig.savefig('figures/j2ej2_synergy.png', dpi=300)
    
    exit()
    field = "value"
    theta2 = pd.read_csv("results/js_synergy_tests.csv", index_col=0)
    theta2["ej2type"] = [x[1] for x in theta2.index]
    theta2["j2"] = ["{}W".format(x[0]) for x in theta2.index]
    theta2["ej2"] = ["W{}".format(x[1:]) for x in theta2.index]
    theta2["x"] = theta2.loc[theta2["j2"], field].values
    theta2["y"] = theta2.loc[theta2["ej2"], field].values

    fig = plt.figure()
    axes = fig.add_subplot(projection="3d")

    df = theta2.loc[theta2["ej2type"] == "H", :]
    axes.scatter(df["x"], df["y"], df[field], c="red")
    df = theta2.loc[theta2["ej2type"] == "M", :]
    axes.scatter(df["x"], df["y"], df[field], c="blue")
    axes.set(xlabel="j2", ylabel="ej2", zlabel="Joint")

    plt.show()
    exit()

    cols = ["EJ2 variant", "EJ2", "J2", "PLT3", "PLT7", "Season"]
    print("Plotting effects of mutations across backgrounds")
    theta1 = pd.read_csv("results/bilinear_model.theta1.csv", index_col=0)
    theta2 = pd.read_csv("results/bilinear_model.theta2.csv", index_col=0)

    theta1["plt3"] = [x[0] for x in theta1.index]
    theta1["plt7"] = [x[1] for x in theta1.index]
    theta1 = pd.pivot_table(
        theta1, index="plt3", columns="plt7", values="param"
    )

    theta2["j2"] = [x[0] for x in theta2.index]
    theta2["ej2"] = [x[1:] for x in theta2.index]
    theta2 = pd.pivot_table(theta2, index="ej2", columns="j2", values="param")
    print(theta2)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")

    df = theta2.loc[[x.startswith("W") for x in theta2.index], :]
    axes.scatter(df["W"], df["H"], df["M"], color="blue")

    df = theta2.loc[[x.startswith("H") for x in theta2.index], :]
    axes.scatter(df["W"], df["H"], df["M"], color="red")

    df = theta2.loc[[x.startswith("M") for x in theta2.index], :]
    axes.scatter(df["W"], df["H"], df["M"], color="black")

    axes.set(xlabel="J2 WT", ylabel="j2/+", zlabel="j2")

    plt.show()

    exit()

    # Init figure
    fig, subplots = plt.subplots(
        1, 2, figsize=(2.3 * 2, 2.5 * 1), sharex=True, sharey=True
    )

    axes = subplots[0]
    plot(theta2, axes, col="H", ref="W", color="grey")
    plot(theta2, axes, col="M", ref="H", color="black")
    a, b = -2.65, 1.170692
    point1 = (a + np.exp(0), a + np.exp(b))
    point2 = (a + np.exp(1), a + np.exp(1 + b))
    axes.axline(point1, point2)
    axes.set_title("$j2$")
    axes.set(ylabel="Mutant $\phi$")

    axes = subplots[1]
    plot(theta1, axes, col="H", ref="W", color="grey")
    plot(theta1, axes, col="M", ref="H", color="black")
    a, b = 0, 0.9804074
    point1 = (a + np.exp(0), a + np.exp(b))
    point2 = (a + np.exp(1), a + np.exp(1 + b))
    axes.axline(point1, point2)
    axes.set_ylabel("")
    axes.set_title("$plt3$")

    fig.supxlabel("Background $\phi$", ha="center", x=0.6, y=0.065)

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0.05, h_pad=1.2)
    fname = "figures/synergy_effects".format()
    fig.savefig("{}.png".format(fname), dpi=300)
    fig.savefig("{}.pdf".format(fname))
