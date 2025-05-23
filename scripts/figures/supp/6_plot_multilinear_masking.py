#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from scripts.settings import FIG_WIDTH


if __name__ == "__main__":
    print("Loading genotype predictions")
    encoding = {"W": 0, "H": 1, "M": 2}
    pred = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    lims = 0.035, 40

    backgrounds = ["WW", "WH", "HW", "WM", "HH", "MW", "HM", "MH"]
    bc_labels = [
        "$PLT3\ PLT7$",
        "$PLT3\ plt7/+$",
        "$plt3/+\ PLT7$",
        "$PLT3\ plt7$",
        "$plt3/+\ plt7/+$",
        "$plt3\ PLT7$",
        "$plt3/+\ plt7$",
        "$plt3\ plt7/+$",
    ]
    label, allele = "$EJ2^{pro8}$", 8
    # Init figure
    fig, subplots = plt.subplots(
        1,
        8,
        figsize=(FIG_WIDTH * 1.1, 0.33 * FIG_WIDTH),
    )

    print("Plotting J2-EJ2 phenotypes across PLT3-PLT7 backgrounds")
    for bc, axes, bc_label in zip(backgrounds, subplots, bc_labels):
        print("\tBackground {}".format(bc))
        pred_j2 = (
            pred.loc[pred["s1"] == bc, :]
            .drop_duplicates(["PLT3", "PLT7", "J2", "EJ2"])
            .drop("Season", axis=1)
            .set_index("s2")
        )
        pred_j2["x"] = [
            encoding[x[0]] + encoding[y[0]]
            for x, y in pred_j2[["EJ2", "J2"]].values
        ]
        idx = [
            x.endswith("W") or x.endswith(str(allele)) for x in pred_j2["EJ2"]
        ]
        df = pred_j2.loc[idx, :]
        axes.scatter(
            df["x"],
            np.exp(df["hierarchical_pred"]),
            c="black",
            zorder=10,
            s=7,
            lw=0.2,
            edgecolor="white",
        )
        c = {"W": 0, "H": 1, "M": 2}
        for s1, s2 in combinations(df.index, 2):
            d = np.sum(
                [np.abs(c[a1] - c[a2]) for a1, a2 in zip(s1[:2], s2[:2])]
            )
            if d == 1:
                sdf = df.loc[[s1, s2], :]
                axes.plot(
                    sdf["x"],
                    np.exp(sdf["hierarchical_pred"]),
                    lw=0.75,
                    c="grey",
                )
        axes.set(
            yscale="log",
            xticks=[0, 1, 2, 3, 4],
            ylim=lims,
            xlim=(-0.5, 4.5),
            title=bc_label,
        )

        if bc != "WW":
            axes.set(yticklabels=[])

        try:
            x, y = df.loc["WM{}".format(allele), ["x", "hierarchical_pred"]]
            axes.text(
                x + 0.1, np.exp(y), label, va="top", ha="left", fontsize=6
            )
        except KeyError:
            pass
        x, y = df.loc["MW", ["x", "hierarchical_pred"]]
        axes.text(x, np.exp(y), "$j2$", va="bottom", ha="right", fontsize=6)
        try:
            x, y = df.loc["MM{}".format(allele), ["x", "hierarchical_pred"]]
            axes.text(
                x,
                np.exp(y),
                label + " $j2$",
                va="bottom",
                ha="right",
                fontsize=6,
            )
        except KeyError:
            pass

    # Axes labels
    subplots[0].set(
        ylabel="branching events",
    )
    fig.supxlabel(
        "Mutational distance to wild-type",
        x=0.525,
        y=0.1,
        fontsize=7,
    )

    # Re-arrange and save figure
    fig.tight_layout(w_pad=0.2, h_pad=-0.25)

    print("Rendering")
    fname = "figures/FigureS8c"
    fig.savefig("{}.png".format(fname), dpi=600)
    fig.savefig("{}.svg".format(fname))
    print("Done")
