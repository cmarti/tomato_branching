#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    theta1 = pd.read_csv("results/bilinear_model.theta1.csv", index_col=0)
    theta1["plt3"] = [x[0] for x in theta1.index]
    theta1["plt7"] = [x[1] for x in theta1.index]
    theta1 = pd.pivot_table(theta1, index="plt3", columns="plt7", values='param')
    print(theta1)
    
    fig, subplots = plt.subplots(1, 1, figsize=(3.5, 2.5), sharex=True, sharey=True)
    lims = -3.25, 3.25
    ticks = np.arange(-3, 4)
    s = 5

    df = theta1
    axes = subplots
    axes.scatter(df["W"], df["M"], s=s, c="black", label='$plt3$')
    axes.scatter(df["W"], df["H"], s=s, c="grey", label='$plt3/+$')
    axes.set(aspect="equal", xlim=lims, ylim=lims,
             xticks=ticks, yticks=ticks,
             ylabel="$PLT3$ Mutant phenotype",
             xlabel='$PLT3$ background')
    axes.grid(alpha=0.2)
    axes.axline((0, 0), (1, 1), linestyle="--", c='lightgrey', lw=0.75)
    axes.legend(loc=4, fontsize=8)

    
    fig.tight_layout()
    fig.savefig('figures/plt3plt7_synergy_dominance.png', dpi=300)
    
