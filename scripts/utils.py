#!/usr/bin/env python
import pandas as pd
import numpy as np

from matplotlib.ticker import LogLocator
from itertools import combinations, product
from scripts.settings import (
    EJ2_SERIES,
    EJ2_SERIES_NAMES,
    G_SITES_SIMPLE,
    G_SITES,
    LIMS,
)


def get_basis_df(basis, rm_zeros=True):
    df = pd.DataFrame(basis)
    if rm_zeros:
        sel_cols = np.any(df != 0, axis=0)
        df = df.loc[:, sel_cols]
    return df


def get_constant_basis(plant_data, rm_zeros=False):
    basis = {"intercept": np.ones(plant_data.shape[0])}
    return get_basis_df(basis, rm_zeros=rm_zeros)


def get_additive_basis(plant_data, rm_zeros=False):
    code = {"W": -0.5, "H": 0.0, "M": 0.5}
    additive = {s: [code[x] for x in plant_data[s]] for s in G_SITES_SIMPLE}
    for ej2_variant in EJ2_SERIES:
        code = {"H{}".format(ej2_variant): 0.0, "M{}".format(ej2_variant): 0.5}
        additive["EJ2({})".format(ej2_variant)] = [
            code.get(x, -0.5) for x in plant_data["EJ2"]
        ]
    return get_basis_df(additive, rm_zeros=rm_zeros)


def get_dominance_basis(plant_data, rm_zeros=False):
    code = {"W": 0.0, "H": 1.0, "M": 0.0}
    dominant = {
        "{}.het".format(s): [code[x] for x in plant_data[s]]
        for s in G_SITES_SIMPLE
    }
    for ej2_variant in EJ2_SERIES:
        dominant["EJ2({}).het".format(ej2_variant)] = (
            (plant_data["EJ2"] == "H{}".format(ej2_variant))
            .astype(float)
            .values
        )
    return get_basis_df(dominant, rm_zeros=rm_zeros)


def get_gxg_basis(additive_basis, rm_zeros=False):
    gxg = {
        "{}_{}".format(s1, s2): additive_basis[s1] * additive_basis[s2]
        for s1, s2 in combinations(G_SITES_SIMPLE, 2)
    }

    for s1 in G_SITES_SIMPLE:
        for s2 in EJ2_SERIES_NAMES:
            gxg["{}_{}".format(s1, s2)] = (
                additive_basis[s1] * additive_basis[s2]
            )
    return get_basis_df(gxg, rm_zeros=rm_zeros)


def get_dxd_basis(dominance_basis, rm_zeros=False):
    dxd = {
        "{}.het_{}.het".format(s1, s2): dominance_basis["{}.het".format(s1)]
        * dominance_basis["{}.het".format(s2)]
        for s1, s2 in combinations(G_SITES_SIMPLE, 2)
    }

    for s1 in G_SITES_SIMPLE:
        for s2 in EJ2_SERIES_NAMES:
            dxd["{}.het_{}.het".format(s1, s2)] = (
                dominance_basis["{}.het".format(s1)]
                * dominance_basis["{}.het".format(s2)]
            )
    return get_basis_df(dxd, rm_zeros=rm_zeros)


def get_gxd_basis(additive_basis, dominance_basis, rm_zeros=False):
    gxd = {
        "{}_{}.het".format(s1, s2): additive_basis[s1]
        * dominance_basis["{}.het".format(s2)]
        for s1, s2 in product(G_SITES_SIMPLE, G_SITES)
        if s1 != s2
    }
    gxd.update(
        {
            "{}_{}.het".format(s1, s2): additive_basis[s1]
            * dominance_basis["{}.het".format(s2)]
            for s1, s2 in product(EJ2_SERIES_NAMES, G_SITES_SIMPLE)
            if s1 != s2
        }
    )
    return get_basis_df(gxd, rm_zeros=rm_zeros)


def get_saturated_basis(plant_data):
    cols = ["PLT3", "PLT7", "J2", "EJ2", "Season"]
    gts = [tuple(x) for x in plant_data[cols].values]
    unique_gts = set(gts)
    basis = pd.DataFrame(
        {
            "_".join(gt): [tuple(x) == gt for x in plant_data[cols].values]
            for gt in unique_gts
        }
    ).astype(float)
    return basis


def get_add_basis(plant_data):
    c = get_constant_basis(plant_data)
    g = get_additive_basis(plant_data)
    d = get_dominance_basis(plant_data)
    additive_basis = get_basis_df(pd.concat([c, g, d], axis=1), rm_zeros=True)
    return additive_basis


def get_pairwise_basis(plant_data):
    c = get_constant_basis(plant_data)
    g = get_additive_basis(plant_data)
    d = get_dominance_basis(plant_data)
    gxg = get_gxg_basis(g)
    gxd = get_gxd_basis(g, d)
    dxd = get_dxd_basis(d)
    pairwise_basis = get_basis_df(
        pd.concat([c, g, d, gxg, gxd, dxd], axis=1), rm_zeros=True
    )
    pairwise_basis.drop("PLT3_PLT7", axis=1)
    return pairwise_basis


def set_log_ticks(axes):
    major_locator = LogLocator(base=10.0, numticks=10)
    minor_locator = LogLocator(
        base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
    )

    axes.xaxis.set_major_locator(major_locator)
    axes.xaxis.set_minor_locator(minor_locator)
    axes.yaxis.set_major_locator(major_locator)
    axes.yaxis.set_minor_locator(minor_locator)


def set_aspect(axes, xlabel=None, ylabel=None, add_diag=True):
    axes.set(
        aspect="equal",
        xscale="log",
        yscale="log",
        xlim=LIMS,
        ylim=LIMS,
        ylabel=ylabel,
        xlabel=xlabel,
    )
    set_log_ticks(axes)
    if add_diag:
        axes.axline(
            (1, 1), (2, 2), lw=0.3, c="grey", linestyle="--", alpha=0.5
        )


def plot_phenotypes_scatter(
    df,
    axes,
    col,
    ref="WW",
    color="black",
    alpha=1,
    label=None,
    add_diag=False,
    add_svd_line=True,
):
    if ref in df.columns and col in df.columns:
        df = np.exp(df.dropna(subset=[ref, col]))
        x, y = df[ref], df[col]

        try:
            dx = np.abs(
                df[["{}_lower".format(ref), "{}_upper".format(ref)]].T - x
            )
        except KeyError:
            dx = None

        try:
            dy = np.abs(
                df[["{}_lower".format(col), "{}_upper".format(col)]].T - y
            )
        except KeyError:
            dy = None

        axes.errorbar(
            x,
            y,
            xerr=dx,
            yerr=dy,
            lw=0.1,
            alpha=alpha,
            ecolor=color,
            fmt="none",
        )
        axes.scatter(
            x,
            y,
            c=color,
            alpha=alpha,
            s=2,
            lw=0.1,
            label=label,
            edgecolor="white",
        )
        if add_svd_line:
            A = np.log(df[[ref, col]].values)
            p0 = np.mean(A, axis=0)
            _, _, V = np.linalg.svd(A - p0)
            p1 = p0 + V[0, :]
            axes.axline(
                np.exp(p0), np.exp(p1), lw=0.5, c="black", linestyle="--"
            )

    set_aspect(axes, add_diag=add_diag)


def add_model_line(axes, theta, gt):
    point1 = (np.exp(theta.loc["WW", "v1"]), np.exp(theta.loc[gt, "v1"]))
    point2 = (np.exp(theta.loc["WW", "v2"]), np.exp(theta.loc[gt, "v2"]))
    axes.axline(
        point1, point2, linestyle="--", lw=0.3, c="black", label="Model"
    )
