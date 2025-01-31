#!/usr/bin/env python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.colors as mc
import colorsys

from matplotlib.ticker import LogLocator
from itertools import combinations, product
from scripts.settings import (
    EJ2_SERIES,
    ALLELES,
    G_SITES_SIMPLE,
    EJ2_SERIES_LABELS,
    SEASONS,
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


def get_env_basis(plant_data, rm_zeros=False):
    seasons = sorted(plant_data["Season"].unique())
    ref = seasons[0]
    if ref == "Summer 22":
        ref = seasons[1]
    seasons = [x for x in seasons if x != ref]

    env = {e: (plant_data["Season"] == e).astype(float).values for e in seasons}
    return get_basis_df(env, rm_zeros=rm_zeros)


def get_gxg_basis(additive_basis, rm_zeros=False):
    gxg = {
        "{}_{}".format(s1, s2): additive_basis[s1] * additive_basis[s2]
        for s1, s2 in combinations(G_SITES_SIMPLE, 2)
    }

    for s1 in G_SITES_SIMPLE:
        for s2 in EJ2_SERIES_LABELS:
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
        for s2 in EJ2_SERIES_LABELS:
            dxd["{}.het_{}.het".format(s1, s2)] = (
                dominance_basis["{}.het".format(s1)]
                * dominance_basis["{}.het".format(s2)]
            )
    return get_basis_df(dxd, rm_zeros=rm_zeros)


def get_3way_basis(
    additive_basis, dominance_basis, rm_zeros=False, keep_plt7=True
):
    sites_set = G_SITES_SIMPLE
    if not keep_plt7:
        sites_set = [s for s in sites_set if s != "PLT7"]

    g_basis = {}
    d_basis = {}
    for sites in combinations(sites_set, 3):
        for components in product(["g", "d"], repeat=3):
            bs = [
                additive_basis[s]
                if x == "g"
                else dominance_basis["{}.het".format(s)]
                for x, s in zip(components, sites)
            ]
            label = "_".join(
                "{}{}".format(x, s) for x, s in zip(components, sites)
            )
            if "d" in components:
                d_basis[label] = np.prod(bs, 0)
            else:
                g_basis[label] = np.prod(bs, 0)

    for s1, s2 in combinations(sites_set, 2):
        for s3 in EJ2_SERIES_LABELS:
            sites = [s1, s2, s3]
            for components in product(["g", "d"], repeat=3):
                bs = [
                    additive_basis[s]
                    if x == "g"
                    else dominance_basis["{}.het".format(s)]
                    for x, s in zip(components, sites)
                ]
                label = "_".join(
                    "{}{}".format(x, s) for x, s in zip(components, sites)
                )
                if "d" in components:
                    d_basis[label] = np.prod(bs, 0)
                else:
                    g_basis[label] = np.prod(bs, 0)
    return (
        get_basis_df(g_basis, rm_zeros=rm_zeros),
        get_basis_df(d_basis, rm_zeros=rm_zeros),
    )


def get_gxd_basis(additive_basis, dominance_basis, rm_zeros=False):
    gxd = {
        "{}_{}.het".format(s1, s2): additive_basis[s1]
        * dominance_basis["{}.het".format(s2)]
        for s1, s2 in get_product(G_SITES_SIMPLE, G_SITES)
        if s1 != s2
    }
    gxd.update(
        {
            "{}_{}.het".format(s1, s2): additive_basis[s1]
            * dominance_basis["{}.het".format(s2)]
            for s1, s2 in get_product(EJ2_SERIES_LABELS, G_SITES_SIMPLE)
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
    # pairwise_basis.drop('PLT3_PLT7', axis=1)
    return pairwise_basis


def get_product(set1, set2):
    for s1 in set1:
        for s2 in set2:
            yield (s1, s2)


def get_x_basis(basis1, basis2=None, rm_zeros=True):
    gxe = {}
    if basis2 is None:
        pairs = combinations(basis1.columns, 2)
    else:
        pairs = get_product(basis1.columns, basis2.columns)

    for s1, s2 in pairs:
        gxe["{}_{}".format(s1, s2)] = basis1[s1] * basis2[s2]
    return get_basis_df(gxe, rm_zeros=rm_zeros)


def get_full_basis(
    plant_data, rm_zeros=True, third_order=True, dominance=True, keep_plt7=True
):
    c = get_constant_basis(plant_data, rm_zeros=rm_zeros)
    g = get_additive_basis(plant_data, rm_zeros=rm_zeros)
    d = get_dominance_basis(plant_data, rm_zeros=rm_zeros)
    e = get_env_basis(plant_data, rm_zeros=rm_zeros)

    gxg = get_gxg_basis(g, rm_zeros=rm_zeros)
    gxd = get_gxd_basis(g, d, rm_zeros=rm_zeros)
    dxd = get_dxd_basis(d, rm_zeros=rm_zeros)
    gxe = get_x_basis(g, e, rm_zeros=rm_zeros)
    dxe = get_x_basis(d, e, rm_zeros=rm_zeros)

    basis = [c, g, e, gxg, gxe]
    if dominance:
        basis.extend([d, gxd, dxd, dxe])

    if third_order:
        g3, d3 = get_3way_basis(g, d, rm_zeros=rm_zeros, keep_plt7=keep_plt7)
        basis.extend([g3])

        if dominance:
            basis.extend([d3])

    X = get_basis_df(pd.concat(basis, axis=1), rm_zeros=rm_zeros)
    return X


def get_phi_to_ypred(alpha):
    likelihood = sm.families.NegativeBinomial(alpha=alpha)
    ys = np.geomspace(0.01, 80, 100)
    phi = likelihood.link(ys)
    df = pd.DataFrame({"phi": phi, "y": ys})
    for q in [0.01, 0.25, 0.05, 0.1, 0.15, 0.2, 0.25]:
        df["y_{}".format(q)] = likelihood.get_distribution(ys).ppf(q)
        df["y_{}".format(1 - q)] = likelihood.get_distribution(ys).ppf(1 - q)
    return df


def get_double_mutant_plant_data():
    records = []
    base_record = dict(zip(G_SITES_SIMPLE + ["EJ2"], ["W"] * 4))

    for season in SEASONS:
        for i, site1 in enumerate(G_SITES_SIMPLE):
            for a1 in ALLELES:
                # Simple combinations
                for site2 in G_SITES_SIMPLE[i + 1 :]:
                    for a2 in ALLELES:
                        record = {
                            "v1": "{}-{}".format(site1, a1),
                            "v2": "{}-{}".format(site2, a2),
                            "Season": season,
                        }
                        record.update(base_record)
                        record[site1] = a1
                        record[site2] = a2
                        records.append(record)

                # Combinations with allelic series
                for variant, label in zip(EJ2_SERIES, EJ2_SERIES_LABELS):
                    for a2 in ALLELES:
                        record = {
                            "v1": "{}-{}".format(site1, a1),
                            "v2": "{}-{}".format(label, a2),
                            "Season": season,
                        }
                        record.update(base_record)
                        record[site1] = a1
                        record["EJ2"] = (
                            "W" if a2 == "W" else "{}{}".format(a2, variant)
                        )
                        records.append(record)

    plant_data = pd.DataFrame(records)
    return plant_data


def get_params_df(results):
    pvals = results.pvalues
    coeff = results.params
    ci = results.conf_int()
    conf_lower = ci[0]
    conf_higher = ci[1]
    results_df = pd.DataFrame(
        {
            "coeff": coeff,
            "pval": pvals,
            "ci95_lower": conf_lower,
            "ci95_upper": conf_higher,
        }
    )
    return results_df


def get_env_plant_data_pred():
    records = []
    base_record = dict(zip(G_SITES_SIMPLE + ["EJ2"], ["W"] * 4))

    for season in SEASONS:
        for alleles in product(ALLELES, repeat=3):
            for variant, label in zip(EJ2_SERIES, EJ2_SERIES_LABELS):
                for a2 in ALLELES:
                    record = dict(zip(G_SITES_SIMPLE, alleles))
                    record["EJ2"] = (
                        "W" if a2 == "W" else "{}{}".format(a2, variant)
                    )
                    record["Season"] = season
                    records.append(record)
    plant_data = (
        pd.DataFrame(records)
        .drop_duplicates()
        .reset_index()
        .drop("index", axis=1)
    )
    return plant_data


def get_masking_plant_data():
    plant_datas = []
    backgrounds = {
        "wt": {},
        "j2": {"J2": "M"},
        "plt3": {"PLT3": "M"},
        "plt7": {"PLT7": "M"},
        "plt3/j2": {"PLT3": "M", "J2": "M"},
        "plt7/j2": {"PLT7": "M", "J2": "M"},
        "plt3h/plt7": {"PLT7": "M", "PLT3": "H"},
        "plt3/plt7h": {"PLT7": "H", "PLT3": "M"},
    }

    for a in ["W", "M"]:
        records = []
        base_record = dict(zip(G_SITES_SIMPLE + ["EJ2"], ["W"] * 4))

        for season in SEASONS:
            for background_label, background in backgrounds.items():
                if "J2" not in background:
                    record = base_record.copy()
                    record.update(background)
                    record["J2"] = a
                    record["Season"] = season
                    record["background"] = background_label
                    record["site"] = "J2"
                    records.append(record)

                for variant, label in zip(EJ2_SERIES, EJ2_SERIES_LABELS):
                    record = base_record.copy()
                    record.update(background)
                    record["EJ2"] = (
                        "W" if a == "W" else "{}{}".format(a, variant)
                    )
                    record["site"] = label
                    record["Season"] = season
                    record["background"] = background_label
                    records.append(record)

        plant_data = (
            pd.DataFrame(records)
            .drop_duplicates()
            .reset_index()
            .drop("index", axis=1)
        )
        plant_datas.append(plant_data)
    return plant_datas


def get_masking_plts_plant_data():
    plant_datas = []
    backgrounds = {"wt": {}, "j2": {"J2": "M"}}
    for variant, label in zip(EJ2_SERIES, EJ2_SERIES_LABELS):
        backgrounds[label.lower()] = {"EJ2": "M{}".format(variant)}
        backgrounds["{}/j2".format(label.lower())] = {
            "EJ2": "M{}".format(variant),
            "J2": "M",
        }

    for a in ["W", "M"]:
        records = []
        base_record = dict(zip(G_SITES_SIMPLE + ["EJ2"], ["W"] * 4))

        for season in SEASONS:
            for background_label, background in backgrounds.items():
                for site in ["PLT3", "PLT7"]:
                    record = base_record.copy()
                    record.update(background)
                    record[site] = a
                    record["Season"] = season
                    record["background"] = background_label
                    record["site"] = site
                    records.append(record)

        plant_data = (
            pd.DataFrame(records)
            .drop_duplicates()
            .reset_index()
            .drop("index", axis=1)
        )
        plant_datas.append(plant_data)
    return plant_datas


def get_plts_masking_plant_data(site1, site2):
    records = []
    contrasts = []
    base_record = dict(zip(G_SITES_SIMPLE + ["EJ2"], ["W"] * 4))
    base_contrast = {
        x: 0
        for x in ["00", "01", "10", "11", "in_WT", "in_J2", "J2"]
        + EJ2_SERIES_LABELS
    }

    base_int = {
        ("W", "W", "W"): 1,
        ("W", "M", "W"): -1,
        ("H", "W", "W"): -1,
        ("H", "M", "W"): 1,
        ("W", "W", "M"): -1,
        ("H", "W", "M"): 1,
        ("W", "M", "M"): 1,
        ("H", "M", "M"): -1,
    }

    for season in SEASONS:
        for a1 in ["W", "H"]:
            for a2 in ["W", "M"]:
                # wt
                a3 = "W"
                record = base_record.copy()
                record.update({site1: a1, site2: a2, "Season": season})
                records.append(record)

                contrast = base_contrast.copy()
                contrast.update(
                    {
                        x: -base_int[(a1, a2, a3)]
                        for x in ["J2"] + EJ2_SERIES_LABELS
                    }
                )
                contrast["in_WT"] = base_int[(a1, a2, a3)]
                contrast["00"] = int(a1 == "W" and a2 == "W")
                contrast["01"] = int(a1 == "W" and a2 == "M")
                contrast["10"] = int(a1 == "H" and a2 == "W")
                contrast["11"] = int(a1 == "H" and a2 == "M")
                contrasts.append(contrast)

                # mutants
                a3 = "M"
                record = base_record.copy()
                record.update(
                    {site1: a1, site2: a2, "J2": a3, "Season": season}
                )
                records.append(record)

                contrast = base_contrast.copy()
                contrast["J2"] = -base_int[(a1, a2, a3)]
                contrast["in_J2"] = -base_int[(a1, a2, a3)]
                contrasts.append(contrast)

                for site3, label in zip(EJ2_SERIES, EJ2_SERIES_LABELS):
                    record = base_record.copy()
                    record.update(
                        {
                            site1: a1,
                            site2: a2,
                            "EJ2": "M{}".format(site3),
                            "Season": season,
                        }
                    )
                    records.append(record)

                    contrast = base_contrast.copy()
                    contrast[label] = -base_int[(a1, a2, a3)]
                    contrasts.append(contrast)

    plant_data = pd.DataFrame(
        records
    )  # .drop_duplicates().reset_index().drop('index', axis=1)
    contrasts = 0.25 * pd.DataFrame(contrasts)
    contrasts["EJ2_mean"] = contrasts[EJ2_SERIES_LABELS].mean(1)
    contrasts.columns = [
        "h{}_{}_{}".format(site1, site2, x) for x in contrasts.columns
    ]
    return (plant_data, contrasts)


def gt_to_coding(gt, season):
    record = {
        "PLT3": gt[0],
        "PLT7": gt[1],
        "J2": gt[2],
        "EJ2": gt[3:],
        "Season": season,
    }
    return record


def define_gts_contrast(contrast_name, gts, values, get_basis):
    B = []
    X = []
    for season in SEASONS:
        for gt, value in zip(gts, values):
            X.append(gt_to_coding(gt, season))
            B.append(value / 4.0)
    X = pd.DataFrame(X)
    B = np.array([B])
    basis = get_basis(X)
    C = B @ basis
    C.index = [contrast_name]
    return C


def define_js_masking_contrasts(get_basis):
    plt_backgrounds = ["WW", "WH", "WM", "HW", "HH", "HM", "MW", "MH"]

    contrasts = []
    for plt_background in plt_backgrounds:
        for gt in ["H", "M"]:
            # J2
            contrast_name = "J2{}_in_{}WW".format(gt, plt_background)
            gts = [
                "{}WW".format(plt_background),
                "{}{}W".format(plt_background, gt),
            ]
            values = [-1, 1]
            contrasts.append(
                define_gts_contrast(contrast_name, gts, values, get_basis)
            )

            # EJ2 variants
            for j2 in ["W", "M"]:
                gts_av = []
                values_av = []
                background = "{}{}".format(plt_background, j2)

                for variant in EJ2_SERIES:
                    contrast_name = "EJ2({}){}_in_{}W".format(
                        variant, gt, background
                    )
                    gts = [
                        "{}W".format(background),
                        "{}{}{}".format(background, gt, variant),
                    ]
                    values = [-1, 1]
                    gts_av.extend(gts)
                    values_av.extend([-1 / 6.0, 1 / 6.0])
                    contrasts.append(
                        define_gts_contrast(
                            contrast_name, gts, values, get_basis
                        )
                    )

                contrast_name = "EJ2{}_mean_in_{}W".format(gt, background)
                contrasts.append(
                    define_gts_contrast(
                        contrast_name, gts_av, values_av, get_basis
                    )
                )

    contrasts = pd.concat(contrasts, axis=0)
    return contrasts


def define_plts_masking_contrasts(get_basis):
    js_backgrounds = []
    for j2 in "WHM":
        js_backgrounds.append("{}W".format(j2))
        for variant in EJ2_SERIES:
            js_backgrounds.append("{}H{}".format(j2, variant))
            js_backgrounds.append("{}M{}".format(j2, variant))

    contrasts = []
    for js_background in js_backgrounds:
        for gt in ["H", "M"]:
            # PLT3
            contrast_name = "PLT3{}_in_WW{}".format(gt, js_background)
            gts = [
                "WW{}".format(js_background),
                "{}W{}".format(gt, js_background),
            ]
            values = [-1, 1]
            contrasts.append(
                define_gts_contrast(contrast_name, gts, values, get_basis)
            )

            # PLT7
            contrast_name = "PLT7{}_in_WW{}".format(gt, js_background)
            gts = [
                "WW{}".format(js_background),
                "W{}{}".format(gt, js_background),
            ]
            values = [-1, 1]
            contrasts.append(
                define_gts_contrast(contrast_name, gts, values, get_basis)
            )

    contrasts = pd.concat(contrasts, axis=0)
    return contrasts


def define_js_synergy_contrasts(get_basis, plt_background="WW"):
    contrasts = []
    for gt in ["H", "M"]:
        # EJ2 wt
        contrast_name = "J2{}_in_{}WW".format(gt, plt_background)
        gts = [
            "{}WW".format(plt_background),
            "{}{}W".format(plt_background, gt),
        ]
        values = [-1, 1]
        contrasts.append(
            define_gts_contrast(contrast_name, gts, values, get_basis)
        )

        # EJ2 variants
        for ej2 in ["H", "M"]:
            gts_av = []
            values_av = []

            for variant in EJ2_SERIES:
                contrast_name = "J2{}_in_{}W{}{}".format(
                    gt, plt_background, ej2, variant
                )
                gts = [
                    "{}W{}{}".format(plt_background, ej2, variant),
                    "{}{}{}{}".format(plt_background, gt, ej2, variant),
                ]
                values = [-1, 1]
                contrasts.append(
                    define_gts_contrast(contrast_name, gts, values, get_basis)
                )
                gts_av.extend(gts)
                values_av.extend([-1 / 6.0, 1 / 6.0])

            contrast_name = "J2{}_in_WWW{}_mean".format(gt, ej2)
            contrasts.append(
                define_gts_contrast(contrast_name, gts_av, values_av, get_basis)
            )

    contrasts = pd.concat(contrasts, axis=0)
    return contrasts


def define_plts_synergy_contrasts(get_basis):
    for j2 in "W":
        js_background = "{}W".format(j2)
        contrasts = []
        for plt3 in "HM":
            for plt7 in "HM":
                if plt3 == "M" and plt7 == "M":
                    continue
                contrast_name = "plt3{}_plt7{}_in_j2{}".format(plt3, plt7, j2)
                gts = [
                    "WW{}".format(js_background),
                    "{}W{}".format(plt3, js_background),
                    "W{}{}".format(plt7, js_background),
                    "{}{}{}".format(plt3, plt7, js_background),
                ]
                values = [1, -1, -1, 1]
                contrasts.append(
                    define_gts_contrast(contrast_name, gts, values, get_basis)
                )

    contrasts = pd.concat(contrasts, axis=0)
    return contrasts


def define_j2_ej2_synergy_contrasts(get_basis, plt_background="WW"):
    contrasts = []
    for j2 in "HM":
        for ej2 in "HM":
            gts_av = []
            values_av = []

            for variant in EJ2_SERIES:
                contrast_name = "j2{}_ej2({}){}_in_plt{}".format(
                    j2, variant, ej2, plt_background
                )
                gts = [
                    "{}WW".format(plt_background),
                    "{}W{}{}".format(plt_background, ej2, variant),
                    "{}{}W".format(plt_background, j2),
                    "{}{}{}{}".format(plt_background, j2, ej2, variant),
                ]
                values = [1, -1, -1, 1]
                contrasts.append(
                    define_gts_contrast(contrast_name, gts, values, get_basis)
                )

                gts_av.extend(gts)
                values_av.extend([1 / 6.0, -1 / 6.0, -1 / 6.0, 1 / 6.0])

            contrast_name = "j2{}_ej2{}_mean_in_plt{}".format(
                j2, ej2, plt_background
            )
            contrasts.append(
                define_gts_contrast(contrast_name, gts_av, values_av, get_basis)
            )

    contrasts = pd.concat(contrasts, axis=0)
    return contrasts


def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    out = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    return out


def plot_mut_scatter(
    axes, gt_data, target, background, show_err=False, alpha=1, model=None
):
    if model is None:
        yvalue, yerr = "log_mean", "log_std_err"
    else:
        yvalue, yerr = "coef", "std err"

    lims = (-12, 6)
    cols = ["EJ2 variant", "PLT3", "PLT7", "J2", "EJ2", "Season"]

    data = gt_data.copy()
    scols = [c for c in cols[1:] if c != target]
    data["label"] = ["{}{}{}-{}".format(*x) for x in gt_data[scols].values]
    for site, allele in background.items():
        data = data.loc[data[site] == allele, :]

    df = pd.pivot_table(data, columns=target, values=yvalue, index="label")
    df["Season"] = [x.split("-")[-1] for x in df.index.values]
    if show_err:
        df = df.join(
            pd.pivot_table(data, columns=target, values=yerr, index="label"),
            rsuffix="_err",
        )

    colors = ["brown", "grey", "purple", "blue"]
    colors = ["grey"] * 4
    # np.random.seed(2)
    # phi = np.random.normal(0, 0.2, size=(4, 3))
    # phi = phi - np.expand_dims(phi.mean(1), 1) - 0.6
    # colors = np.exp(phi) / (1 + np.exp(phi))

    for season, color in zip(SEASONS, colors):
        for col, label, lightness in zip("HM", ("Het", "Homo"), (1.6, 0.5)):
            c = adjust_lightness(color, lightness)
            columns = ["W", col]  # + err_cols
            if show_err:
                err_cols = ["W_err", "{}_err".format(col)]
                columns += err_cols

            d = df.loc[df["Season"] == season, columns].dropna(subset=columns)
            x, y = d["W"], d[col]

            # for variant in EJ2_SERIES:
            #     mapping = {'W': 0, 'H': 1, 'M': 2}
            #     idx = [x.split('-')[0][-1] == variant for x in d.index]
            #     lines_df = d.loc[idx, :]
            #     axes.plot(lines_df['W'], lines_df[col], color=c, lw=0.75)

            if show_err:
                errs = df[err_cols].fillna(np.nanmax(df[err_cols].values))
                dx, dy = errs["W_err"], errs[err_cols[-1]]
                axes.errorbar(
                    x,
                    y,
                    xerr=dx,
                    yerr=dy,
                    lw=0.5,
                    alpha=alpha,
                    ecolor=c,
                    fmt="o",
                    label=label,
                )
            else:
                axes.scatter(
                    x, y, color=c, alpha=alpha, s=15, lw=0, label=label
                )

    axes.plot(lims, lims, lw=0.5, c="grey", linestyle="--", alpha=alpha)
    title = (
        "/".join(["{}{}".format(k, v) for k, v in background.items()])
        + " background"
    )
    axes.set(
        xlabel="{}-WT".format(target),
        ylabel="{}-Mutant".format(target),
        xlim=lims,
        ylim=lims,
        title=title,
    )
    axes.grid(alpha=0.2)


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
    # axes.grid(alpha=0.2, lw=0.3)
    if add_diag:
        axes.axline((1, 1), (2, 2), lw=0.3, c="grey", linestyle="--", alpha=0.5)


def plot_phenotypes_scatter(
    df,
    axes,
    col="M",
    ref="W",
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
            dx = np.abs(df[["{}_lower".format(ref), "{}_upper".format(ref)]].T - x)
        except KeyError:
            dx = None

        try:
            dy = np.abs(df[["{}_lower".format(col), "{}_upper".format(col)]].T - y)
        except KeyError:
            dy = None

        axes.errorbar(
            x, y, xerr=dx, yerr=dy, lw=0.1, alpha=alpha, ecolor=color, fmt="none"
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
            _, _, V = np.linalg.svd(A-p0)
            p1 = p0 + V[0, :]
            axes.axline(np.exp(p0), np.exp(p1), lw=0.5, c='black', linestyle='--')
            
    set_aspect(axes, add_diag=add_diag)

def add_model_line(axes, theta, gt):
    point1 = (np.exp(theta.loc["WW", "v1"]), np.exp(theta.loc[gt, "v1"]))
    point2 = (np.exp(theta.loc["WW", "v2"]), np.exp(theta.loc[gt, "v2"]))
    axes.axline(
        point1, point2, linestyle="--", lw=0.3, c="black", label="Model"
    )
