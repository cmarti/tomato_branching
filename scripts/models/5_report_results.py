#!/usr/bin/env python
import numpy as np
import pandas as pd

from statsmodels.iolib.smpickle import load_pickle
from scipy.stats import chi2
from scripts.settings import EJ2_SERIES


def lr_test(larger, restricted=None, add_one=False):
    if restricted is None:
        return None
    statistic = -2 * (restricted.llf_scaled() - larger.llf_scaled())
    p_val = chi2.sf(statistic, larger.df_model + add_one - restricted.df_model)

    print(
        "LLR test:\n\tll_diff = {}\n\tdf1 = {}\n\tdf2 = {}\n\tp-value = {}".format(
            statistic / 2,
            larger.df_model + add_one,
            restricted.df_model,
            p_val,
        )
    )


def calc_dev_perc(model):
    print(model.deviance, model.null_deviance)
    return 100 * (1 - model.deviance / model.null_deviance)


if __name__ == "__main__":
    null_model = load_pickle("results/null_model.pkl")
    saturated_model = load_pickle("results/saturated_model.pkl")

    null_ll = null_model.llf_scaled()
    saturated_ll = saturated_model.llf_scaled()
    null_deviance = 2 * (saturated_ll - null_ll)

    print("== Comparing Poisson and NB saturated models")
    saturated_model = load_pickle("results/saturated_model.pkl")
    saturated_poisson = load_pickle("results/saturated_poisson.pkl")
    lr_test(saturated_model, saturated_poisson, add_one=True)

    print("\n== Comparing additive models at different scales")
    print("Deviance of null model = {}".format(null_deviance))

    poisson_linear = load_pickle("results/poisson_linear_model.pkl")
    dev_perc = 100 * (
        1 - poisson_linear.deviance / poisson_linear.null_deviance
    )
    print("Poisson linear % deviance explained = {:.2f}".format(dev_perc))

    poisson_log = load_pickle("results/poisson_log_model.pkl")
    dev_perc = 100 * (1 - poisson_log.deviance / poisson_log.null_deviance)
    print("Poisson log % deviance explained = {:.2f}".format(dev_perc))

    nb_linear = load_pickle("results/model.nb_linear.pkl")
    dev_perc = 100 * (1 - nb_linear.deviance / nb_linear.null_deviance)
    print("NB linear % deviance explained = {:.2f}".format(dev_perc))

    nb_log = load_pickle("results/model.nb_log.pkl")
    dev_perc = 100 * (1 - nb_log.deviance / nb_log.null_deviance)
    print("NB log % deviance explained = {:.2f}".format(dev_perc))

    print("\n== Comparing additive and pairwise NB models == ")
    additive_model = load_pickle("results/additive_model.pkl")
    deviance = 2 * (saturated_ll - additive_model.llf_scaled())
    additive_dev_perc = 100 * (1 - deviance / null_deviance)
    print("Additive % deviance explained = {:.2f}".format(additive_dev_perc))
    print("Additive model AIC = {}".format(additive_model.aic))

    pairwise_model = load_pickle("results/pairwise_model.pkl")
    deviance = 2 * (saturated_ll - pairwise_model.llf_scaled())
    pairwise_dev_perc = 100 * (1 - deviance / null_deviance)
    print("Pairwise % deviance explained = {:.2f}".format(pairwise_dev_perc))
    print("Pairwise model AIC = {}".format(pairwise_model.aic))

    lr_test(pairwise_model, additive_model)

    print("\n== Reporting synergistic interactions ==")
    pairwise_basis = (
        pd.read_csv("data/pairwise_basis.csv", index_col=0)
        .set_index("gt")
        .drop_duplicates()
    )
    c = pd.DataFrame(
        np.zeros((4, pairwise_basis.shape[0])),
        columns=pairwise_basis.index,
        index=["PLT3h_PLT7", "PLT3_PLT7h", "J2h_EJ2(8)", "J2_EJ2(8)h"],
    )
    c.loc["PLT3h_PLT7", ["W_W_W_W", "H_W_W_W", "W_M_W_W", "H_M_W_W"]] = [
        1,
        -1,
        -1,
        1,
    ]
    c.loc["PLT3_PLT7h", ["W_W_W_W", "M_W_W_W", "W_H_W_W", "M_H_W_W"]] = [
        1,
        -1,
        -1,
        1,
    ]
    c.loc["J2h_EJ2(8)", ["W_W_W_W", "W_W_H_W", "W_W_W_M8", "W_W_H_M8"]] = [
        1,
        -1,
        -1,
        1,
    ]
    c.loc["J2_EJ2(8)h", ["W_W_W_W", "W_W_M_W", "W_W_W_H8", "W_W_M_H8"]] = [
        1,
        -1,
        -1,
        1,
    ]
    contrasts = c @ pairwise_basis
    results1 = pairwise_model.t_test(contrasts).summary_frame()
    results1.index = contrasts.index

    contrasts = np.eye(pairwise_model.params.shape[0])
    results2 = pairwise_model.t_test(contrasts).summary_frame()
    results2.index = pairwise_model.params.index
    results2.to_csv("results/pairwise_model.coeffs.csv")

    coefs = ["J2_EJ2({})".format(a) for a in EJ2_SERIES]
    results = pd.concat([results1, results2.loc[coefs, :]])
    results["fold_change"] = np.exp(results["coef"])
    results.to_excel('results/pairwise_interactions.xlsx')
    print(results)

    pairwise = np.array(["_" in x for x in results2.index])
    nsig = np.sum(pairwise & (results2["P>|z|"] < 0.05))
    ntotal = np.sum(pairwise)
    print(
        "Number of significant pairwise coefficients = {} / {}".format(
            nsig, ntotal
        )
    )

    print("\n== Multilinear model ==")
    history = pd.read_csv("results/multilinear.history.csv")
    ll = -history["loss"].values[-1]

    deviance = 2 * (saturated_ll - ll)
    dev_perc = 100 * (1 - deviance / null_deviance)
    print("Multilinear model % deviance explained = {:.2f}".format(dev_perc))
    n_params = 1 + 8 + 2 + 6 * 6
    aic = 2 * n_params - 2 * ll
    print("Multilinear model AIC = {}".format(aic))
    print("Delta AIC with pairwise model = {}".format(aic - pairwise_model.aic))
