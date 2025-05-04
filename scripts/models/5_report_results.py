#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch

from statsmodels.iolib.smpickle import load_pickle
from scipy.stats import chi2
from scripts.settings import EJ2_SERIES
from scripts.models.hierarchical_model import HierarchicalModel


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


if __name__ == "__main__":
    models_deviance = []

    null_model = load_pickle("results/null_model.pkl")
    saturated_model = load_pickle("results/saturated_model.pkl")

    null_ll = null_model.llf_scaled()
    saturated_ll = saturated_model.llf_scaled()
    null_deviance = 2 * (saturated_ll - null_ll)

    def calc_dev_perc(ll):
        deviance = 2 * (saturated_ll - ll)
        dev_perc = 100 * (1 - deviance / null_deviance)
        return dev_perc

    models_deviance.append({"model": "null model", "deviance": 0})

    print("== Comparing Poisson and NB saturated models")
    saturated_model = load_pickle("results/saturated_model.pkl")
    saturated_poisson = load_pickle("results/saturated_poisson.pkl")
    lr_test(saturated_model, saturated_poisson, add_one=True)

    print("\n== Comparing additive models at different scales")
    nb_linear = load_pickle("results/model.nb_linear.pkl")
    dev_perc = calc_dev_perc(nb_linear.llf_scaled())
    print("linear % deviance explained = {:.2f}".format(dev_perc))
    models_deviance.append(
        {"model": "negative binomial linear", "deviance": dev_perc}
    )

    nb_log = load_pickle("results/model.nb_log.pkl")
    dev_perc = calc_dev_perc(nb_log.llf_scaled())
    print("log % deviance explained = {:.2f}".format(dev_perc))
    models_deviance.append(
        {"model": "negative binomial log", "deviance": dev_perc}
    )

    print("\n== Comparing additive and pairwise models == ")
    additive_model = load_pickle("results/additive_model.pkl")
    dev_perc = calc_dev_perc(additive_model.llf_scaled())
    models_deviance.append({"model": "additive model", "deviance": dev_perc})
    print("Additive % deviance explained = {:.2f}".format(dev_perc))

    pairwise_model = load_pickle("results/pairwise_model.pkl")
    dev_perc = calc_dev_perc(pairwise_model.llf_scaled())
    print("Pairwise % deviance explained = {:.2f}".format(dev_perc))
    models_deviance.append({"model": "pairwise model", "deviance": dev_perc})
    lr_test(pairwise_model, additive_model)

    print("\n== Reporting synergistic interactions ==")
    pairwise_basis = (
        pd.read_csv("results/pairwise_basis.csv", index_col=0)
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
    results["fold_change_lower"] = np.exp(results["Conf. Int. Low"])
    results["fold_change_upper"] = np.exp(results["Conf. Int. Upp."])
    results.to_excel("results/pairwise_interactions.xlsx")
    print(results.iloc[:, [3, 6, 7, 8]])

    pairwise_epistatic_coef = np.array(
        ["_" in x and "het" not in x for x in results2.index]
    )
    nsig = np.sum(pairwise_epistatic_coef & (results2["P>|z|"] < 0.05))
    ntotal = np.sum(pairwise_epistatic_coef)
    print(
        "Number of significant pairwise coefficients = {} / {}".format(
            nsig, ntotal
        )
    )
    non_paralogous_epi_coeff = np.array(
        [
            "_" in x and "het" not in x and "J2_EJ2" not in x
            for x in results2.index
        ]
    )
    nsig = np.sum(non_paralogous_epi_coeff & (results2["P>|z|"] < 0.05))
    ntotal = np.sum(non_paralogous_epi_coeff)
    print(
        "Number of significant non-paralogous pairwise coefficients = {} / {}".format(
            nsig, ntotal
        )
    )

    print("\n== hierarchical model ==")
    hierarchical_model = torch.load("results/hierarchical.pkl")
    ll = hierarchical_model.llf
    dev_perc = calc_dev_perc(ll)
    models_deviance.append(
        {"model": "hierarchical model", "deviance": dev_perc}
    )
    print("Hierarchical model % deviance explained = {:.2f}".format(dev_perc))
    hierarchical_aic = 2 * hierarchical_model.n_params - 2 * ll
    delta_aic = hierarchical_aic - pairwise_model.aic
    print("Hierarchical model AIC = {}".format(hierarchical_aic))
    print(
        "Delta AIC with pairwise model = {} ({} times more likely)".format(
            delta_aic, np.exp(-delta_aic / 2)
        )
    )

    models_deviance = pd.DataFrame(models_deviance).set_index("model")
    models_deviance.to_csv("results/models.deviance.csv")
