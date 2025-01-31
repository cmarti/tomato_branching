#!/usr/bin/env python
import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy.stats import chi2
from scipy.optimize import minimize
from scripts.utils import (
    get_constant_basis,
    get_additive_basis,
    get_dominance_basis,
    get_env_basis,
    get_gxg_basis,
    get_gxd_basis,
    get_dxd_basis,
    get_x_basis,
    get_3way_basis,
    get_full_basis,
)


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


def fit_model(y, X, exposure):
    def loss(alpha):
        model = sm.GLM(
            y,
            X,
            family=sm.families.NegativeBinomial(alpha=alpha),
            exposure=exposure,
        ).fit()
        return -model.llf_scaled()

    res = minimize(
        loss, x0=0.3, method="nelder-mead", bounds=[(0, None)], tol=1e-2
    )
    alpha = res.x[0]
    model = sm.GLM(
        y,
        X,
        family=sm.families.NegativeBinomial(alpha=alpha),
        exposure=exposure,
    ).fit()
    model.alpha = alpha
    return model


if __name__ == "__main__":
    # Load raw data
    print("Loading processed data for model fitting")
    plant_data = pd.read_csv("data/plant_data.csv", index_col=0)
    y = plant_data["branches"]
    exposure = plant_data["influorescences"].values

    # # Use saturated model to learn overdispersion parameter that best captures within genotype variability
    # print('Fitting saturated model to learn plant-plant variability')
    # saturated_basis = get_saturated_basis(plant_data)
    # saturated_model = sm.NegativeBinomial(y, saturated_basis, exposure=exposure).fit(maxiter=2000)
    # saturated_model.save('results/saturated_model.pkl')
    # alpha = saturated_model.params.loc['alpha']
    # print('\tEstimated alpha = {:.2f}'.format(alpha))

    print("Constructing basis")
    c = get_constant_basis(plant_data)
    g = get_additive_basis(plant_data)
    d = get_dominance_basis(plant_data)
    e = get_env_basis(plant_data)
    gxg = get_gxg_basis(g)
    gxd = get_gxd_basis(g, d)
    dxd = get_dxd_basis(d)
    gxe = get_x_basis(g, e)
    dxe = get_x_basis(d, e)
    g3, d3 = get_3way_basis(g, d, keep_plt7=True)

    print("Fitting models of increasing complexity")
    subsets = {
        "constant": c,
        "additive": g,
        "dominance": d,
        "environment": e,
        "gxg": gxg,
        "gxd": gxd,
        "dxd": dxd,
        "gxe": gxe,
        "dxe": dxe,
        "g3": g3,
        "d3": d3,
    }

    # for label, X in [('pairwise', get_full_basis(plant_data, third_order=False, dominance=False)),
    #                  ('threeway', get_full_basis(plant_data, third_order=True, dominance=False)),
    #                  ('pw_dom', get_full_basis(plant_data, third_order=False, dominance=True)),
    #                  ('threeway_noplt7', get_full_basis(plant_data, third_order=True, dominance=True, keep_plt7=False)),
    #                  ('threeway_dom', get_full_basis(plant_data, third_order=True, dominance=True, keep_plt7=True)),
    #                  ]:
    #     print('\tFitting model {}: {} params'.format(label, X.shape[1]))
    #     model = fit_model(y, X, exposure)
    #     model.save('results/model.{}.pkl'.format(label))

    basis = []
    for label, subset in subsets.items():
        basis.append(subset)
        X = pd.concat(basis, axis=1)
        print("\tFitting model {}: {} params".format(label, X.shape[1]))
        model = fit_model(y, X, exposure)
        model.save("results/model.{}.pkl".format(label))
