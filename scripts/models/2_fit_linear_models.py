#!/usr/bin/env python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

from scripts.settings import SEASONS
from scripts.utils import (
    get_saturated_basis,
    get_add_basis,
    get_pairwise_basis,
)


def fit_model_linear(y, X, exposure):
    link = sm.families.links.Identity()
    X_ext = X * np.expand_dims(exposure, 1)

    def loss(alpha):
        model = sm.GLM(
            y,
            X_ext,
            family=sm.families.NegativeBinomial(alpha=alpha, link=link),
        ).fit()
        return -model.llf_scaled()

    res = minimize(
        loss, x0=0.3, method="nelder-mead", bounds=[(0, None)], tol=1e-2
    )
    alpha = res.x[0]
    model = sm.GLM(
        y,
        X_ext,
        family=sm.families.NegativeBinomial(alpha=alpha, link=link),
    ).fit()
    model.alpha = alpha
    return model


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
    np.random.seed(0)
    print("Loading processed data for model fitting")
    plant_data = pd.read_csv("data/plant_data.csv", index_col=0)
    plant_data["gt"] = [
        "_".join(x) for x in plant_data[["PLT3", "PLT7", "J2", "EJ2"]].values
    ]
    gts = np.unique(plant_data["gt"])
    held_out_gts = np.random.choice(gts, size=20, replace=False)
    np.save("results/held_out_gts.npy", held_out_gts)
    test_idx = np.isin(plant_data["gt"], held_out_gts)
    train_idx = ~test_idx
    np.save("results/train_idx.npy", train_idx)
    y = plant_data["branches"]
    exposure = plant_data["influorescences"].values

    print("Constructing basis for regression")
    additive_basis = get_add_basis(plant_data)
    pairwise_basis = get_pairwise_basis(plant_data).drop("PLT3_PLT7", axis=1)
    null_basis = pd.DataFrame(
        np.ones((y.shape[0], 1)), index=additive_basis.index
    )
    saturated_basis = get_saturated_basis(plant_data)
    pairwise_basis.join(plant_data[['gt']]).to_csv('results/pairwise_basis.csv')

    print("Fitting null NB model")
    model = fit_model(y, null_basis, exposure)
    model.save("results/null_model.pkl")

    print("Fitting NB linear")
    data = pd.DataFrame({"obs_mean": plant_data["obs_mean"]})
    results = fit_model_linear(y, additive_basis, exposure)
    data["nb_linear"] = results.predict(additive_basis)
    results.save("results/model.nb_linear.pkl")

    print("Fitting NB log")
    results = fit_model(y, additive_basis, exposure)

    data["nb_log"] = results.predict(additive_basis)
    results.save("results/model.nb_log.pkl")
    data.to_csv("results/plant_predictions.csv")

    print("Fitting saturated Poisson model")
    results = sm.GLM(
        y,
        saturated_basis,
        exposure=exposure,
        family=sm.families.Poisson(),
    ).fit()
    results.save("results/saturated_poisson.pkl")

    print("Fitting saturated NB model")
    model = fit_model(y, saturated_basis, exposure)
    model.save("results/saturated_model.pkl")

    print("Fitting additive and pairwise NB models")
    subsets = {
        "additive": additive_basis,
        "pairwise": pairwise_basis,
    }

    for label, X in subsets.items():
        print("\tFitting model {}: {} params".format(label, X.shape[1]))
        print("\t\tFitting to complete dataset")
        model = fit_model(y, X, exposure)
        model.save("results/{}_model.pkl".format(label))

        print(
            "\t\tFitting to 90% of the genotypes ({} data points)".format(
                train_idx.sum()
            )
        )
        model = fit_model(y[train_idx], X.loc[train_idx], exposure[train_idx])
        model.save("results/{}_model.train.pkl".format(label))

        # Leave season out fitting
        for season in SEASONS:
            test = (plant_data["Season"] == season).values
            train = ~test
            print(
                "\t\tLeaving out season: {} ({} data points)".format(
                    season, train.sum()
                )
            )
            x = X.loc[train, :]
            x = x.loc[:, np.any(x != 0, axis=0)]
            model = fit_model(y[train], x, exposure[train])
            model.save("results/{}_model.{}.pkl".format(label, season))
