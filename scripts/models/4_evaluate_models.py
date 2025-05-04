#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
from statsmodels.iolib.smpickle import load_pickle
import statsmodels.api as sm

from scripts.settings import SEASONS
from scripts.utils import get_add_basis, get_pairwise_basis, get_saturated_basis
from scripts.models.hierarchical_model import encode_data, HierarchicalModel


def calc_deviance(model_ll, saturated_ll):
    deviance = 2 * (model_ll - saturated_ll)
    return deviance


if __name__ == "__main__":
    cols = ["PLT3", "PLT7", "J2", "EJ2", "Season"]

    # Load raw data
    plant_data = pd.read_csv("data/plant_data.csv", index_col=0)
    plant_data["gt"] = ["_".join(x) for x in plant_data[cols].values]
    plant_data["genes"] = ["_".join(x) for x in plant_data[cols[:-1]].values]
    x1, x2 = encode_data(plant_data)
    cols1, cols2 = x1.columns, x2.columns

    gt_data = (
        plant_data.groupby(["gt"] + cols)[["obs_mean"]]
        .mean()
        .reset_index()
        .set_index("gt")
    )
    devs = []

    print("Building basis for genotype predictions")
    additive_basis = get_add_basis(gt_data)
    pairwise_basis = get_pairwise_basis(gt_data)
    saturated_basis = get_saturated_basis(gt_data)
    x1, x2 = encode_data(gt_data)
    x1 = x1[cols1]
    x2 = x2[cols2]

    print("Loading held out genotypes")
    held_out_gts = np.load("results/held_out_gts.npy", allow_pickle=True)
    gt = np.array(["_".join(x.split("_")[:-1]) for x in plant_data["gt"]])
    held_out_gts = np.load("results/held_out_gts.npy", allow_pickle=True)
    test_idx = np.isin(gt, held_out_gts)
    test_plant_data = plant_data.loc[test_idx, :]

    null_test_model = load_pickle("results/null_model.pkl")
    yhat = test_plant_data["influorescences"] * np.exp(
        null_test_model.params[0]
    )
    y = test_plant_data["branches"]
    null_test_ll = null_test_model.family.loglike_obs(y, yhat).sum()
    saturated_test_ll = load_pickle(
        "results/saturated_model.test.pkl"
    ).llf_scaled()

    print("Making predictions with linear models")
    subsets = {
        "saturated": saturated_basis,
        "additive": additive_basis,
        "pairwise": pairwise_basis,
    }

    cols = [""]
    for label, X in subsets.items():
        print("\tLoading model {}: {} params".format(label, X.shape[1]))
        model = load_pickle("results/{}_model.pkl".format(label))
        params = model.params.index
        estimates = model.t_test(X[params]).summary_frame()
        gt_data["{}_pred".format(label)] = estimates["coef"].values
        gt_data["{}_lower".format(label)] = estimates["Conf. Int. Low"].values
        gt_data["{}_upper".format(label)] = estimates["Conf. Int. Upp."].values

        if label in ["saturated", "poisson_linear", "poisson_log"]:
            continue

        model = load_pickle("results/{}_model.train.pkl".format(label))

        params = model.params.index
        estimates = model.t_test(X[params]).summary_frame()
        gt_data["{}_train_pred".format(label)] = estimates["coef"].values
        gt_data["{}_train_lower".format(label)] = estimates[
            "Conf. Int. Low"
        ].values
        gt_data["{}_train_upper".format(label)] = estimates[
            "Conf. Int. Upp."
        ].values

        # Compute deviance
        colname = "{}_pred".format(label)
        test_plant_data = test_plant_data.join(gt_data[[colname]], on="gt")
        yhat = (
            np.exp(test_plant_data[colname])
            * test_plant_data["influorescences"]
        )
        y = test_plant_data["branches"]
        ll = model.family.loglike_obs(y, yhat).sum()
        null_deviance = calc_deviance(null_test_ll, saturated_test_ll)
        deviance = calc_deviance(ll, saturated_test_ll)
        dev_perc = 100 * (1 - deviance / null_deviance)
        devs.append(
            {
                "model": label,
                "test_data": "test",
                "dev_perc": dev_perc,
                "ll": ll,
                "n_plant": y.shape[0],
            }
        )

        for season in SEASONS:
            print("\t\t Leaving season {} out".format(season))
            model_label = "{}_model.{}".format(label, season)
            model = load_pickle("results/{}.pkl".format(model_label))
            params = model.params.index
            estimates = model.t_test(X[params]).summary_frame()
            gt_data["{}_pred".format(model_label)] = estimates["coef"].values
            gt_data["{}_lower".format(model_label)] = estimates[
                "Conf. Int. Low"
            ].values
            gt_data["{}_upper".format(model_label)] = estimates[
                "Conf. Int. Upp."
            ].values

            # Add plant level model predictions
            season_data = plant_data.loc[plant_data["Season"] == season, :]
            pred = pd.DataFrame({"pred": model.predict(X[params])})
            pred.index = gt_data.index
            season_data = season_data.join(pred, on="gt")

            yhat = season_data["pred"] * season_data["influorescences"]
            y = season_data["branches"]
            ll = model.family.loglike_obs(y, yhat).sum()

            null_season_model = load_pickle(
                "results/null_model.{}.pkl".format(season)
            )
            yhat = (
                np.exp(null_season_model.params[0])
                * season_data["influorescences"]
            )
            null_season_ll = null_season_model.family.loglike_obs(y, yhat).sum()
            sat_season_ll = load_pickle(
                "results/saturated_model.test.{}.pkl".format(season)
            ).llf_scaled()

            null_deviance = calc_deviance(null_season_ll, sat_season_ll)
            deviance = calc_deviance(ll, sat_season_ll)
            dev_perc = 100 * (1 - deviance / null_deviance)
            devs.append(
                {
                    "model": label,
                    "test_data": season,
                    "ll": ll,
                    "dev_perc": dev_perc,
                    "n_plant": y.shape[0],
                }
            )

    print("Making predictions with hierarchical models")
    model = torch.load("results/hierarchical.pkl")
    mu = np.log(model.predict(x1, x2).detach().numpy())
    gt_data["hierarchical_pred"] = mu

    model = torch.load("results/hierarchical.train.pkl")
    mu = np.log(model.predict(x1, x2).detach().numpy())
    gt_data["hierarchical_train_pred"] = mu

    # Deviance calculations
    test_plant_data = test_plant_data.join(
        gt_data[["hierarchical_train_pred"]], on="gt"
    )
    yhat = torch.Tensor(
        np.exp(test_plant_data["hierarchical_train_pred"]).values
    )
    y = torch.Tensor(test_plant_data["branches"].values)
    exposure = torch.Tensor(test_plant_data["influorescences"].values)
    ll = model.loglikelihood(yhat, counts=y, exposure=exposure).item()

    null_deviance = calc_deviance(null_test_ll, saturated_test_ll)
    deviance = calc_deviance(ll, saturated_test_ll)
    dev_perc = 100 * (1 - deviance / null_deviance)
    devs.append(
        {
            "model": "hierarchical",
            "test_data": "test",
            "ll": ll,
            "dev_perc": dev_perc,
            "n_plant": y.shape[0],
        }
    )

    for season in SEASONS:
        print("\t Leaving season {} out".format(season))
        model = torch.load("results/hierarchical.{}.pkl".format(season))
        mu = np.log(model.predict(x1, x2).detach().numpy())
        colname = "hierarchical_{}_pred".format(season)
        gt_data[colname] = mu

        # Deviance calculations
        season_data = plant_data.loc[plant_data["Season"] == season, :]
        season_data = season_data.join(gt_data[[colname]], on="gt")

        yhat = torch.Tensor(np.exp(season_data[colname]).values)
        y = torch.Tensor(season_data["branches"].values)
        exposure = torch.Tensor(season_data["influorescences"].values)

        ll = model.loglikelihood(yhat, counts=y, exposure=exposure).item()

        null_season_model = load_pickle(
            "results/null_model.{}.pkl".format(season)
        )
        yhat = (
            np.exp(null_season_model.params[0]) * season_data["influorescences"]
        )
        null_season_ll = null_season_model.family.loglike_obs(
            y.numpy(), yhat
        ).sum()
        sat_test_model = load_pickle(
            "results/saturated_model.test.{}.pkl".format(season)
        )
        saturated_ll = sat_test_model.llf_scaled()

        null_deviance = calc_deviance(null_season_ll, saturated_ll)
        deviance = calc_deviance(ll, saturated_ll)
        dev_perc = 100 * (1 - deviance / null_deviance)
        devs.append(
            {
                "model": "hierarchical",
                "test_data": season,
                "ll": ll,
                "dev_perc": dev_perc,
                "n_plant": y.shape[0],
            }
        )

    gt_data.to_csv("results/genotype_predictions.csv")

    devs = pd.DataFrame(devs)
    devs.to_csv("results/model.deviance_explained.csv")
    print("Done")
