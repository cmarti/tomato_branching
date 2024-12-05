#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
from statsmodels.iolib.smpickle import load_pickle
from scripts.settings import SEASONS
from scripts.utils import get_add_basis, get_pairwise_basis, get_saturated_basis
from scripts.models.multilinear_model import encode_data, MultilinearModel


if __name__ == "__main__":
    cols = ["PLT3", "PLT7", "J2", "EJ2", "Season"]

    # Load raw data
    plant_data = pd.read_csv("data/plant_data.csv", index_col=0)
    plant_data["gt"] = ["_".join(x) for x in plant_data[cols].values]
    x1, x2 = encode_data(plant_data)
    cols1, cols2 = x1.columns, x2.columns

    gt_data = (
        plant_data.groupby(["gt"] + cols)[["obs_mean"]]
        .mean()
        .reset_index()
        .set_index("gt")
    )

    print("Building basis for genotype predictions")
    additive_basis = get_add_basis(gt_data)
    pairwise_basis = get_pairwise_basis(gt_data)
    saturated_basis = get_saturated_basis(gt_data)
    x1, x2 = encode_data(gt_data)
    x1 = x1[cols1]
    x2 = x2[cols2]

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

        # Leave season out fitting
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

    print("Making predictions with multilinear models")
    model = torch.load("results/multilinear.pkl")
    mu = np.log(model.predict(x1, x2).detach().numpy())
    gt_data["multilinear_pred"] = mu

    model = torch.load("results/multilinear.train.pkl")
    mu = np.log(model.predict(x1, x2).detach().numpy())
    gt_data["multilinear_train_pred"] = mu

    for season in SEASONS:
        print("\t Leaving season {} out".format(season))
        model = torch.load("results/multilinear.{}.pkl".format(season))
        mu = np.log(model.predict(x1, x2).detach().numpy())
        gt_data["multilinear_{}_pred".format(season)] = mu

    gt_data.to_csv("results/genotype_predictions.csv")
