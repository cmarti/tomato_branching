#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch

from scripts.utils import get_saturated_basis
from scripts.settings import SEASONS
from scripts.models.multilinear_model import MultilinearModel


def encode_data(plant_data):
    plant_data["s1"] = [
        "{}{}".format(*x) for x in plant_data[["PLT3", "PLT7"]].values
    ]
    plant_data["s2"] = [
        "{}{}".format(*x) for x in plant_data[["J2", "EJ2"]].values
    ]
    # seasons = pd.DataFrame(
    #     np.vstack([plant_data["Season"] == s for s in SEASONS[1:]]).T,
    #     columns=SEASONS[1:],
    # )
    cols1 = plant_data["s1"].unique()
    cols2 = plant_data["s2"].unique()
    x1 = pd.DataFrame(
        np.vstack([plant_data["s1"] == s for s in cols1]).T, columns=cols1
    )  # .join(seasons)
    x2 = pd.DataFrame(
        np.vstack([plant_data["s2"] == s for s in cols2]).T, columns=cols2
    )  # .join(seasons)
    return (x1, x2)


def prepare_data(plant_data):
    x1, x2 = encode_data(plant_data)
    y = plant_data["branches"].values
    exposure = plant_data["influorescences"].values
    return (x1, x2, y, exposure)


if __name__ == "__main__":
    n_iter = 10000
    lr = 0.01
    
    # Load raw data
    plant_data = pd.read_csv("data/plant_data.csv", index_col=0)
    train = np.load("results/train_idx.npy")
    X0 = np.ones((plant_data.shape[0], 1))
    X = get_saturated_basis(plant_data).values
    x1, x2, y, exposure = prepare_data(plant_data)

    print("Training model on 90% of the data")
    model = MultilinearModel()
    model.set_data(
        x1.loc[train, :], x2.loc[train, :], y[train], exposure[train]
    )
    model.fit(n_iter=n_iter, lr=lr)
    torch.save(model, "results/multilinear.train.pkl")

    print("Training model on 100% of the data")
    model = MultilinearModel()
    model.set_data(x1, x2, y, exposure)
    model.fit(n_iter=n_iter, lr=lr)
    torch.save(model, "results/multilinear.pkl")

    history = pd.DataFrame({"loss": model.history})
    history.to_csv("results/multilinear.history.csv")

    for season in SEASONS:
        print("Leaving out season: {}".format(season))
        train = (plant_data["Season"] != season).values
        model = MultilinearModel()
        model.set_data(
            x1.loc[train, :], x2.loc[train, :], y[train], exposure[train]
        )
        model.fit(n_iter=n_iter, lr=lr)
        torch.save(model, "results/multilinear.{}.pkl".format(season))
