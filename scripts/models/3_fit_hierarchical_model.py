#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch

from scripts.utils import get_saturated_basis
from scripts.settings import SEASONS
from scripts.models.hierarchical_model import HierarchicalModel, prepare_data


if __name__ == "__main__":
    n_iter = 10000
    lr = 0.01

    # Load raw data
    plant_data = pd.read_csv("data/plant_data.csv", index_col=0)
    train = np.load("results/train_idx.npy")
    X0 = np.ones((plant_data.shape[0], 1))
    X = get_saturated_basis(plant_data).values
    x1, x2, y, exposure = prepare_data(plant_data)

    print("Training model on 100% of the data")
    model = HierarchicalModel()
    model.set_data(x1, x2, y, exposure)
    model.fit(n_iter=n_iter, lr=lr)
    torch.save(model, "results/hierarchical.pkl")

    history = pd.DataFrame({"loss": model.history})
    history.to_csv("results/hierarchical.history.csv")

    print("Training model on 90% of the data")
    model = HierarchicalModel()
    model.set_data(
        x1.loc[train, :], x2.loc[train, :], y[train], exposure[train]
    )
    model.fit(n_iter=n_iter, lr=lr)
    torch.save(model, "results/hierarchical.train.pkl")

    for season in SEASONS:
        print("Leaving out season: {}".format(season))
        train = (plant_data["Season"] != season).values
        model = HierarchicalModel()
        model.set_data(
            x1.loc[train, :], x2.loc[train, :], y[train], exposure[train]
        )
        model.fit(n_iter=n_iter, lr=lr)
        torch.save(model, "results/hierarchical.{}.pkl".format(season))
