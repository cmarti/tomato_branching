#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import Parameter
from scipy.stats import pearsonr

from scripts.utils import get_saturated_basis
from scripts.settings import SEASONS


class NegativeBinomial(object):
    def __init__(self, mu, alpha):
        self.mu = mu
        self.alpha = alpha

        self.var = mu + alpha * mu**2

        self.p = mu / self.var
        self.n = mu**2 / (self.var - mu)

    def logpmf(self, y):
        coeff = (
            torch.lgamma(self.n + y)
            - torch.lgamma(y + 1)
            - torch.lgamma(self.n)
        )
        return (
            coeff
            + self.n * torch.log(self.p)
            + torch.special.xlog1py(y, -self.p)
        )


class LinearModel(torch.nn.Module):
    def set_data(self, X, counts, exposure, alpha=None):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(counts)
        self.exposure = torch.Tensor(exposure)
        self.init_params(n=self.X.shape[1], alpha=alpha)

    def loglikelihood(self, yhat, counts, exposure):
        mean = yhat * exposure
        alpha = torch.exp(self.log_alpha)
        ll = NegativeBinomial(mean, alpha).logpmf(counts).sum()
        return ll

    def x_to_mu(self, X):
        mu = torch.exp(X @ self.theta)
        return mu

    def predict(self, X):
        X_ = torch.Tensor(X)
        return self.x_to_mu(X_)

    def fit(self, n_iter=1000, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history = []
        pbar = tqdm(range(n_iter), desc="Optimizing log-likelihood")
        for i in pbar:
            optimizer.zero_grad()
            mu = self.x_to_mu(self.X)
            loss = -self.loglikelihood(mu, self.y, self.exposure)
            loss.backward()
            optimizer.step()

            loss_value = loss.detach().item()
            history.append(loss_value)
            report_dict = {"loss": "{:.3f}".format(loss_value)}
            pbar.set_postfix(report_dict)
            # if i == 0:
            #     print(history[-1])
        self.history = history

    def init_params(self, n, alpha=None):
        if alpha is None:
            self.log_alpha = Parameter(torch.zeros(size=(1,)))
        else:
            self.log_alpha = Parameter(
                torch.Tensor(np.log([alpha])), requires_grad=False
            )
        self.theta = Parameter(torch.normal(torch.zeros(size=(n,))))


class MultilinearModel(torch.nn.Module):
    def set_data(self, x1, x2, counts, exposure, alpha=None):
        self.cols1 = x1.columns
        self.cols2 = x2.columns

        self.x1 = torch.Tensor(x1.values)
        self.x2 = torch.Tensor(x2.values)
        self.y = torch.Tensor(counts)
        self.exposure = torch.Tensor(exposure)

        self.init_params(n1=self.x1.shape[1], n2=self.x2.shape[1], alpha=alpha)

    def loglikelihood(self, yhat, counts, exposure):
        mean = yhat * exposure
        alpha = torch.exp(self.log_alpha)
        ll = NegativeBinomial(mean, alpha).logpmf(counts).sum()
        return ll

    def bilinear_function(self, f1, f2, beta):
        b0 = beta[0]
        b1 = torch.exp(beta[1])
        return b0 + f1 + f2 - b1 * f1 * f2

    def x_to_mu(self, x1, x2):
        f1 = self.calc_f(x1, self.theta1)
        f2 = self.calc_f(x2, self.theta2)
        log_mu = self.bilinear_function(f1, f2, self.beta)
        mu = torch.exp(log_mu)
        return mu

    def predict(self, x1, x2):
        x1_ = torch.Tensor(x1.values)
        x2_ = torch.Tensor(x2.values)
        return self.x_to_mu(x1_, x2_)

    def fit(self, n_iter=1000, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history = []
        pbar = tqdm(range(n_iter), desc="Optimizing log-likelihood")
        for i in pbar:
            optimizer.zero_grad()
            mu = self.x_to_mu(self.x1, self.x2)
            loss = -self.loglikelihood(mu, self.y, self.exposure)
            loss.backward()
            optimizer.step()

            loss_value = loss.detach().item()
            history.append(loss_value)
            report_dict = {"loss": "{:.3f}".format(loss_value)}
            pbar.set_postfix(report_dict)
            # if i == 0:
            #     print(history[-1])
        self.history = history

    def summary(self, pred=None, obs=None):
        print("===========================")
        print("Log-likelihood = {:.2f}".format(model.history[-1]))
        print("======= Parameters ========")
        for param, values in self.get_params().items():
            print(param, values)

        if pred is not None and obs is not None:
            r1 = pearsonr(np.log(pred + 1), np.log(obs + 1))[0]
            r2 = pearsonr(np.log(pred + 1e-2), np.log(obs + 1e-2))[0]
            r3 = pearsonr(np.log(pred + 1e-6), np.log(obs + 1e-6))[0]
            print("======= Predictions ========")
            print("log(x+1e-6) Pearson r = {:.2f}".format(r3))
            print("log(x+1e-2) Pearson r = {:.2f}".format(r2))
            print("log(x+1) Pearson r = {:.2f}".format(r1))

    def init_params(self, n1, n2, alpha=None):
        if alpha is None:
            self.log_alpha = Parameter(torch.zeros(size=(1,)))
        else:
            self.log_alpha = Parameter(
                torch.Tensor(np.log([alpha])), requires_grad=False
            )
        self.theta1_raw = Parameter(torch.normal(torch.zeros(size=(n1 - 1,))))
        self.theta2_raw = Parameter(torch.normal(torch.zeros(size=(n2 - 1,))))
        self.beta = Parameter(torch.normal(torch.zeros(size=(2,))))

    @property
    def theta1(self):
        theta1 = torch.zeros(self.theta1_raw.shape[0] + 1)
        theta1[1:] = self.theta1_raw
        return theta1

    @property
    def theta2(self):
        theta2 = torch.zeros(self.theta2_raw.shape[0] + 1)
        theta2[1:] = self.theta2_raw
        return theta2

    def calc_f(self, x, theta):
        return x @ theta

    def get_params(self):
        theta1 = pd.DataFrame(
            {
                "param": self.theta1.detach().numpy(),
                "v1": self.bilinear_function(self.theta1, -1.0, self.beta)
                .detach()
                .numpy(),
                "v2": self.bilinear_function(self.theta1, 1.0, self.beta)
                .detach()
                .numpy(),
            },
            index=self.cols1,
        )
        theta2 = pd.DataFrame(
            {
                "param": self.theta2.detach().numpy(),
                "v1": self.bilinear_function(-1.0, self.theta2, self.beta)
                .detach()
                .numpy(),
                "v2": self.bilinear_function(1.0, self.theta2, self.beta)
                .detach()
                .numpy(),
            },
            index=self.cols2,
        )

        params = {
            "alpha": np.exp(self.log_alpha.detach().numpy()[0]),
            "beta": self.beta.detach().numpy(),
            "theta1": theta1,
            "theta2": theta2,
        }
        return params


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
