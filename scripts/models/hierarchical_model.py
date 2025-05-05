#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import Parameter


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


class HierarchicalModel(torch.nn.Module):
    def set_data(self, x1, x2, counts, exposure, alpha=None):
        self.cols1 = x1.columns
        self.cols2 = x2.columns

        self.x1 = torch.Tensor(x1.values)
        self.x2 = torch.Tensor(x2.values)
        self.y = torch.Tensor(counts)
        self.exposure = torch.Tensor(exposure)

        self.init_params(n1=self.x1.shape[1], n2=self.x2.shape[1], alpha=alpha)

    def init_params(self, n1, n2, alpha=None):
        self.log_alpha = Parameter(torch.zeros(size=(1,)))
        self.theta1_raw = Parameter(torch.normal(torch.zeros(size=(n1 - 1,))))
        self.theta2_raw = Parameter(torch.normal(torch.zeros(size=(n2 - 1,))))
        self.theta_wt = Parameter(torch.normal(torch.zeros(size=(1,))))
        self.log_theta_int = Parameter(torch.normal(torch.zeros(size=(1,))))
        self.n_params = n1 + n2 + 1

        print(self.log_alpha, self.log_theta_int, self.theta_wt)
        print(self.theta1_raw)
        print(self.theta2_raw)

    def calc_loglikelihood(self, yhat, counts, exposure):
        mean = yhat * exposure
        alpha = torch.exp(self.log_alpha)
        ll = NegativeBinomial(mean, alpha).logpmf(counts).sum()
        return ll

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

    def calc_phi(self, x, theta):
        return x @ theta

    def calc_multilinear_function(self, phi1, phi2):
        theta_int = torch.exp(self.log_theta_int)
        return self.theta_wt + phi1 + phi2 - theta_int * phi1 * phi2

    def x_to_mu(self, x1, x2):
        phi1 = self.calc_phi(x1, self.theta1)
        phi2 = self.calc_phi(x2, self.theta2)
        log_mu = self.calc_multilinear_function(phi1, phi2)
        mu = torch.exp(log_mu)
        return mu

    def predict(self, x1, x2):
        x1_ = torch.Tensor(x1.values)
        x2_ = torch.Tensor(x2.values)
        return self.x_to_mu(x1_, x2_)

    def fit(self, n_iter=1000, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, maximize=True)

        self.history = []
        pbar = tqdm(range(n_iter), desc="Optimizing log-likelihood")
        for i in pbar:
            optimizer.zero_grad()
            mu = self.x_to_mu(self.x1, self.x2)
            ll = self.calc_loglikelihood(mu, self.y, self.exposure)
            ll.backward()
            optimizer.step()

            self.llf = ll.detach().item()
            self.history.append(self.llf)
            pbar.set_postfix({"ll": "{:.3f}".format(self.llf)})


def encode_data(plant_data):
    plant_data["s1"] = [
        "{}{}".format(*x) for x in plant_data[["PLT3", "PLT7"]].values
    ]
    plant_data["s2"] = [
        "{}{}".format(*x) for x in plant_data[["J2", "EJ2"]].values
    ]
    cols1 = plant_data["s1"].unique()
    cols2 = plant_data["s2"].unique()
    x1 = pd.DataFrame(
        np.vstack([plant_data["s1"] == s for s in cols1]).T, columns=cols1
    )
    x2 = pd.DataFrame(
        np.vstack([plant_data["s2"] == s for s in cols2]).T, columns=cols2
    )
    return (x1, x2)


def prepare_data(plant_data):
    x1, x2 = encode_data(plant_data)
    y = plant_data["branches"].values
    exposure = plant_data["influorescences"].values
    return (x1, x2, y, exposure)
