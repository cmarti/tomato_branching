#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import Parameter
from scipy.stats import pearsonr
from scripts.settings import EJ2_SERIES, EJ2_SERIES_LABELS


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


class BaseModel(torch.nn.Module):
    def set_data(self, x1, x2, x3, counts, exposure):
        self.cols1 = x1.columns
        self.cols2 = x2.columns
        self.cols3 = x3.columns

        self.x1 = torch.Tensor(x1.values)
        self.x2 = torch.Tensor(x2.values)
        self.x3 = torch.Tensor(x3.values)
        self.y = torch.Tensor(counts)
        self.exposure = torch.Tensor(exposure)

        self.init_params(n1=self.x1.shape[1],
                         n2=self.x2.shape[1],
                         n3=self.x3.shape[1])

    def loglikelihood(self, yhat, counts, exposure):
        mean = yhat * exposure
        alpha = torch.exp(self.log_alpha)
        ll = NegativeBinomial(mean, alpha).logpmf(counts).sum()
        return ll

    def bilinear_function(self, f1, f2, beta):
        b1 = torch.exp(beta)
        return f1 + f2 - b1 * f1 * f2

    def x_to_mu(self, x1, x2, x3):
        f1 = self.calc_f(x1, self.theta1)
        f2 = self.calc_f(x2, self.theta2)
        log_mu = self.bilinear_function(f1, f2, self.beta)
        # log_mu += x3 @ self.theta3
        mu = torch.exp(log_mu)
        return mu

    def predict(self, x1, x2, x3):
        x1_ = torch.Tensor(x1.values)
        x2_ = torch.Tensor(x2.values)
        x3_ = torch.Tensor(x3.values)
        return self.x_to_mu(x1_, x2_, x3_)

    def fit(self, n_iter=1000, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history = []
        pbar = tqdm(range(n_iter), desc="Optimizing log-likelihood")
        for i in pbar:
            optimizer.zero_grad()
            mu = self.x_to_mu(self.x1, self.x2, self.x3)
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


class BilinearModel(BaseModel):
    def init_params(self, n1, n2, n3):
        self.log_alpha = Parameter(torch.zeros(size=(1,)))
        self.theta1_raw = Parameter(torch.normal(torch.zeros(size=(n1 - 1,))))
        self.theta2 = Parameter(torch.normal(torch.zeros(size=(n2,))))
        self.theta3 = Parameter(torch.normal(torch.zeros(size=(n3,))))
        self.beta = Parameter(torch.normal(torch.zeros(size=(1,))))

    @property
    def theta1(self):
        theta1 = torch.zeros(self.theta1_raw.shape[0] + 1)
        theta1[1:] = self.theta1_raw
        return theta1

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
        
        theta3 = pd.DataFrame(
            {
                "param": self.theta3.detach().numpy(),
            },
            index=self.cols3,
        )

        params = {
            "alpha": np.exp(self.log_alpha.detach().numpy()[0]),
            "beta": self.beta.detach().numpy(),
            "theta1": theta1,
            "theta2": theta2,
            "theta3": theta3,
        }
        return params


def get_dosage_encoding(plant_data):
    dosage_encoding = {"W": 0, "H": 1, "M": 2.0}

    plts = {
        "wt": np.ones(plant_data.shape[0]),
        "PLT3": [dosage_encoding[x] for x in plant_data["PLT3"]],
        "PLT7": [dosage_encoding[x] for x in plant_data["PLT7"]],
    }
    js = {
        "wt": np.ones(plant_data.shape[0]),
        "J2": [dosage_encoding[x] for x in plant_data["J2"]],
    }
    for variant in EJ2_SERIES:
        dosage_encoding = {"H{}".format(variant): 1, "M{}".format(variant): 2.0}
        js["EJ2({})".format(variant)] = [
            dosage_encoding.get(x, 0) for x in plant_data["EJ2"]
        ]
    x1 = pd.DataFrame(plts)
    x2 = pd.DataFrame(js)
    return (x1, x2)


def get_j2ej2_encoding(plant_data):
    data = {"wt": np.ones(plant_data.shape[0])}
    # code = {'W': 0, 'H': 1, 'M': 2}
    for i in (1, 3, 4, 6, 7, 8):
        data["H{}".format(i)] = [
            float(x[-1] == str(i) and x[0] == "H") for x in plant_data["EJ2"]
        ]
        data["M{}".format(i)] = [
            float(x[-1] == str(i) and x[0] == "M") for x in plant_data["EJ2"]
        ]
        # data['EJ2({})'.format(i)] = [code[x[0]] for x in plant_data['EJ2']]

    v = np.array([x[0] + y for x, y in plant_data[["EJ2", "J2"]].values])
    for c in ["WH", "HH", "MH", "WM", "HM", "MM"]:
        data[c] = v == c

    data = pd.DataFrame(data).astype(float)
    return data


def get_j2ej2_encoding2(plant_data):
    data = {}
    # data = {"wt": np.ones(plant_data.shape[0])}

    v = np.array([x[0] + y for x, y in plant_data[["EJ2", "J2"]].values])
    for c in ["WW", "WH", "WM", "HW", "HH", "HM", "MW", "MH", "MM"]:
        data[c] = v == c

    allele = np.array([x[-1] for x in plant_data["EJ2"]])
    gt = np.array([x[0] for x in plant_data["EJ2"]])
    for i in (3, 4, 6, 7, 8):
        data["EJ2({})H".format(i)] = (allele == str(i)) * (gt == 'H')
        data["EJ2({})M".format(i)] = (allele == str(i)) * (gt == 'M')
    
    # for i in (3, 4, 6, 7, 8):
        # data["EJ2({})".format(i)] = (allele == str(i))

    data = pd.DataFrame(data).astype(float)
    return data


if __name__ == "__main__":
    # Load raw data
    plant_data = pd.read_csv("data/plant_data.csv", index_col=0)

    # Prepare data
    # plant_data['s1'] = ['{}{}-{}'.format(*x) for x in plant_data[['PLT3', 'PLT7', 'Season']].values]
    # plant_data['s2'] = ['{}{}-{}'.format(*x) for x in plant_data[['J2', 'EJ2', 'Season']].values]
    plant_data["s1"] = [
        "{}{}".format(*x) for x in plant_data[["PLT3", "PLT7"]].values
    ]
    plant_data["s2"] = [
        "{}{}".format(*x) for x in plant_data[["J2", "EJ2"]].values
    ]

    cols1 = plant_data["s1"].unique()
    cols2 = plant_data["s2"].unique()

    y = plant_data["branches"].values
    exposure = plant_data["influorescences"].values

    # # Fit model
    # x1, x2 = get_dosage_encoding(plant_data)
    # x1_, x2_ = torch.Tensor(x1.drop_duplicates().values), x2.drop_duplicates
    # model = ExpBilinearModel()
    # model.set_data(x1, x2, y, exposure)
    # model.fit(n_iter=5000, lr=0.025)
    # # model.fit(n_iter=5000, lr=0.025)
    # # model.fit(n_iter=5000, lr=0.01)

    # # print(model.calc_f(x1_, model.theta1, model.c1))

    # mu = model.predict(x1, x2).detach()
    # model.summary(pred=mu, obs=plant_data["obs_mean"])
    # params = model.get_params()
    # plant_data["pred"] = np.log(mu)
    # plant_data.to_csv("results/data.predictions.expbilinear.csv")

    # params["theta1"].to_csv("results/expbilinear_model.theta1.csv")
    # params["theta2"].to_csv("results/expbilinear_model.theta2.csv")
    # exit()

    # Fit model
    x1 = pd.DataFrame(
        np.vstack([plant_data["s1"] == s for s in cols1]).T, columns=cols1
    )
    x2 = pd.DataFrame(
        np.vstack([plant_data["s2"] == s for s in cols2]).T, columns=cols2
    )
    x2 = get_j2ej2_encoding2(plant_data)

    # Get predictive subsets
    x1_pred1 = pd.DataFrame(
        np.eye(x1.shape[1]), columns=x1.columns, index=x1.columns
    )
    x2_pred2 = pd.DataFrame(
        np.eye(9),
        columns=x2.columns[:9],
        index=x2.columns[:9],
    )
    for a in x2.columns[9:]:
        x2_pred2[a] = 0

    x2_pred1 = pd.DataFrame(
        np.vstack([[1] + [0] * (x2_pred2.shape[1] - 1)] * x1_pred1.shape[0]),
        columns=x2_pred2.columns,
    )
    x1_pred2 = pd.DataFrame(
        np.vstack([[1] + [0] * (x1_pred1.shape[1] - 1)] * x2_pred2.shape[0]),
        columns=x1_pred1.columns,
    )
    x2_pred1, x3_pred1 = x2_pred1.iloc[:, :], x2_pred1.iloc[:, 9:]
    x2_pred2, x3_pred2 = x2_pred2.iloc[:, :], x2_pred2.iloc[:, 9:]
    # x3_pred2['EJ2(8)'] = 1
    # x3_pred2.loc['WW', 'EJ2(8)'] = 0
    
    x2, x3 = x2.iloc[:, :], x2.iloc[:, 9:]

    model = BilinearModel()
    model.set_data(x1, x2, x3, y, exposure)
    model.fit(n_iter=5000, lr=0.01)

    mu = model.predict(x1, x2, x3).detach()
    model.summary(pred=mu, obs=plant_data["obs_mean"])
    params = model.get_params()

    plant_data["pred"] = np.log(mu)
    plant_data.to_csv("results/data.predictions.bilinear.csv")

    params["theta1"].to_csv("results/bilinear_model.theta1.csv")
    params["theta2"].to_csv("results/bilinear_model.theta2.csv")

    pred1 = np.log(model.predict(x1_pred1, x2_pred1, x3_pred1).detach().numpy())
    pred1 = pd.DataFrame({"y": pred1}, index=x1_pred1.index.values)
    pred1.to_csv("results/bilinear_model.pred1.csv")

    pred2 = np.log(model.predict(x1_pred2, x2_pred2, x3_pred2).detach().numpy())
    pred2 = pd.DataFrame({"y": pred2}, index=x2_pred2.index.values)
    pred2.to_csv("results/bilinear_model.pred2.csv")
