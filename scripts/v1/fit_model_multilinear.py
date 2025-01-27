#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import Parameter
from scipy.stats import pearsonr

from scripts.settings import EJ2_SERIES


class NegativeBinomial(object):
    def __init__(self, mu, alpha):
        self.mu = mu
        self.alpha = alpha
        
        self.var = mu + alpha * mu ** 2

        self.p = mu / self.var
        self.n = mu ** 2 / (self.var - mu)
    
    def logpmf(self, y):
        coeff = torch.lgamma(self.n + y) - torch.lgamma(y + 1) - torch.lgamma(self.n)
        return(coeff + self.n * torch.log(self.p) + torch.special.xlog1py(y, -self.p))

class BaseMultilinearModel(torch.nn.Module):
    @property
    def plt_effects(self):
        return(torch.exp(self.plt_effects_log))

    def loglikelihood(self, yhat, counts, exposure):
        mean = yhat * exposure
        alpha = torch.exp(self.log_alpha)
        ll = NegativeBinomial(mean, alpha).logpmf(counts).sum()
        return(ll)
    
    def summary(self, pred=None, obs=None):
        print('===========================')
        print('Log-likelihood = {:.2f}'.format(model.history[-1]))
        print('======= Parameters ========')
        for param, values in self.get_params().items():
            print(param)
            print(values)

        if pred is not None and obs is not None:
            r1 = pearsonr(np.log(pred + 1), np.log(obs + 1))[0]
            r2 = pearsonr(np.log(pred + 1e-2), np.log(obs + 1e-2))[0]
            r3 = pearsonr(np.log(pred + 1e-6), np.log(obs + 1e-6))[0]
            print('======= Predictions ========')
            print('log(x+1e-6) Pearson r = {:.2f}'.format(r3))
            print('log(x+1e-2) Pearson r = {:.2f}'.format(r2))
            print('log(x+1) Pearson r = {:.2f}'.format(r1))
    
    
class MultilinearModel(BaseMultilinearModel):
    def __init__(self, plts, js, seasons):
        super().__init__()
        self.plt_labels = plts
        self.js_labels = js
        self.season_labels = seasons
        
    def init_params(self, n_plts=7, n_js=38, n_seasons=4):
        self.log_alpha = Parameter(torch.zeros(size=(1,)))
        self.max_value = Parameter(torch.ones(size=(1,)))
        self.plt_effects_log = Parameter(torch.zeros(size=(n_plts,)))
        self.js_effects_log = Parameter(torch.zeros(size=(n_js,)))
        self.seasons = Parameter(torch.zeros(size=(n_seasons,)))
        self.seasons0 = Parameter(torch.zeros(size=(n_seasons,)))
        self.wt = Parameter(torch.zeros(size=(1,)))
    
    @property
    def js_effects(self):
        return(torch.exp(self.js_effects_log))

    def predict(self, X_plts, X_js, X_seasons):
        baseline = X_seasons @ self.seasons
        plts = baseline + X_plts @ self.plt_effects
        plts_min = plts.min()
        plts_max = self.max_value #plts.max()
        p = (plts_max - plts) / (plts_max - plts_min)
        js = p * (X_js @ self.js_effects + baseline)
        mu = torch.exp(plts + js + X_seasons @ self.seasons0)
        return(mu)
    
    def fit(self, X_plts, X_js, X_seasons, counts, exposure, n_iter=1000, lr=0.1):
        self.init_params(n_plts=X_plts.shape[1], 
                         n_js=X_js.shape[1],
                         n_seasons=X_seasons.shape[1])
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history = []
        for i in tqdm(range(n_iter)):
            optimizer.zero_grad()
            yhat = self.predict(X_plts, X_js, X_seasons)
            loss = -self.loglikelihood(yhat, counts, exposure)
            loss.backward()
            history.append(loss.detach().item())
            # if i == 0:
            #     print(history[-1])
            optimizer.step()
        self.history = history
    
    def get_params(self):
        params = {'alpha': np.exp(self.log_alpha.detach().numpy()[0]),
                  'max_value': self.max_value.detach().numpy()[0],
                  'seasons': pd.Series(self.seasons.detach(), index=self.season_labels),
                  'seasons0': pd.Series(self.seasons0.detach(), index=self.season_labels),
                  'plt_effects': pd.Series(self.plt_effects.detach(), index=self.plt_labels),
                  'js_effects': pd.Series(self.js_effects.detach(), index=self.js_labels)
                  }
        return(params)
    

class FullMultilinearModel(BaseMultilinearModel):
    def init_params(self):
        self.log_alpha = Parameter(torch.zeros(size=(1,)))
        self.max_value = Parameter(torch.ones(size=(1,)))
        
        self.log_effects = Parameter(torch.zeros(size=(9,)))
        self.het_logit = Parameter(torch.zeros(size=(2,)))
        self.seasons_theta = Parameter(torch.zeros(size=(4, 2)))
    
    @property
    def het(self):
        p = torch.exp(self.het_logit) / (1 + torch.exp(self.het_logit))
        # het = torch.zeros((2,))
        # het[0] = p / 2
        # het[1] = p
        return(p)
    
    @property
    def j2_effects(self):
        j2_effect = torch.exp(self.log_effects[2])
        j2_effects = torch.zeros(size=(2,))
        j2_effects[0] = self.het[0] * j2_effect
        j2_effects[1] = j2_effect
        return(j2_effects)
    
    @property
    def ej2_effects(self):
        ej2s = torch.exp(self.log_effects[3:])
        ej2_effects = torch.zeros(size=(12,))
        ej2_effects[:6] = self.het[1] * ej2s
        ej2_effects[6:] = ej2s
        return(ej2_effects)
    
    @property
    def plt3_effects(self):
        plt3 = torch.exp(self.log_effects[0])
        plt3_effects = torch.zeros(size=(2,))
        plt3_effects[0] = self.het[0] * plt3
        plt3_effects[1] = plt3
        return(plt3_effects)
    
    @property
    def plt7_effects(self):
        plt7 = torch.exp(self.log_effects[1])
        plt7_effects = torch.zeros(size=(2,))
        plt7_effects[0] = self.het[1] * plt7
        plt7_effects[1] = plt7
        return(plt7_effects)
    
    @property
    def ej2_slopes(self):
        ej2_slopes = torch.zeros(size=(12,))
        ej2_slopes[:6] = self.j2_effects[0] #self.het[0] * self.ej2_slope
        ej2_slopes[6:] = self.j2_effects[1] #self.ej2_slope
        return(ej2_slopes)

    @property
    def plt7_slopes(self):
        plt7_slopes = torch.zeros(size=(2,))
        plt7_slopes[0] = self.plt3_effects[0] #self.het[0] * self.plt7_slope
        plt7_slopes[1] = self.plt3_effects[1] #self.plt7_slope
        return(plt7_slopes)
    
    def calc_j2ej2_effect(self, X_j2, X_ej2):
        j2 = X_j2 @ self.j2_effects
        ej2 = X_ej2 @ self.ej2_effects + (X_ej2 @ self.ej2_slopes) * j2
        return(j2 + ej2)
    
    def calc_plt3plt7_effect(self, X_plt3, X_plt7):
        plt3 = X_plt3 @ self.plt3_effects
        plt7 = (X_plt7 @ self.plt7_slopes) * plt3
        plt7 += X_plt7 @ self.plt7_effects
        return(plt3 + plt7)
    
    def predict(self, X_plt3, X_plt7, X_j2, X_ej2, X_seasons):
        baseline = X_seasons @ self.seasons_theta[:, 1]

        plts = baseline + self.calc_plt3plt7_effect(X_plt3, X_plt7)
        plts_min = plts.min()
        plts_max = self.max_value #plts.max()
        p = (plts_max - plts) / (plts_max - plts_min)
        js = p * self.calc_j2ej2_effect(X_j2, X_ej2)
        mu = torch.exp(plts + js + X_seasons @ self.seasons_theta[:, 0])
        return(mu)
    
    def fit(self, X_plt3, X_plt7, X_j2, X_ej2, X_seasons, counts, exposure, n_iter=1000, lr=0.1):
        self.init_params()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history = []
        for i in tqdm(range(n_iter)):
            optimizer.zero_grad()
            yhat = self.predict(X_plt3, X_plt7, X_j2, X_ej2, X_seasons)
            loss = -self.loglikelihood(yhat, counts, exposure)
            loss.backward()
            history.append(loss.detach().item())
            optimizer.step()
        self.history = history
    
    def get_params(self):
        params = {
                  'seasons_theta': pd.DataFrame(self.seasons_theta.detach(),
                                                columns=['baseline_effect', 'plt_effect']),
                  
                  'alpha': np.exp(self.log_alpha.detach().numpy()[0]),
                  'max_value': self.max_value.detach().numpy()[0],

                  'effects': torch.exp(self.log_effects.detach()),
                  'het': self.het.detach(),
                  }
        return(params)
    
    
class SimplifiedMultilinearModel(BaseMultilinearModel):
    def init_params(self):
        self.log_alpha = Parameter(torch.zeros(size=(1,)))
        
        self.theta = Parameter(torch.zeros(size=(4,)))
        self.log_effects = Parameter(torch.zeros(size=(9,))-5)
        self.a = Parameter(torch.zeros(size=(2,)))
        self.seasons_theta = Parameter(torch.zeros(size=(4, 2)))
        self.b = Parameter(torch.tensor([0, 1.]))
    
    @property
    def effects(self):
        return(torch.exp(self.log_effects))
    
    def calc_j2ej2_effect(self, X_j2, X_ej2, X_seasons=None):
        if X_seasons is None:
            theta0 = self.a[1]
        else:
            theta0 = X_seasons @ self.seasons_theta[:, 1]

        f = torch.exp(theta0 + X_j2 * self.effects[2] + X_ej2 @ self.effects[3:])
        return(f)
    
    def calc_plt3plt7_effect(self, X_plt3, X_plt7, X_seasons=None):
        if X_seasons is None:
            theta0 = self.a[0]
        else:
            theta0 = X_seasons @ self.seasons_theta[:, 0]

        f = torch.exp(theta0 + X_plt3 * self.effects[0] + X_plt7 * self.effects[1])
        return(f)
    
    def predict(self, X_plt3, X_plt7, X_j2, X_ej2, X_seasons):
        plts = self.calc_plt3plt7_effect(X_plt3, X_plt7, X_seasons=X_seasons)
        js = self.calc_j2ej2_effect(X_j2, X_ej2, X_seasons=X_seasons)
        f = self.theta[0] + torch.exp(self.theta[1]) * plts + torch.exp(self.theta[2]) * js - self.theta[3] * plts * js
        mu = torch.exp(self.b[0] + self.b[1]  * torch.exp(f) / (1 + torch.exp(f)))
        # mu = torch.exp(f)
        return(mu)
    
    def fit(self, X_plt3, X_plt7, X_j2, X_ej2, X_seasons, counts, exposure, n_iter=1000, lr=0.1):
        self.init_params()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history = []
        for i in tqdm(range(n_iter)):
            # print(self.get_params())
            # input()
            optimizer.zero_grad()
            yhat = self.predict(X_plt3, X_plt7, X_j2, X_ej2, X_seasons)
            loss = -self.loglikelihood(yhat, counts, exposure)
            loss.backward()
            history.append(loss.detach().item())
            optimizer.step()
        self.history = history
    
    def get_params(self):
        params = {
                  'seasons_theta': pd.DataFrame(self.seasons_theta.detach(),
                                                columns=['plt_effect', 'js_effect']),
                  'alpha': np.exp(self.log_alpha.detach().numpy()[0]),
                  'theta': self.theta.detach().numpy(),
                  'effects': self.effects.detach(),
                  'baselines': self.a.detach(),
                  'b': self.b.detach(),
                  }
        return(params)


def get_js_encoding(plant_data):
    j2 = pd.DataFrame({'J2H': plant_data['J2'] != 'W',
                       'J2M': plant_data['J2'] == 'M'}).astype(float)
    ej2 = {}
    for variant in EJ2_SERIES:
        ej2['EJ2H{}'.format(variant)] = (plant_data['EJ2'] == 'H{}'.format(variant)) | (plant_data['EJ2'] == 'M{}'.format(variant))
        ej2['EJ2M{}'.format(variant)] = plant_data['EJ2'] == 'M{}'.format(variant)
    ej2 = pd.DataFrame(ej2).astype(float)

    j2_ej2 = {}
    for col1 in j2.columns:
        for col2 in ej2.columns:
            j2_ej2['{}_{}'.format(col1, col2)] = j2[col1] * ej2[col2]
    j2_ej2 = pd.DataFrame(j2_ej2)
    X_js = pd.concat([j2, ej2, j2_ej2], axis=1)
    return(X_js)


def get_js_encoding_full_multilinear(plant_data):
    j2 = pd.DataFrame({'J2H': plant_data['J2'] == 'H',
                       'J2M': plant_data['J2'] == 'M'}).astype(float)
    
    ej2 = {}
    for variant in EJ2_SERIES:
        ej2['EJ2H{}'.format(variant)] = plant_data['EJ2'] == 'H{}'.format(variant)
    
    for variant in EJ2_SERIES:
        ej2['EJ2M{}'.format(variant)] = plant_data['EJ2'] == 'M{}'.format(variant)
    ej2 = pd.DataFrame(ej2).astype(float)
    return(j2, ej2)


def get_dosage_encoding(plant_data):
    dosage_encoding = {'W': 0, 'H': 1, 'M': 2.}
    
    plt3 = [dosage_encoding[x] for x in plant_data['PLT3']]
    plt7 = [dosage_encoding[x] for x in plant_data['PLT7']]
    j2 = [dosage_encoding[x] for x in plant_data['J2']]
    
    ej2 = {}
    for variant in EJ2_SERIES:
        dosage_encoding = {'H{}'.format(variant): 1, 'M{}'.format(variant): 2.}
        ej2['EJ2({})'.format(variant)] = [dosage_encoding.get(x, 0) for x in plant_data['EJ2']]
    ej2 = pd.DataFrame(ej2).values
    return(plt3, plt7, j2, ej2)

    
if __name__ == '__main__':
    # X = np.array([[0, 0, 0, 0],
    #               [1, 0, 0, 0],
    #               [1, 1, 0, 0],
                  
    #               [0, 0, 1, 0],
    #               [1, 0, 1, 0],
    #               [1, 1, 1, 0],
                  
    #               [0, 0, 1, 1],
    #               [1, 0, 1, 1],
    #               [1, 1, 1, 1],])
    # s = 1.5
    # b1 = 0.5
    # b2 = 0.8
    
    # f = X[:, 0] * b1
    # f+= X[:, 1] * b1 * (1 + s)
    # f+= X[:, 2] * b2 * (1 + s)
    # f+= X[:, 3] * b2 * (1 + s)
    # print(f.reshape((3, 3)))
    # exit()
    
    
    # Load raw data
    plant_data = pd.read_csv('data/plant_data.csv', index_col=0)
    plant_data['plts'] = ['{}{}'.format(plt3, plt7)
                          for plt3, plt7 in zip(plant_data['PLT3'], plant_data['PLT7'])]
    plant_data['js'] = ['{}{}'.format(j2, ej2)
                        for j2, ej2 in zip(plant_data['J2'], plant_data['EJ2'])]
    plts = plant_data['plts'].unique()
    # js = plant_data['js'].unique()
    seasons = plant_data['Season'].unique()
    X_js = get_js_encoding(plant_data)
    X_j2, X_ej2 = get_js_encoding_full_multilinear(plant_data)
    X_plts = pd.DataFrame({x: (plant_data['plts'] == x) for x in plts}).astype(float).drop('WW', axis=1)
    X_plt3 = pd.DataFrame({'plt3' + x: (plant_data['PLT3'] == x) for x in 'HM'}).astype(float)
    X_plt7 = pd.DataFrame({'plt7' + x: (plant_data['PLT7'] == x) for x in 'HM'}).astype(float)

    js = X_js.columns
    j2 = X_j2.columns
    ej2 = X_ej2.columns
    plt3 = X_plt3.columns
    plt7 = X_plt7.columns

    X_js_pred = X_js.join(plant_data[['js']]).drop_duplicates().set_index('js')
    X_plts_pred = X_plts.join(plant_data[['plts']]).drop_duplicates().set_index('plts')
    X_j2ej2_pred = pd.concat([X_j2, X_ej2], axis=1).join(plant_data[['js']]).drop_duplicates().set_index('js')
    X_j2_pred = torch.Tensor(X_j2ej2_pred[j2].values)
    X_ej2_pred = torch.Tensor(X_j2ej2_pred[ej2].values)
    X_plt3plt7_pred = pd.concat([X_plt3, X_plt7], axis=1).join(plant_data[['plts']]).drop_duplicates().set_index('plts')
    X_plt3_pred = torch.Tensor(X_plt3plt7_pred[plt3].values)
    X_plt7_pred = torch.Tensor(X_plt3plt7_pred[plt7].values)
    
    X_plts = torch.Tensor(X_plts.values)
    X_plt3 = torch.Tensor(X_plt3.values)
    X_plt7 = torch.Tensor(X_plt7.values)
    X_js = torch.Tensor(X_js.values)
    X_j2 = torch.Tensor(X_j2.values)
    X_ej2 = torch.Tensor(X_ej2.values)
    X_seasons = torch.Tensor(pd.DataFrame({x: (plant_data['Season'] == x) for x in seasons}).astype(float).values)
    y = torch.Tensor(plant_data['branches'].values)
    exposure = torch.Tensor(plant_data['influorescences'].values)

    model = MultilinearModel(plts[1:], js, seasons)
    model.fit(X_plts, X_js, X_seasons, y, exposure, n_iter=1000, lr=0.01)
    mu = model.predict(X_plts, X_js, X_seasons).detach()
    model.summary(pred=mu, obs=plant_data['obs_mean'])
    params = model.get_params()
    plant_data['pred'] = np.log(mu)
    plant_data.to_csv('results/data.predictions.multilinear.csv')


    X_plt3, X_plt7, X_j2, X_ej2 = (torch.Tensor(x) for x in get_dosage_encoding(plant_data))
    # model = FullMultilinearModel(plts[1:], EJ2_SERIES, seasons)
    model = SimplifiedMultilinearModel()
    model.fit(X_plt3, X_plt7, X_j2, X_ej2, X_seasons, y, exposure, n_iter=5000, lr=0.01)
    
    mu = model.predict(X_plt3, X_plt7, X_j2, X_ej2, X_seasons).detach()
    model.summary(pred=mu, obs=plant_data['obs_mean'])
    plant_data['pred'] = np.log(mu)
    plant_data.to_csv('results/data.predictions.multilinear2.csv')
    
    X_js_pred_2 = pd.concat([pd.DataFrame([X_j2.numpy()]).T, pd.DataFrame(X_ej2)], axis=1).join(plant_data[['js']]).drop_duplicates().set_index('js')
    X_plts_pred_2 = pd.DataFrame([X_plt3.numpy(), X_plt7.numpy()]).T.join(plant_data[['plts']]).drop_duplicates().set_index('plts')
    
    X_j2_pred = torch.Tensor(X_js_pred_2.iloc[:, 0].values)
    X_ej2_pred = torch.Tensor(X_js_pred_2.iloc[:, 1:].values)
    X_plt3_pred = torch.Tensor(X_plts_pred_2.iloc[:, 0].values)
    X_plt7_pred = torch.Tensor(X_plts_pred_2.iloc[:, 1].values)
    
    X_seasons_pred = torch.ones((X_plt7_pred.shape[0], 4)) * 0.25
    plts_pred = pd.DataFrame({'log_branches_1': X_plts_pred @ params['plt_effects'],
                              'log_branches_2': model.calc_plt3plt7_effect(X_plt3_pred, X_plt7_pred, X_seasons=X_seasons_pred).detach()
                            })
    plts_pred.to_csv('results/plts_predictions.multilinear.csv')
    
    X_seasons_pred = torch.ones((X_j2_pred.shape[0], 4)) * 0.25
    js_pred = pd.DataFrame({'log_branches_1': X_js_pred @ params['js_effects'],
                            'log_branches_2': model.calc_j2ej2_effect(X_j2_pred, X_ej2_pred, X_seasons=X_seasons_pred).detach()})
    js_pred.to_csv('results/js_predictions.multilinear.csv')