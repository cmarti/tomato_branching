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


class NBtwoDimensinalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def init_params(self, n_plts=7, n_js=38, n_seasons=4):
        self.log_alpha = Parameter(torch.zeros(size=(1,)))
        
        self.plt_effects_log = Parameter(torch.zeros(size=(n_plts,)))
        self.plt_effects_log_js = Parameter(torch.zeros(size=(n_plts,)))
        self.plt_seasons = Parameter(torch.zeros(size=(n_seasons,)))
        
        self.js_effects_log = Parameter(torch.zeros(size=(n_js,)))
        self.js_seasons = Parameter(torch.zeros(size=(n_seasons,)))
    
    @property
    def plt_effects(self):
        return(torch.exp(self.plt_effects_log))
    
    @property
    def plt_effects_js(self):
        return(torch.exp(self.plt_effects_log_js))
    
    @property
    def js_effects(self):
        return(torch.exp(self.js_effects_log))

    def predict(self, X_plts, X_js, X_seasons):
        plts = X_plts @ self.plt_effects + X_seasons @ self.plt_seasons
        js = X_js @ self.js_effects + X_plts @ self.plt_effects_js + X_seasons @ self.js_seasons
        mu = torch.exp(plts) + torch.exp(js)
        return(mu)
    
    def loglikelihood(self, yhat, counts, exposure):
        mean = yhat * exposure
        alpha = torch.exp(self.log_alpha)
        ll = NegativeBinomial(mean, alpha).logpmf(counts).sum()
        return(ll)
    
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
            optimizer.step()
        self.history = history
    
    def get_params(self, plts, js, seasons):
        params = {'alpha': np.exp(self.log_alpha.detach().numpy()[0]),
                  
                  'plt_seasons': pd.Series(self.plt_seasons.detach(), index=seasons),
                  'plt_effects': pd.Series(self.plt_effects.detach(), index=plts),
                  
                  'plt_effects_js': pd.Series(self.plt_effects_js.detach(), index=plts),
                  'js_seasons': pd.Series(self.js_seasons.detach(), index=seasons),
                  'js_effects': pd.Series(self.js_effects.detach(), index=js)
                  }
        return(params)


def get_js_encoding(plant_data):
    encoding = {'W': 0, 'H': 1, 'M': 2}
    X = pd.DataFrame({'J2': [encoding[x] for x in plant_data['J2']]})
    
    for variant in EJ2_SERIES:
        encoding = {'W': 0, 'H{}'.format(variant): 1, 'M{}'.format(variant): 2}
        X['EJ2({})'.format(variant)] = [encoding.get(x, 0) for x in plant_data['EJ2']]
    return(X)


def get_plt_encoding(plant_data):
    encoding = {'W': 0, 'H': 1, 'M': 2}
    X = pd.DataFrame({'PLT3': [encoding[x] for x in plant_data['PLT3']],
                      'PLT7': [encoding[x] for x in plant_data['PLT7']]})
    return(X)
    
if __name__ == '__main__':
    # Load raw data
    plant_data = pd.read_csv('data/plant_data.csv', index_col=0)
    
    X_js = get_js_encoding(plant_data)
    X_plts = get_plt_encoding(plant_data)
    print(X_plts)
    print(X_js)
    
    js = X_js.columns
    plts = X_plts.columns
    seasons = plant_data['Season'].unique()
    
    X_plts = torch.Tensor(X_plts.values)
    X_js = torch.Tensor(X_js.values)
    X_seasons = torch.Tensor(pd.DataFrame({x: (plant_data['Season'] == x) for x in seasons}).astype(float).values)
    y = torch.Tensor(plant_data['branches'].values)
    exposure = torch.Tensor(plant_data['influorescences'].values)

    print(X_plts.shape, X_js.shape, X_seasons.shape)
    model = NBtwoDimensinalModel()
    model.fit(X_plts, X_js, X_seasons, y, exposure, n_iter=10000, lr=0.01)
    
    params = model.get_params(plts, js, seasons)
    mu = model.predict(X_plts, X_js, X_seasons).detach()
    print(pearsonr(np.log(mu + 1), np.log(plant_data['obs_mean'] + 1)))
    plant_data['pred'] = np.log(mu)
    plant_data.to_csv('results/data.predictions.nb2d.csv')
    print(plant_data)
    print(model.history[-1])
    print(params)