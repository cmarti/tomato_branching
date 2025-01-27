#!/usr/bin/env python
import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy.stats import chi2
from scipy.optimize import minimize
from scripts.utils import (get_constant_basis, get_additive_basis, get_dominance_basis, get_env_basis,
                           get_gxg_basis, get_gxd_basis, get_dxd_basis, get_x_basis,
                           get_3way_basis, get_full_basis, define_plts_synergy_contrasts,
                           define_j2_ej2_synergy_contrasts)


def get_saturated_basis(plant_data):
    cols = ['PLT3', 'PLT7', 'J2', 'EJ2', 'Season']
    gts = [tuple(x) for x in plant_data[cols].values]
    unique_gts = set(gts)
    basis = pd.DataFrame({'_'.join(gt): [tuple(x) == gt for x in plant_data[cols].values]
                          for gt in unique_gts}).astype(float)
    return(basis)


def fit_model(y, X, exposure):
    
    def loss(alpha):
            model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha), exposure=exposure).fit()
            return(-model.llf_scaled())
        
    res = minimize(loss, x0=0.3, method='nelder-mead',
                    bounds=[(0, None)], tol=1e-2)
    alpha = res.x[0]
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha), exposure=exposure).fit()
    model.alpha = alpha
    return(model)


if __name__ == '__main__':
    # Load raw data
    print('Loading processed data for model fitting')
    plant_data = pd.read_csv('data/plant_data.csv', index_col=0)
    print(plant_data.shape)
    plant_data = plant_data.loc[(plant_data['J2'] == 'W')&(plant_data['EJ2'] == 'W'), :]
    y = plant_data['branches'].values
    exposure = plant_data['influorescences'].values

    print('Constructing basis')
    c = get_constant_basis(plant_data)
    g = get_additive_basis(plant_data)
    d = get_dominance_basis(plant_data)
    e = get_env_basis(plant_data)
    gxg = get_gxg_basis(g)
    gxd = get_gxd_basis(g, d)
    dxd = get_dxd_basis(d)
    gxe = get_x_basis(g, e)
    dxe = get_x_basis(d, e)
    basis = [c, g, d, e, gxg, gxd, dxd, gxe, dxe]
    basis = [c, g, d, e, gxg, gxd, dxd]

    X = pd.concat(basis, axis=1)
    cols = [x for x in X.columns if 'J2' not in x]
    X = X[cols]
    print(X.shape)

    model = fit_model(y, X, exposure)
    print(model.summary())
    def get_basis(X):
        return(get_full_basis(X, rm_zeros=False)[cols])
    C = define_plts_synergy_contrasts(get_basis)
    results = model.t_test(C).summary_frame()
    results.index = C.index
    print(results)

    plant_data = pd.read_csv('data/plant_data.csv', index_col=0)
    plant_data = plant_data.loc[(plant_data['PLT3'] == 'W')&(plant_data['PLT7'] == 'W'), :]
    y = plant_data['branches'].values
    exposure = plant_data['influorescences'].values

    print('Constructing basis')
    c = get_constant_basis(plant_data)
    g = get_additive_basis(plant_data)
    d = get_dominance_basis(plant_data)
    e = get_env_basis(plant_data)
    gxg = get_gxg_basis(g)
    gxd = get_gxd_basis(g, d)
    dxd = get_dxd_basis(d)
    gxe = get_x_basis(g, e)
    dxe = get_x_basis(d, e)
    basis = [c, g, d, e, gxg, gxd, dxd, gxe, dxe]
    basis = [c, g, d, e, gxg, gxd, dxd]
    X = pd.concat(basis, axis=1)
    cols = [x for x in X.columns if 'PLT' not in x]
    X = X[cols]
    print(X.shape)

    model = fit_model(y, X, exposure)        
    print(model.summary())
    def get_basis(X):
        return(get_full_basis(X, rm_zeros=False)[cols])
    C = define_j2_ej2_synergy_contrasts(get_basis)
    results = model.t_test(C).summary_frame()
    results.index = C.index
    print(results)


    
