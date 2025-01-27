#!/usr/bin/env python
import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy.optimize import minimize
from scripts.utils import (get_constant_basis, get_additive_basis, get_dominance_basis, get_env_basis,
                           get_gxg_basis, get_gxd_basis, get_dxd_basis, get_x_basis,
                           get_3way_basis, get_full_basis)

def get_nb_model(y, X, exposure):
    
    def loss(alpha):
            model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha), exposure=exposure).fit()
            return(-model.llf_scaled())
        
    res = minimize(loss, x0=0.3, method='nelder-mead',
                    bounds=[(0, None)], tol=1e-2)
    alpha = res.x[0]
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha), exposure=exposure).fit()
    return(model)




if __name__ == '__main__':
    # Load raw data
    print('Loading processed data for model fitting')
    plant_data = pd.read_csv('data/plant_data.csv', index_col=0)
    y = plant_data['branches']
    exposure = plant_data['influorescences'].values

    print('Constructing basis for regression')
    c = get_constant_basis(plant_data)
    g = get_additive_basis(plant_data)
    e = get_env_basis(plant_data)
    X = pd.concat([c, g, e], axis=1)
    X_ext = X * np.expand_dims(exposure, 1)
    data = {'obs': y / exposure, 'n_plants': exposure}

    print('Fitting OLS')
    results = sm.OLS(y, X_ext).fit()
    print('log-likelihood = {}'.format(results.llf))
    data['ols_linear'] = results.predict(X_ext) / exposure
    results.save('results/model.lsq_linear.pkl')
    
    print('Fitting OLS log')
    results = sm.OLS(np.log(y/exposure + 1), X).fit()
    print('log-likelihood = {}'.format(results.llf))
    data['ols_log'] = (np.exp(results.predict(X)) - 1)
    results.save('results/model.lsq_log.pkl')
    
    print('Fitting Poisson linear')
    results = sm.GLM(y, X_ext, family=sm.families.Poisson(link=sm.families.links.Identity())).fit()
    print('log-likelihood = {}'.format(results.llf_scaled()))
    data['poisson_linear'] = results.predict(X_ext)/ exposure
    results.save('results/model.poisson_linear.pkl')
    
    print('Fitting Poisson log')
    results = sm.Poisson(y, X, exposure=exposure).fit()
    print('log-likelihood = {}'.format(results.llf))
    data['poisson_log'] = results.predict(X)
    results.save('results/model.poisson_log.pkl')
    
    print('Fitting NB log')
    results = get_nb_model(y, X, exposure)
    print('log-likelihood = {}'.format(results.llf_scaled()))
    data['nb_log'] = results.predict(X)
    results.save('results/model.nb_log.pkl')

    print('Storing models predictions')
    data = pd.DataFrame(data)
    data.to_csv('results/basic_models_predictions.csv')
        
    
