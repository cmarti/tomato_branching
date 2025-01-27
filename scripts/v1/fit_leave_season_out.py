#!/usr/bin/env python
import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy.optimize import minimize
from statsmodels.iolib.smpickle import load_pickle

from scripts.settings import SEASONS
from scripts.utils import get_full_basis


def get_svd_projection(X_train, threshold=1e-4):
    _, S, Vh = np.linalg.svd(X_train, full_matrices=False)
    idx = S > threshold
    S_inv = 1. / S[idx]
    D_inv = np.diag(S_inv)
    P = Vh.T[:, idx] @ D_inv
    return(P)
                  

if __name__ == '__main__':
    model_label = 'pairwise'
    config = {'pairwise': (True, False),
              'threeway': (False, True)}
    dominance, third_order = config[model_label]
    
    # Load raw data
    print('Loading data from file')
    plant_data = pd.read_csv('data/plant_data.csv', index_col=0)
    gt_data = pd.read_csv('data/genotype_means.csv', index_col=0)
    
    # Prepare data
    print('Preparing data for model fitting')
    gt_X = get_full_basis(gt_data)
    gt_X_test = gt_X.join(gt_data, rsuffix='_r')
    
    # Run cross-validation on the different seasons
    print('Running leave one season out analysis')
    cv_data = []
    idx_cols = ['PLT3_r', 'PLT7_r', 'J2_r', 'EJ2']
    for season in SEASONS:
        print('\tLeaving out season: {}'.format(season))
        
        # Prepare training data
        train = plant_data.loc[plant_data['Season'] != season, :]
        X_train = get_full_basis(train, third_order=third_order, dominance=dominance)
        season_cols = X_train.columns
        y_train = train['branches'].values
        exposure_train = train['influorescences'].values
        
        # Fit model
        P = get_svd_projection(X_train)
        U_train = X_train @ P

        def loss(alpha):
            family = sm.families.NegativeBinomial(alpha=alpha)
            model = sm.GLM(y_train, U_train, family=family, exposure=exposure_train).fit()
            return(-model.llf_scaled())
        
        res = minimize(loss, x0=0.3, method='nelder-mead',
                       bounds=[(0, None)], tol=1e-2)
        alpha = res.x[0]
        family = sm.families.NegativeBinomial(alpha=alpha)
        model = sm.GLM(y_train, U_train, family=family, exposure=exposure_train).fit()

        X_test = gt_X_test.loc[gt_X_test['Season'] == season, :]
        X_test = X_test.groupby(idx_cols)[X_train.columns].mean()
        U_test = X_test @ P
        
        # # Make predictions
        df = model.get_prediction(U_test).summary_frame().reset_index()
        df['Season'] = season
        cv_data.append(df)
        
    cv_data = pd.concat(cv_data, axis=0).set_index(idx_cols + ['Season'])
    cv_gt_data = gt_data.join(cv_data, on=['PLT3', 'PLT7', 'J2', 'EJ2', 'Season'])
    cv_gt_data.to_csv('results/leave_season_out_results.{}.csv'.format(model_label))
