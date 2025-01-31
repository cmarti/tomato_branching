#!/usr/bin/env python
import pandas as pd
import numpy as np

from itertools import combinations
from scipy.linalg import orth
from statsmodels.iolib.smpickle import load_pickle

from scripts.settings import G_SITES_SIMPLE, EJ2_SERIES_LABELS, SEASONS
from scripts.utils import (get_full_basis, get_params_df, get_env_plant_data_pred,
                           get_masking_plant_data, get_phi_to_ypred,
                           get_plts_masking_plant_data,
                           get_masking_plts_plant_data, define_js_synergy_contrasts,
                           define_js_masking_contrasts, define_plts_masking_contrasts,
                           define_plts_synergy_contrasts, define_j2_ej2_synergy_contrasts)




if __name__ == '__main__':
    model_label = 'dxe'
    sites = G_SITES_SIMPLE + ['EJ2']

    # Load raw data
    print('Loading data from file')
    plant_data = pd.read_csv('data/plant_data.csv', index_col=0)
    gt_data = pd.read_csv('data/genotype_means.csv', index_col=0)
    print(gt_data)
    exit()
    
    print('Loading model {} fit'.format(model_label))
    model = load_pickle('results/model.{}.pkl'.format(model_label))
    df = get_params_df(model)
    n_params = df.shape[0]
    param_names = df.index.values
    df.to_csv('results/model_parameters.{}.csv'.format(model_label))

    def get_basis(X):
        return(get_full_basis(X, rm_zeros=False)[param_names])

    # Prepare data
    print('Preparing data for model fitting')
    X = get_basis(plant_data)
    gt_X = get_basis(gt_data)

    print('Testing average pairwise coefficients for EJ2')
    x = []
    for site in G_SITES_SIMPLE:
        v = pd.Series(np.zeros(gt_X.shape[1]), index=gt_X.columns)
        names = ['{}_{}'.format(site, variant) for variant in EJ2_SERIES_LABELS]
        v.loc[names] = 1/6.
        x.append(v)
    x = pd.DataFrame(x, index=G_SITES_SIMPLE)
    results = model.t_test(x).summary_frame()
    results.index = x.index
    results.to_csv('results/test_ej2_pairwise.{}.csv'.format(model_label))

    print('Testing PLT3/PLT7 synergy')
    C = define_plts_synergy_contrasts(get_basis)
    results = model.t_test(C).summary_frame()
    results.index = C.index
    print(results)

    print('Testing J2/EJ2 synergy')
    C = define_j2_ej2_synergy_contrasts(get_basis)
    results = model.t_test(C).summary_frame()
    results.index = C.index
    print(results)
    
    print('Testing J2 effects across EJ2 backgrounds')
    C = define_js_synergy_contrasts(get_basis)
    result = model.t_test(C).summary_frame()
    result.index = C.index
    print(result)
    result.to_csv('results/test_j2_effects_across_ej2_backgrounds.{}.csv'.format(model_label))
    exit()
    

    print('Testing PLT3 and PLT7 effects across Js backgrounds')
    C = define_plts_masking_contrasts(get_basis)
    result = model.t_test(C).summary_frame()
    result.index = C.index
    print(result)
    result.to_csv('results/test_plts_effects_across_backgrounds.{}.csv'.format(model_label))

    print('Testing J2 and EJ2 effects across PLT backgrounds')
    C = define_js_masking_contrasts(get_basis)
    result = model.t_test(C).summary_frame()
    result.index = C.index
    print(result)
    result.to_csv('results/test_j2_effects_across_backgrounds.{}.csv'.format(model_label))

    print('Testing for epistatic masking of J2/EJ2 by PLT3 and PLT7')
    d1, d2 = get_masking_plant_data()
    x1 = get_full_basis(d1, rm_zeros=False)[param_names]
    x2 = get_full_basis(d2, rm_zeros=False)[param_names]
    x = (x2 - x1)
    x['label'] = ['_'.join(a) for a in d2[['background', 'site']].values]
    x = x.groupby('label').mean()
    x_ej2_mean = x.copy()
    x_ej2_mean['idx'] = ['/'.join([a.split('(')[0] for a in b.split('/')])
                           for b in x_ej2_mean.index]
    x_ej2_mean = x_ej2_mean.groupby('idx').mean()
    x = pd.concat([x, x_ej2_mean]).drop_duplicates()
    masking = model.t_test(x).summary_frame()
    masking.index = x.index
    masking.to_csv('results/test_masking.{}.csv'.format(model_label))
    
    print('Testing for epistatic masking of PLT3/PLT7 by J2 and EJ2')
    d1, d2 = get_masking_plts_plant_data()
    x1 = get_full_basis(d1, rm_zeros=False)[param_names]
    x2 = get_full_basis(d2, rm_zeros=False)[param_names]
    x = (x2 - x1)
    x['label'] = ['_'.join(a) for a in d2[['background', 'site']].values]
    x = x.groupby('label').mean()
    x_ej2_mean = x.copy()
    x_ej2_mean['idx'] = ['/'.join([a.split('(')[0] for a in b.split('_')[0].split('/')]) + '_{}'.format(b.split('_')[-1])
                           for b in x_ej2_mean.index]
    x_ej2_mean = x_ej2_mean.groupby('idx').mean()
    x = pd.concat([x, x_ej2_mean]).drop_duplicates()
    masking = model.t_test(x).summary_frame()
    masking.index = x.index
    masking.to_csv('results/test_masking_plts.{}.csv'.format(model_label))
    
    print('Making predictions in the data')
    pred = model.get_prediction(X).summary_frame()
    phi = model.t_test(X).summary_frame()[['coef', 'std err']]
    phi.index = pred.index
    plant_data.join(pred).join(phi).to_csv('results/model_predictions.{}.csv'.format(model_label))
    
    print('Making predictions in the data at genotype level')
    pred = model.get_prediction(gt_X).summary_frame()
    phi = model.t_test(gt_X).summary_frame()[['coef', 'std err']]
    phi.index = pred.index
    gt_data.join(pred).join(phi).to_csv('results/model_predictions.genotypes.{}.csv'.format(model_label))

    print('Obtaining estimates for all possible genotypes and seasons')
    env_plant_data = get_env_plant_data_pred()
    env_plant_data['gt'] = [''.join(x) for x in env_plant_data[G_SITES_SIMPLE + ['EJ2']].values]
    basis = get_full_basis(env_plant_data)[param_names]
    phi = model.t_test(basis).summary_frame()
    phi.index = env_plant_data.index
    env_plant_data = env_plant_data.join(phi)
    env_plant_data.to_csv('results/genotypes_season_estimates.{}.csv'.format(model_label))

    print('Obtaining estimates for all possible genotypes across seasons')
    basis['gt'] = [''.join(x) for x in env_plant_data[G_SITES_SIMPLE + ['EJ2']].values]
    basis['Season'] = env_plant_data['Season']
    mean_basis = basis.groupby('gt')[param_names].mean()
    gt_estimates = model.t_test(mean_basis).summary_frame().set_index(mean_basis.index).drop(['z', 'P>|z|'], axis=1)
    gt_estimates.to_csv('results/genotypes_estimates.{}.csv'.format(model_label))
    exit()
    
    print('Testing for seasonal effects across all possible genotypes')
    contrasts = []
    for season, season_basis in basis.groupby('Season'):
        season_basis = season_basis.set_index('gt')[param_names].loc[mean_basis.index.values, :]
        contrasts.append(season_basis - mean_basis)
    contrasts = pd.concat(contrasts, axis=0)
    results = model.t_test(contrasts).summary_frame()
    results['gt'] = contrasts.index.values
    results['Season'] = env_plant_data['Season'].values
    results.to_csv('results/season_effects_contrasts.{}.csv'.format(model_label))

    # Make predictions through the whole latent space
    print('Producing phi-y mapping')
    phi_pred = get_phi_to_ypred(model.alpha)
    phi_pred.to_csv('results/predictive_distribution.{}.csv'.format(model_label))

    print('Testing masking of PLT3-PLT7 interaction by J2/EJ2 with PLT3-het')
    plant_data_pred, contrasts = get_plts_masking_plant_data(site1='PLT3', site2='PLT7')
    X = get_full_basis(plant_data_pred, rm_zeros=False)[param_names]
    x = contrasts.T @ X
    plant_data_pred, contrasts = get_plts_masking_plant_data(site1='PLT7', site2='PLT3')
    X = get_full_basis(plant_data_pred, rm_zeros=False)[param_names]
    x = pd.concat([x, contrasts.T @ X], axis=0)
    
    results = model.t_test(x).summary_frame()
    results.index = x.index
    print(results)
    results.to_csv('results/plt_synergy_masking.{}.csv'.format(model_label))

    # Make contrasts of 3 way coefficients in bulk for EJ2 alleles
    print('Comparing 3 way epistatic coefficients in average for EJ2 allelic series')
    contrasts = {}
    x = pd.Series(np.zeros(gt_X.shape[1]), index=gt_X.columns)
    x.loc['gPLT3_gPLT7_gJ2'] = 1.
    contrasts['PLT3_PLT7_J2'] = x
    for v1, v2 in combinations(G_SITES_SIMPLE, 2):
        x = pd.Series(np.zeros(gt_X.shape[1]), index=gt_X.columns)
        names = ['g{}_g{}_g{}'.format(v1, v2, target) for target in EJ2_SERIES_LABELS]
        x.loc[names] = 1. / len(EJ2_SERIES_LABELS)
        contrasts['{}_{}_EJ2'.format(v1, v2)] = x

    contrasts = pd.DataFrame(contrasts)
    contrasts_test = model.t_test(contrasts.T).summary_frame()
    contrasts_test.index = contrasts.columns
    print(contrasts_test)
    contrasts_test.to_csv('results/3way_average_contrasts.{}.csv'.format(model_label))
    
    # # Make contrasts of epistatic coefficients
    # print('Comparing epistatic coefficients for EJ2 across the allelic series')
    # contrasts = {}
    # labels = []
    # for v1, v2 in combinations(EJ2_SERIES_LABELS, 2):
    #     for background in G_SITES_SIMPLE:
    #         c = pd.Series(np.zeros(n_params), index=param_names)
    #         coeff1, coeff2 = '{}_{}'.format(background, v1), '{}_{}'.format(background, v2)
    #         c.loc[coeff1] = -1
    #         c.loc[coeff2] = 1
    #         contrasts['{}_vs_{}'.format(coeff2, coeff1)] = c
    #         labels.append({'a1': v1, 'a2': v2, 'background': background})
    # contrasts = pd.DataFrame(contrasts)
    # labels = pd.DataFrame(labels, index=contrasts.columns)
    # contrasts_test = model.t_test(contrasts.T).summary_frame()
    # contrasts_test.index = contrasts.columns
    # contrasts_test = contrasts_test.join(labels)