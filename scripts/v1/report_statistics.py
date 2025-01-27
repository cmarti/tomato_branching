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
                           get_masking_plts_plant_data)
                        

if __name__ == '__main__':
    model_label = 'dxe'
    params = pd.read_csv('results/model_parameters.{}.csv'.format(model_label), index_col=0)
    ej2_pairwise = pd.read_csv('results/test_ej2_pairwise.{}.csv'.format(model_label), index_col=0)
    js_masking = pd.read_csv('results/test_masking.{}.csv'.format(model_label), index_col=0)
    plts_interactions = pd.read_csv('results/plt_synergy_masking.{}.csv'.format(model_label), index_col=0)
    plts_masking = pd.read_csv('results/test_masking_plts.{}.csv'.format(model_label), index_col=0)

    print('Reporting relevant statistics for Pairwise model')
    row = plts_interactions.loc['hPLT3_PLT7_in_WT', :]
    print('\tplt7 has a {:.2f} fold higher response on the plt3+ background than in the wt (p={})'.format(np.exp(row['coef']), row['P>|z|']))
    row = plts_interactions.loc['hPLT7_PLT3_in_WT', :]
    print('\tplt3 has a {:.2f} fold higher response on the plt7+ background than in the wt (p={})'.format(np.exp(row['coef']), row['P>|z|']))
    row = ej2_pairwise.loc['J2', :]
    print('\tej2 variants have an average {:.2f} fold higher response on the j2 background than in the wt (p={})'.format(np.exp(row['coef']), row['P>|z|']))
    
    row = params.loc['PLT3_J2', :]
    print('\tj2 has a {:.2f} fold lower response on the plt3 background than in the wt (p={})'.format(np.exp(-row['coeff']), row['pval']))
    row = params.loc['PLT7_J2', :]
    print('\tj2 has a {:.2f} fold lower response on the plt7 background than in the wt (p={})'.format(np.exp(-row['coeff']), row['pval']))
    row = ej2_pairwise.loc['PLT3', :]
    print('\tej2 variants have an average {:.2f} fold lower response on the plt3 background than in the wt (p={})'.format(np.exp(-row['coef']), row['P>|z|']))
    row = ej2_pairwise.loc['PLT7', :]
    print('\tej2 variants have an average {:.2f} fold lower response on the plt7 background than in the wt (p={})'.format(np.exp(-row['coef']), row['P>|z|']))
    
    model_label = 'd3'
    js_masking = pd.read_csv('results/test_masking.{}.csv'.format(model_label), index_col=0)
    plts_masking = pd.read_csv('results/test_masking_plts.{}.csv'.format(model_label), index_col=0)
    plts_synergy_masking = pd.read_csv('results/plt_synergy_masking.{}.csv'.format(model_label), index_col=0)
    threeway = pd.read_csv('results/3way_average_contrasts.{}.csv'.format(model_label), index_col=0)
    
    print('Reporting relevant statistics for Threeway model')
    row = threeway.loc['PLT3_J2_EJ2', :]
    print('\tj2/ej2 synergy is in average {:.2f} fold lower on the plt3 background than in the wt (p={})'.format(np.exp(-row['coef']), row['P>|z|']))
    row = threeway.loc['PLT7_J2_EJ2', :]
    print('\tj2/ej2 synergy is in average {:.2f} fold lower on the plt7 background than in the wt (p={})'.format(np.exp(-row['coef']), row['P>|z|']))
    
    row = plts_synergy_masking.loc['hPLT3_PLT7_J2', :]
    print('\tplt3+/plt7 synergy is {:.2f} fold lower on the j2 background than in the wt (p={})'.format(np.exp(-row['coef']), row['P>|z|']))
    row = plts_synergy_masking.loc['hPLT3_PLT7_EJ2_mean', :]
    print('\tplt3+/plt7 synergy is {:.2f} fold lower on the ej2 background than in the wt (p={})'.format(np.exp(-row['coef']), row['P>|z|']))
    
    row = plts_synergy_masking.loc['hPLT7_PLT3_J2', :]
    print('\tplt3/plt7+ synergy is {:.2f} fold lower on the j2 background than in the wt (p={})'.format(np.exp(-row['coef']), row['P>|z|']))
    row = plts_synergy_masking.loc['hPLT7_PLT3_EJ2_mean', :]
    print('\tplt3/plt7+ synergy is {:.2f} fold lower on the ej2 background than in the wt (p={})'.format(np.exp(-row['coef']), row['P>|z|']))
    
    row = js_masking.loc['plt3h/plt7_J2', :]
    print('\tEstimated J2 effect of {:.2f} fold on the plt3+/plt7 background (p={})'.format(np.exp(row['coef']), row['P>|z|']))
    row = js_masking.loc['plt3/plt7h_J2', :]
    print('\tEstimated J2 effect of {:.2f} fold on the plt3/plt7+ background (p={})'.format(np.exp(row['coef']), row['P>|z|']))
    
    row = js_masking.loc['plt3h/plt7_EJ2', :]
    print('\tEstimated EJ2 average effect of {:.2f} fold on the plt3+/plt7 background (p={})'.format(np.exp(row['coef']), row['P>|z|']))
    row = js_masking.loc['plt3/plt7h_EJ2', :]
    print('\tEstimated EJ2 average effect of {:.2f} fold on the plt3/plt7+ background (p={})'.format(np.exp(row['coef']), row['P>|z|']))
    
    row = plts_masking.loc['ej2/j2_PLT3', :]
    print('\tEstimated PLT3 effect of {:.2f} fold on the j2/ej2 background (p={})'.format(np.exp(row['coef']), row['P>|z|']))
    row = plts_masking.loc['ej2/j2_PLT7', :]
    print('\tEstimated PLT7 effect of {:.2f} fold on the j2/ej2 background (p={})'.format(np.exp(row['coef']), row['P>|z|']))