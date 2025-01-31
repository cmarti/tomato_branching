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
    cols = ['mutation', 'background', 'coef', 'Conf. Int. Low', 'Conf. Int. Upp.']
    
    model_label = 'dxe'
    js_masking = pd.read_csv('results/test_masking.{}.csv'.format(model_label), index_col=0)
    plts_masking = pd.read_csv('results/test_masking_plts.{}.csv'.format(model_label), index_col=0)
    masking = pd.concat([js_masking, plts_masking])
    
    masking = masking.loc[['('not in x for x in masking.index], :]
    masking['mutation'] = [x.split('_')[-1] for x in masking.index]
    masking['background'] = [x.split('_')[0].replace('/', ' ').replace('h', '/+') for x in masking.index]
    masking = masking[cols]
    masking[cols[2:]] = np.exp(masking[cols[2:]])
    masking.columns = ['mutation', 'background', 'Fold change', 'Conf. Int. Low', 'Conf. Int. Upp.']
    print(masking)
    masking.to_csv('results/mut_effs.{}.csv'.format(model_label), index=False)
    
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