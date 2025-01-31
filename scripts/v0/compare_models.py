#!/usr/bin/env python
import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.iolib.smpickle import load_pickle
from scipy.stats import chi2


def lr_test(larger, restricted=None):
    if restricted is None:
        return(None)
    statistic = -2 * (restricted.llf_scaled() - larger.llf_scaled())
    p_val = chi2.sf(statistic, larger.df_model - restricted.df_model)
    return(p_val)


if __name__ == '__main__':
    # Load raw data
    subsets = ['constant', 'additive', 'dominance', 'environment',
               'gxg', 'gxd', 'dxd',
               'gxe', 'dxe',
            #    'g3', 
               'd3',
            #    'pairwise', 'pw_dom', 
            #    'threeway_noplt7', 'threeway_dom',
               ]

    print('Loading model fits')    
    model_comparison = []
    prev_model = None

    for label in subsets:
        model = load_pickle('results/model.{}.pkl'.format(label))
        ll, df = model.llf_scaled(), model.df_model
        deviance_perc = 100 * (1 - model.deviance / model.null_deviance)
        p_val = lr_test(model, prev_model)
        prev_model = model

        record = {'label': label, 'll': ll,  'df': df, 'pvalue': p_val,
                  'deviance_perc': deviance_perc}
        model_comparison.append(record)
        print('Model {}: logL = {:.2f} \t df = {} \t % deviance explained = {:.2f} \t p-value = {}'.format(label, ll, df, deviance_perc, p_val))

    model_comparison = pd.DataFrame(model_comparison)
    model_comparison.to_csv('results/model_comparison.csv')
