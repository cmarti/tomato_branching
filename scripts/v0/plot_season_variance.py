#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scripts.settings import G_SITES_SIMPLE, ALLELES, EJ2_SERIES_LABELS
from scripts.utils import get_full_basis, fit_model, get_plant_data_pred, get_phi_to_ypred, get_params_df
                        
                        
if __name__ == '__main__':
    # Load estimates
    estimates = pd.read_csv('results/genotypes_season_estimates.csv', index_col=0)
    season_effects = pd.read_csv('results/env_effects_contrasts.csv', index_col=0)

    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))
    bins = np.linspace(-4, 5, 50)
    sns.histplot(estimates['coef'], bins=bins, color='purple', lw=1, edgecolor='black', stat='percent', alpha=0.5, label='Phenotypic distribution', ax=axes)
    sns.histplot(season_effects['coef'], bins=bins, color='orange', lw=1, edgecolor='black', stat='percent', alpha=0.5, label='Environmental effects', ax=axes)
    axes.set(xlabel='$\log(time)$', ylabel='% of genotypes', ylim=(0, 30))
    axes.legend(loc=1, fontsize=8)
    axes.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig('plots/phenotypic_distributions.png', dpi=300)
    