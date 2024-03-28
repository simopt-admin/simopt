# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:52:50 2024

@author: Owner
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

default_results_filename = './default ASTRODF/default_ASTRODF_on_cnt_design.csv' # file location of csv results for problem design run on default version of ASTRODF
results_filename = '' # file location of csv results for cross design of solver design & problem design
default_data = pd.read_csv(default_results_filename)

# problem pedictors
c_1 = 1 # cost of material one
c_r = 1 # material cost ratio
r_r = 1 # recourse cost ration
s_r = 1 # salvage price ratio




# find average default performance for corresponding problem (final relative optimality gap)
n_dp = default_data['DesignPt#'].max()
data_summary = pd.DataFrame()
data_summary['Problem Design Point'] = range(n_dp)
data_summary['Default Average Optimality Gap'] = 0
for dp in range(n_dp):
    opt_gaps = default_data.loc[default_data['DesignPt#'] == dp, 'Final Relative Optimality Gap']
    avg_opt_gap = opt_gaps.mean()
    data_summary.at[dp, 'Default Average Optimality Gap'] = avg_opt_gap
print(data_summary)


