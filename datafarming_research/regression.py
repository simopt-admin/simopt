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
results_filename = './first_ASTRODF_and_cnt_design/first_ASTRODF_cnt_design_results_edited.csv' # file location of csv results for cross design of solver design & problem design
problem_ratio_design_filename = 'problem_ratio_design.csv'
default_data = pd.read_csv(default_results_filename)
exp_data = pd.read_csv(results_filename)
ratio_design = pd.read_csv(problem_ratio_design_filename)

# problem pedictors
c_1 = 1 # cost of material one
c_r = 1 # material cost ratio
r_r = 1 # recourse cost ration
s_r = 1 # salvage price ratio




# find average default performance for corresponding problem (final relative optimality gap)
n_dp = default_data['DesignPt#'].max() + 1
data_summary = pd.DataFrame()
data_summary['Problem Design Point'] = range(n_dp)
data_summary['Default Average Optimality Gap'] = 0
for dp in range(n_dp):
    opt_gaps = default_data.loc[default_data['DesignPt#'] == dp, 'Final Relative Optimality Gap']
    avg_opt_gap = opt_gaps.mean()
    data_summary.at[dp, 'Default Average Optimality Gap'] = avg_opt_gap
    
    
    
#  find solver for each problem dp with best average performance    
sol_n_dp = exp_data['solver_dp'].max() + 1
prob_n_dp = exp_data['problem_dp'].max() + 1
data_summary['Average Best Optimality Gap'] = 0
data_summary['Best Solver'] = 0

for prob_dp in range(prob_n_dp): # cycle through problem versions
    prob_df = exp_data.loc[exp_data['problem_dp'] == prob_dp, ['problem_dp', 'solver_dp', 'Final Relative Optimality Gap']]
    
    # get average performance for each solver for problem version
    avg_gaps = pd.DataFrame()
    avg_gaps['solver_dp'] = 0
    avg_gaps['Average Optimality Gap'] = 0
    for sol_dp in range(sol_n_dp):
        sol_df = prob_df.loc[prob_df['solver_dp']== sol_dp, ['solver_dp','Final Relative Optimality Gap']]
        avg_gap = sol_df['Final Relative Optimality Gap'].mean()
        avg_gaps.at[sol_dp, 'solver_dp'] = sol_dp
        avg_gaps.at[sol_dp, 'Average Optimality Gap'] = avg_gap
    
    # find best average on problem
    best_gap = avg_gaps['Average Optimality Gap'].min()
    best_solver = avg_gaps.loc[avg_gaps['Average Optimality Gap'] == best_gap, 'solver_dp'].iloc[0]
    data_summary.at[prob_dp, 'Average Best Optimality Gap'] = best_gap
    data_summary.at[prob_dp, 'Best Solver'] = best_solver
        

    
    print(prob_df)

# add problem ratios (x values) to data summary
data_summary['c_m1'] = 0
data_summary['Mratio'] = 0
data_summary['Rratio'] = 0
data_summary['Sratio'] = 0

for index in range(prob_n_dp):
    c_m1 = ratio_design.at[index, 'c_m1']
    Mratio = ratio_design.at[index, 'Mratio']
    Rratio = ratio_design.at[index, 'Rratio']
    Sratio = ratio_design.at[index, 'Sratio']
    print(c_m1)
    data_summary.at[index, 'c_m1'] = c_m1
    data_summary.at[index, 'Mratio'] = Mratio
    data_summary.at[index, 'Rratio'] = Rratio
    data_summary.at[index, 'Sratio'] = Sratio
    
# find best - default for each problem
data_summary['Difference btwn Best and Default Avg Optimality Gap'] = 0
for index, row in data_summary.iterrows():
    diff = row['Average Best Optimality Gap'] - row['Default Average Optimality Gap']
    data_summary.at[index, 'Difference btwn Best and Default Avg Optimality Gap'] = diff
    

print(data_summary)

data_summary.to_csv('test_data_summary.csv', index = False)

