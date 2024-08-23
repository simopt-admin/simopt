# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:52:50 2024

@author: Owner
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP.models import (ModelSpec as MS,
summarize ,
poly)


import pandas as pd

#default_results_filename = './default ASTRODF/default_ASTRODF_on_cnt_design.csv' # file location of csv results for problem design run on default version of ASTRODF
results_filename = 'small_experiment_results.csv' # file location of csv results for cross design of solver design & problem design
problem_ratio_design_filename = 'problem_ratio_design.csv'
#default_data = pd.read_csv(default_results_filename)
exp_data = pd.read_csv(results_filename)
ratio_design = pd.read_csv(problem_ratio_design_filename)

solver_n_dp = 15 # temp
problem_n_dp = 16 #temp




# find average default performance for corresponding problem (final relative optimality gap)
#n_dp = default_data['DesignPt#'].max() + 1



data_summary = exp_data.loc[(exp_data['solver_dp']=='9') | (exp_data['solver_dp']=='14'), ['problem_dp', 'solver_dp', 'MacroRep#', 'Final Relative Optimality Gap']]
data_summary['default'] = 0

for p_dp in range(problem_n_dp):
    for macrorep in range(10):
        default = exp_data.loc[(exp_data['problem_dp']==p_dp) & (exp_data['MacroRep#']== macrorep)& (exp_data['solver_dp']=='default'), ['Final Relative Optimality Gap']].iloc[0,0]
        data_summary.loc[(data_summary['problem_dp']==p_dp) & (data_summary['MacroRep#']== macrorep), 'default'] = default                                                   
data_summary['default-solver'] = data_summary['default'] - data_summary['Final Relative Optimality Gap']        
print(data_summary)

# add problem ratios (x values) to data summary
data_summary['c_m1'] = 0
data_summary['Mratio'] = 0
data_summary['Rratio'] = 0
data_summary['Sratio'] = 0

for index in range(problem_n_dp):
    c_m1 = ratio_design.at[index, 'c_m1']
    Mratio = ratio_design.at[index, 'Mratio']
    Rratio = ratio_design.at[index, 'Rratio']
    Sratio = ratio_design.at[index, 'Sratio']
    data_summary.loc[data_summary['problem_dp']==index, ['c_m1']] = c_m1
    data_summary.loc[data_summary['problem_dp']==index, ['Mratio']] = Mratio
    data_summary.loc[data_summary['problem_dp']==index, ['Rratio']] = Rratio
    data_summary.loc[data_summary['problem_dp']==index, ['Sratio']] = Sratio

# load data summaries
model_1_data = data_summary
model_2_data = pd.read_csv('model_2_data_summary.csv')

# model 1 regression
X_1 = MS ([poly('c_m1', degree =2), poly('Mratio', degree =2), poly('Rratio', degree =2), poly('Sratio', degree=2)]).fit_transform(model_1_data)

Y_1 = model_1_data[['default-solver']] # response

model_1 = sm.OLS(Y_1,X_1)
results_1 = model_1.fit()
summary_1 = results_1.summary()
print(summary_1)

with open('model_1_results.txt', 'w') as f:
    f.write('Model 1 Results: \n\n')
    f.write(str(summary_1))


print(data_summary)


