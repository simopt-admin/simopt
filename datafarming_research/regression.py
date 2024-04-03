# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:52:50 2024

@author: Owner
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
summarize ,
poly)


# load data summaries
model_1_data = pd.read_csv('model_1_data_summary.csv')
model_2_data = pd.read_csv('model_2_data_summary.csv')

# model 1 regression
X_1 = MS ([poly('c_m1', degree =2), poly('Mratio', degree =2), poly('Rratio', degree =2), poly('Sratio', degree=2)]).fit_transform(model_1_data)

Y_1 = model_1_data[['Difference btwn Best and Default Avg Optimality Gap']] # response

model_1 = sm.OLS(Y_1,X_1)
results_1 = model_1.fit()
summary_1 = results_1.summary()
print(summary_1)

with open('model_1_results.txt', 'w') as f:
    f.write('Model 1 Results: \n\n')
    f.write(str(summary_1))


# model 2 regressions
prob_n_dp = int(model_2_data['problem_dp'].max() + 1)

model_2_results = pd.DataFrame()
with open('model_2_results.txt', 'w') as f:
    f.write("Model 2 Results: \n\n")
for prob_dp in range(prob_n_dp):
    prob_df = model_2_data.loc[model_2_data['problem_dp'] == prob_dp, ['problem_dp', 'solver_dp', 'Average Optimality Gap', 'gamma_1', 'gamma_2', 'eta_1', 'eta_2']]
    X_2 = MS ([poly('gamma_1', degree =2), poly('gamma_2', degree =2), poly('eta_1', degree=2), poly('eta_2', degree =2)]).fit_transform(prob_df)
    Y_2 = prob_df[['Average Optimality Gap']]
    model_2 = sm.OLS(Y_2, X_2)    
    results_2 = model_2.fit()
    summary_2 = results_2.summary()

    with open('model_2_results.txt', 'a') as f:
        f.write(f'\n Problem Design Point: {prob_dp}: \n')
        f.write(str(summary_2))
        f.write('\n')
        f.write('\n')
    
