# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:00:53 2024

@author: Owner
"""

import pandas as pd
import ast
import pickle
import numpy as np
import os
import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
from simopt.experiment_base import ProblemsSolvers


# results_filename = './final_200_postreps/results.csv' # location of csv file that you want to edit
# problem_design_filename =  'cnt_design.xlsx' # location of file with  problem design points
# solver_design_filename = 'ASTRODF_design.txt' #location of file with solver design points
edited_filename = './final_200_postreps/200_postrep_results_edited.csv' # save name of new edited file
# problem_design = pd.read_excel(problem_design_filename)
# solver_design = pd.read_csv(solver_design_filename, sep = '\t', header=None)
# results = pd.read_csv(results_filename)
results = pd.read_csv(edited_filename)
pickle_filename = './final_200_postreps/combined_200.pickle'
new_file = './final_200_postreps/200_postrep_results_extended.csv'
with open(pickle_filename, 'rb') as f:
    PSobject = pickle.load(f)
print('pickle loaded')
experiments = PSobject.experiments    

exp_dict = {} # relates experiment objects back to sol_dp & prob_dp
    
prob_n_dp = 49
sol_n_dp = 49

new_results = pd.DataFrame()
            
# add aditional run information
new_results['Initial Solution'] = 0
new_results['Initial Objective Function Value'] = 0
new_results['Optimal Solution'] =0
new_results['Optimal Objective Function Value'] = 0

row = 0

for solver in experiments:
    for exp in solver:
        for mrep in range(10):
            print('mrep', mrep)
            print(exp.x0)
            new_results.loc[row, 'Initial Solution'] = str(exp.x0)
            new_results.loc[row, 'Initial Objective Function Value'] = exp.x0_postreps[mrep]
            new_results.loc[row, 'Optimal Solution'] = str(tuple([round(x, 4) for x in exp.all_recommended_xs[mrep][-1]]))
            new_results.loc[row, 'Optimal Objective Function Value'] = exp.all_est_objectives[mrep][-1]
            row += 1
        
print(new_results)      

extended = pd.concat([results,new_results], axis = 1)
print(extended)

extended.to_csv(new_file, index = False)     


    