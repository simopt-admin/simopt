# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:12:44 2024

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

filename = './4-25_correct_gaps/group_ASTRODF_on_CNTNEWS-1.pickle'

with open(filename, 'rb') as f:
    exp = pickle.load(f)

obj_results =  []
sol_results = []


for solver in exp.experiments:
    for dp in solver:
        for macrep in range(10):
            solver_name = dp.solver.name
            problem_name = dp.problem.name
            obj_list = dp.all_est_objectives[macrep]
            sol_list = dp.all_recommended_xs[macrep]
            row_obj = [solver_name, problem_name, macrep]
            row_sol = [solver_name, problem_name, macrep]
            for item in obj_list:
                row_obj.append(item)
            for item in sol_list:
                row_sol.append(item)
            obj_results.append(row_obj)
            sol_results.append(row_sol)

obj_df = pd.DataFrame(obj_results)
sol_df = pd.DataFrame(sol_results)

obj_df.to_csv('./4-25_correct_gaps/4-25_obj_values.csv', index = False)
sol_df.to_csv('./4-25_correct_gaps/4-25_sol_values.csv', index = False)