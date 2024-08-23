# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:48:01 2024

@author: Owner
"""

import os
import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
import pandas as pd
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles

def main():
    
    solver_name = 'RNDSRCH'
    problem_name = 'EXAMPLE-1'    
    solver_factors_1 = {'sample_size': 10}
    solver_factors_2 = {'sample_size': 5}
    problem_factors_1 = {}
    problem_factors_2 = {"x": (1.0,1.0)}
    
    solver_names = [solver_name, solver_name]
    problem_names = [problem_name, problem_name]
    
    solver_factors = [solver_factors_1 , solver_factors_2]
    problem_factors = [problem_factors_1,  problem_factors_2]
    
    solver_renames = ['RNDSRCH_1', 'RNDSRCH_2']
    problem_renames = ['EX_1', 'EX_2']
    
    exp = ProblemsSolvers(solver_factors = solver_factors,
                          problem_factors = problem_factors,
                          solver_names= solver_names,
                          problem_names = problem_names,
                          solver_renames = solver_renames,
                          problem_renames = problem_renames)
    
    exp.run(n_macroreps = 5)
    exp.post_replicate( n_postreps = 10, crn_across_macroreps =True)
    exp.post_normalize(n_postreps_init_opt = 10)
    exp.record_group_experiment_results()
    exp.log_group_experiment_results()
    exp.report_group_statistics()

    
    print('experiment info')    
    for solver in exp.experiments:
        for dp in solver:
            print('solver name', dp.solver.name)
            print('problem_name', dp.problem.name)
        
        
if (__name__ == "__main__"):
    main()