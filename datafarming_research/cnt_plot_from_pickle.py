"""
This script is intended to load a design over problem factors to be run on
the updated version of CNTNEWS-1.
"""
import os
import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
import pandas as pd
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles
import pickle

pickle_file= './experiments/outputs/group_ASTRODF_on_CNTNEWS-1.pickle'
with open(pickle_file, 'rb') as f:
    PSobject = pickle.load(f)
print(PSobject)
# seperate into experiment groups
for solver in PSobject.experiments:
    print(solver)
    for exp in solver:
        print('test')
        print(exp.solver.name)
        if exp.solver.name == 'ASTRODF_default':
            exp_group_1 = [exp]
            exp_group_2 = [exp]
            exp_group_3 = [exp]
            exp_group_4 = [exp]
            exp_group_5 = [exp]
            exp_group_6 = [exp]
            exp_group_7 = [exp]
        for i in range(7):
           if exp.solver.name == f"ASTRODF_{i}":
               exp_group_1.append(exp)
        for i in range(7, 14):
           if exp.solver.name == f"ASTRODF_{i}":
               exp_group_2.append(exp)
        for i in range(14, 21):
           if exp.solver.name == f"ASTRODF_{i}":
               exp_group_3.append(exp)
        for i in range(21, 28):
           if exp.solver.name == f"ASTRODF_{i}":
               exp_group_4.append(exp)
        for i in range(28, 35):
           if exp.solver.name == f"ASTRODF_{i}":
               exp_group_5.append(exp)       
        for i in range(35, 42):
           if exp.solver.name == f"ASTRODF_{i}":
               exp_group_6.append(exp)    
        for i in range(42, 49):
           if exp.solver.name == f"ASTRODF_{i}":
               exp_group_7.append(exp)

# make diff cdf plots
plot_solvability_profiles(experiments=exp_group_1, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
plot_solvability_profiles(experiments=exp_group_2, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
plot_solvability_profiles(experiments=exp_group_3, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
plot_solvability_profiles(experiments=exp_group_4, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
plot_solvability_profiles(experiments=exp_group_5, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
plot_solvability_profiles(experiments=exp_group_6, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
plot_solvability_profiles(experiments=exp_group_7, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
    
# cdf solvability plots
plot_solvability_profiles(experiments=exp_group_1, plot_type="cdf_solvability")
plot_solvability_profiles(experiments=exp_group_2, plot_type="cdf_solvability")
plot_solvability_profiles(experiments=exp_group_3, plot_type="cdf_solvability")
plot_solvability_profiles(experiments=exp_group_4, plot_type="cdf_solvability")
plot_solvability_profiles(experiments=exp_group_5, plot_type="cdf_solvability")
plot_solvability_profiles(experiments=exp_group_6, plot_type="cdf_solvability")
plot_solvability_profiles(experiments=exp_group_7, plot_type="cdf_solvability")
# Produce basic plots of the solvers on the problems.
# Plots will be saved in the folder experiments/plots.
print("Finished. Plots can be found in experiments/plots folder.")






