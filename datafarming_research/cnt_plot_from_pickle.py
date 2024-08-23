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

pickle_file= './1_stack_final_run/group_ASTRODF_on_CNTNEWS-1.pickle'
with open(pickle_file, 'rb') as f:
    PSobject = pickle.load(f)
print(PSobject)
# seperate into experiment groups
for i, solver in enumerate(PSobject.experiments):
    if i == 0:
        name = 'ASTRODF_default'
    else:
        name = f'ASTRODF_{i}'
    for  exp in solver:
        exp.solver.name = name 
    
    if i == 0:
        exp_group_1 = [solver]
        exp_group_2 = [solver]
        exp_group_3 = [solver]
        exp_group_4 = [solver]
        exp_group_5 = [solver]
    else:
        if i in range(4):
            exp_group_1.append(solver)
        if i in range(4,7):
            exp_group_2.append(solver)
        if i  in range(7,10):
            exp_group_3.append(solver)
        if i in range(10,13):
            exp_group_4.append(solver)
        if i in range(13,16):
            exp_group_5.append(solver)
            
        
        
        
       
print(exp_group_1)        
# make diff cdf plots
#plot_solvability_profiles(experiments=exp_group_1, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
#print("Group 1 Finished. Plots can be found in experiments/plots folder.")
#plot_solvability_profiles(experiments=exp_group_2, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
#print("Group 2 Finished. Plots can be found in experiments/plots folder.")
#plot_solvability_profiles(experiments=exp_group_3, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
#print("Group 3 Finished. Plots can be found in experiments/plots folder.")
plot_solvability_profiles(experiments=exp_group_4, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
print("Group 4 Finished. Plots can be found in experiments/plots folder.")
# plot_solvability_profiles(experiments=exp_group_5, plot_type="diff_cdf_solvability", ref_solver = 'ASTRODF_default')
    
# # cdf solvability plots
# plot_solvability_profiles(experiments=exp_group_1, plot_type="cdf_solvability")
# plot_solvability_profiles(experiments=exp_group_2, plot_type="cdf_solvability")
# plot_solvability_profiles(experiments=exp_group_3, plot_type="cdf_solvability")

# Produce basic plots of the solvers on the problems.
# Plots will be saved in the folder experiments/plots.
print("Finished. Plots can be found in experiments/plots folder.")






