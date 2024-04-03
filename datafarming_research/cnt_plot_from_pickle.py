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

pickle_1 = './datafarming_research/first_ASTRODF_and_cnt_design/pickles/group_ASTRODF_on_CNTNEWS-1.pickle'
with open(pickle_1, 'rb') as f:
    experiment_1 = pickle.load(f)

# pickle_2 = './datafarming_research/ASTRODF_cnt_design_1/group_ASTRODF_on_CNTNEWS-1.pickle'
# with open(pickle_2, 'rb') as f:
#     experiment_2 = pickle.load(f)
    
# # give new names to differentiate between solver variants
# for solver in experiment_1.experiments:
#     for dp in solver:
#         print(dp)
#         dp.solver.name = 'ASTRODF_gamma_1-4' 

experiments = experiment_1.experiments 

print("Plotting results.")
# Produce basic plots of the solvers on the problems.
plot_solvability_profiles(experiments=experiments, plot_type="diff_cdf_solvability")
# Plots will be saved in the folder experiments/plots.
print("Finished. Plots can be found in experiments/plots folder.")





