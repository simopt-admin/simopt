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

pickle1= './final_200_postreps/Missing dp 0/group_ASTRODF_on_CNTNEWS-1.pickle'
with open(pickle1, 'rb') as f:
    exp1 = pickle.load(f)

pickle2= './final_run_dp0/group_dp_0.pickle'
with open(pickle2, 'rb') as f:
    exp2 = pickle.load(f)


for solver in exp2.experiments:
    exp1.experiments.append(solver)

n_postnormal = 200 # number of post replications at x0 and x*

exp1.post_normalize(n_postreps_init_opt = n_postnormal)
exp1.record_group_experiment_results()
exp1.log_group_experiment_results()
exp1.report_group_statistics()



