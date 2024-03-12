"""
This script is intended to help with debugging problems and solvers.
It create problem-solver groups (using the directory) and runs multiple
macroreplications of each problem-solver pair.
"""

import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the ProblemsSolvers class and other useful functions
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles
from simopt.models.san import SANLongestPath
# from simopt.models.san_1 import SANLongestPath1
from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress


# !! When testing a new solver/problem, first go to directory.py.
# There you should add the import statement and an entry in the respective
# dictionary (or dictionaries).
# See directory.py for more details.

# Specify the names of the solver and problem to test.
# These names are strings and should match those input to directory.py.
# Ex:
# solver_names = ["ASTRODF", "Boom-PGD", "Boom-FW", "RNDSRCH", "GASSO", "NELDMD"]
solver_names = ['ACTIVESET','Boom-PGD', "Boom-FW"]
# problem_names = ["OPENJACKSON-1", 'SAN-1', 'SMFCVX-1', 'SMF-1', 'CASCADE-1', 'NETWORK-1']
# problem_names = ["DYNAMNEWS-1", "SSCONT-1", "SAN-1"] "OPENJ-1"
problem_names = ["SMF-1", 'SMFCVX-1', 'CASCADE-1']
# problems = [SANLongestPath, SMF_Max, RMITDMaxRevenue, MM1MinMeanSojournTime]


# Initialize an instance of the experiment class.
mymetaexperiment = ProblemsSolvers(solver_names, problem_names)
# mymetaexperiment = ProblemsSolvers(solver_names=solver_names, problems = problems)

n_solvers = len(mymetaexperiment.experiments)
n_problems = len(mymetaexperiment.experiments[0])

# Run a fixed number of macroreplications of each solver on each problem.
mymetaexperiment.run(n_macroreps=10)


print("Post-processing results.")
# Run a fixed number of postreplications at all recommended solutions.
mymetaexperiment.post_replicate(n_postreps=20)
# Find an optimal solution x* for normalization.
mymetaexperiment.post_normalize(n_postreps_init_opt=20)

print("Plotting results.")
# Produce basic plots of the solvers on the problems.
plot_solvability_profiles(experiments=mymetaexperiment.experiments, plot_type="cdf_solvability") # cdf_solvability
# plot_solvability_profiles(experiments=mymetaexperiment.experiments, plot_type="diff_quantile_solvability", ref_solver='RNDSRCH', all_in_one=True, plot_CIs=True, print_max_hw=True) 
# plot_terminal_scatterplots(experiments=mymetaexperiment.experiments, all_in_one=True)

# Plot the mean progress curves of the solvers on the problems.
CI_param = True
for i in range(n_problems):
    plot_progress_curves([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="mean", all_in_one=True, plot_CIs=CI_param, print_max_hw=True)


# Plots will be saved in the folder experiments/plots.
print("Finished. Plots can be found in experiments/plots folder.")