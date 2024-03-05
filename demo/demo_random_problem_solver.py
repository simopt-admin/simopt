"""
This script is intended to help with debugging random problems and solvers.
It create a problem-solver pairing by importing problems and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o
import numpy as np
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import ProblemSolver, read_experiment_results, post_normalize, plot_progress_curves, plot_solvability_cdfs
from rng.mrg32k3a import MRG32k3a
from simopt.models.san_2 import SANLongestPath, SANLongestPathConstr
from simopt.models.smf import SMF_Max
from simopt.models.smfcvx import SMFCVX_Max
from simopt.models.rmitd0 import RMITDMaxRevenue
from simopt.models.mm1queue import MM1MinMeanSojournTime
from simopt.models.cascade import CascadeMax
from simopt.models.network_2 import NetworkMinTotalCost


# !! When testing a new solver/problem, first go to directory.py.
# See directory.py for more details.
# Specify the names of the solver to test.

# -----------------------------------------------
solver_name = "ACTIVESET"  # Random search solver
# -----------------------------------------------


def rebase(random_rng, n):
    new_rngs = []
    for rng in random_rng:
        stream_index = rng.s_ss_sss_index[0]
        substream_index = rng.s_ss_sss_index[1]
        subsubstream_index = rng.s_ss_sss_index[2]
        new_rngs.append(MRG32k3a(s_ss_sss_index=[stream_index, substream_index + n, subsubstream_index]))
    random_rng = new_rngs
    return random_rng

def strtobool(t):
    t = t.lower()
    if t == "t":
        return True
    else:
        return False

n_inst = int(input('Please enter the number of instance you want to generate: '))
rand = input('Please decide whether you want to generate random instances or determinent instances (T/F): ')
rand = strtobool(rand)

# model_fixed_factors = {'num_nodes': 70, 'num_arcs': 100}  # Override model factors
# fixed_factors = {'initial_solution': (15,)*100, 'budget': 100 * 400}  # Override solver factors
model_fixed_factors, fixed_factors = {}, {}
myproblem = NetworkMinTotalCost(random=True, fixed_factors=fixed_factors, model_fixed_factors=model_fixed_factors)
# myproblem = SMFCVX_Max(random=True, fixed_factors=fixed_factors, model_fixed_factors=model_fixed_factors)
# myproblem = CascadeMax(random=True, fixed_factors=fixed_factors, model_fixed_factors=model_fixed_factors)

random_rng = [MRG32k3a(s_ss_sss_index=[2, 4, ss]) for ss in range(myproblem.model.n_random, myproblem.model.n_random + myproblem.n_rngs)]
rng_list2 = [MRG32k3a(s_ss_sss_index=[2, 4, ss]) for ss in range(myproblem.model.n_random)]

# Generate 5 random problem instances
for i in range(n_inst):
    random_rng = rebase(random_rng, 1)
    rng_list2 = rebase(rng_list2, 1)
    myproblem = NetworkMinTotalCost(random=rand, fixed_factors=fixed_factors, random_rng=rng_list2, model_fixed_factors=model_fixed_factors)
    # myproblem = SMFCVX_Max(random=rand, fixed_factors=fixed_factors, random_rng=rng_list2, model_fixed_factors=model_fixed_factors)
    # myproblem = CascadeMax(random=rand, fixed_factors=fixed_factors, random_rng=rng_list2, model_fixed_factors=model_fixed_factors)
    myproblem.attach_rngs(random_rng)
    problem_name = myproblem.model.name + str(i)
    print('-------------------------------------------------------')
    print(f"Testing solver {solver_name} on problem {problem_name}.")

    # Specify file path name for storing experiment outputs in .pickle file.
    file_name_path = "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
    print(f"Results will be stored as {file_name_path}.")

    # Initialize an instance of the experiment class.
    myexperiment = ProblemSolver(solver_name=solver_name, problem=myproblem)

    # Run a fixed number of macroreplications of the solver on the problem.
    myexperiment.run(n_macroreps=10)

    # If the solver runs have already been performed, uncomment the
    # following pair of lines (and uncommmen the myexperiment.run(...)
    # line above) to read in results from a .pickle file.
    # myexperiment = read_experiment_results(file_name_path)

    print("Post-processing results.")
    # Run a fixed number of postreplications at all recommended solutions.
    myexperiment.post_replicate(n_postreps=30) #200, 10
    # Find an optimal solution x* for normalization.
    post_normalize([myexperiment], n_postreps_init_opt=10) #200, 5

    # Log results.
    myexperiment.log_experiment_results()
    print('initial solution: ', myproblem.factors['initial_solution'])
    print("Optimal solution: ",np.array(myexperiment.xstar))
    print("Optimal Value: ", myexperiment.all_est_objectives[0])

    print("Plotting results.")
    # Produce basic plots of the solver on the problem.
    plot_progress_curves(experiments=[myexperiment], plot_type="all", normalize=False)
    plot_progress_curves(experiments=[myexperiment], plot_type="mean", normalize=False)
    plot_progress_curves(experiments=[myexperiment], plot_type="quantile", beta=0.90, normalize=False)
    plot_solvability_cdfs(experiments=[myexperiment], solve_tol=0.1)

    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")