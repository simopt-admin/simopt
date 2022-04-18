"""
This script creates a problem-solver pairing for random san and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the Experiment class and other useful functions
from wrapper_base import Experiment, read_experiment_results, post_normalize, plot_progress_curves, plot_solvability_cdfs

# !! When testing a new solver/problem, first go to directory.py.
# There you should add the import statement and an entry in the respective
# dictionary (or dictionaries).
# See directory.py for more details.

# Specify the names of the solver and problem to test.
# These names are strings and should match those input to directory.py.
# solver_name = "NELDMD"
solver_name = "RNDSRCH"
problem_name = "RNDSAN-1"
print(f"Testing solver {solver_name} on problem {problem_name}.")


# Specify file path name for storing experiment outputs in .pickle file.
file_name_path = "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
print(f"Results will be stored as {file_name_path}.")

# Randomly generate a network.
num_nodes = 20
fwd_arcs = 3
fwd_reach = 12
arcs = []
from rng.mrg32k3a import MRG32k3a
arc_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
for i in range(num_nodes-1):
    indices = [arc_rng.randint(i+2, min(num_nodes, i+1+fwd_reach)) for _ in range(fwd_arcs)]
    for j in range(fwd_arcs):
        arcs.append([i+1,indices[j]])
for i in range(1,num_nodes):
    prev_index = arc_rng.randint(max(i+1-fwd_reach, 1), i)
    arcs.append([prev_index, i+1])
arcs = list(set(tuple(a) for a in arcs))
fixed_factors = {"num_nodes": num_nodes, "arcs": arcs, "arc_means": (1,)*len(arcs)}
prob_fixed_factors = {"initial_solution": (1,)*len(arcs)}

# Initialize an instance of the experiment class.
myexperiment = Experiment(solver_name, problem_name, problem_fixed_factors=prob_fixed_factors, model_fixed_factors=fixed_factors)

# Run a fixed number of macroreplications of the solver on the problem.
myexperiment.run(n_macroreps=10)

# If the solver runs have already been performed, uncomment the
# following pair of lines (and uncomment the myexperiment.run(...)
# line above) to read in results from a .pickle file.
# myexperiment = read_experiment_results(file_name_path)

print("Post-processing results.")
# Run a fixed number of postreplications at all recommended solutions.
myexperiment.post_replicate(n_postreps=10)
# Find an optimal solution x* for normalization.
post_normalize([myexperiment], n_postreps_init_opt=10)

print("Plotting results.")
# Produce basic plots of the solver on the problem
plot_progress_curves(experiments=[myexperiment], plot_type="all", normalize=False)
plot_progress_curves(experiments=[myexperiment], plot_type="mean", normalize=False)
plot_progress_curves(experiments=[myexperiment], plot_type="quantile", beta=0.90, normalize=False)
plot_solvability_cdfs(experiments=[myexperiment], solve_tol=0.1)

# Plots will be saved in the folder experiments/plots.
print("Finished. Plots can be found in experiments/plots folder.")