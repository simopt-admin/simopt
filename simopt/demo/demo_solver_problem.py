"""
This script is intended to help with debugging problems and solvers.
It create a problem-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""
import numpy as np
import sys
import os.path as o
import os

from numpy.lib.ufunclike import fix
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the Experiment class and other useful functions
from wrapper_base import Experiment, read_experiment_results, post_normalize, plot_progress_curves, plot_solvability_cdfs

# !! When testing a new solver/problem, first go to directory.py.
# There you should add the import statement and an entry in the respective
# dictionary (or dictionaries).
# See directory.py for more details.

# Specify the names of the solver and problem to test.
# These names are strings and should match those input to directory.py.
# Ex:
solver_name = "RNDSRCH"  # Random search solver
problem_name = "DYNAMNEWS-1"  # Continuous newsvendor problem
# solver_name = <solver_name>
# problem_name = <problem_name>
print(f"Testing solver {solver_name} on problem {problem_name}.")


# Specify file path name for storing experiment outputs in .pickle file.
file_name_path = "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
print(f"Results will be stored as {file_name_path}.")

c_utility = []
for j in range(1, 11):
    c_utility.append(5 + j)

fixed_factors = {
    "num_prod": 10,
    "num_customer": 30,
    "c_utility": c_utility,
    "mu": 1.0,
    "init_level":  list(3 * np.ones(10)),
    "price": list(9 * np.ones(10)),
    "cost": list(5 * np.ones(10)),
    "initial_solution": list(3 * np.ones(10))}

# Initialize an instance of the experiment class.
# myexperiment = Experiment(solver_name, problem_name)

myexperiment = Experiment(solver_name, problem_name, problem_fixed_factors = fixed_factors, model_fixed_factors= fixed_factors)

# Run a fixed number of macroreplications of the solver on the problem.
myexperiment.run(n_macroreps=10)

# If the solver runs have already been performed, uncomment the
# following pair of lines (and uncommmen the myexperiment.run(...)
# line above) to read in results from a .pickle file.
# myexperiment = read_experiment_results(file_name_path)

print("Post-processing results.")
# Run a fixed number of postreplications at all recommended solutions.
myexperiment.post_replicate(n_postreps=200)
# Find an optimal solution x* for normalization.
post_normalize([myexperiment], n_postreps_init_opt=200)

print("Plotting results.")
# Produce basic plots of the solver on the problem
plot_progress_curves(experiments=[myexperiment], plot_type="all", normalize=False)
plot_progress_curves(experiments=[myexperiment], plot_type="mean", normalize=False)
plot_progress_curves(experiments=[myexperiment], plot_type="quantile", beta=0.90, normalize=False)
plot_solvability_cdfs(experiments=[myexperiment], solve_tol=0.1)

# Plots will be saved in the folder experiments/plots.
print("Finished. Plots can be found in experiments/plots folder.")