"""
This script is intended to help with debugging problems and solvers.
It create a problem-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the Experiment class and other useful functions
from experiment_base import Experiment, read_experiment_results, post_normalize, plot_progress_curves, plot_solvability_cdfs

# !! When testing a new solver/problem, first go to directory.py.
# There you should add the import statement and an entry in the respective
# dictionary (or dictionaries).
# See directory.py for more details.

# Specify the names of the solver and problem to test.

# solver_name = <solver_name>
# problem_name = <problem_name>
# These names are strings and should match those input to directory.py.

# Example with random search solver on continuous newsvendor problem.
# -----------------------------------------------
solver_name = "RNDSRCH"  # Random search solver
problem_name = "CNTNEWS-1"  # Continuous newsvendor problem
# -----------------------------------------------

print(f"Testing solver {solver_name} on problem {problem_name}.")

# Specify file path name for storing experiment outputs in .pickle file.
file_name_path = "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
print(f"Results will be stored as {file_name_path}.")

# Initialize an instance of the experiment class.
myexperiment = Experiment(solver_name, problem_name)

# Run a fixed number of macroreplications of the solver on the problem.
myexperiment.run(n_macroreps=20)

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
# Produce basic plots of the solver on the problem.
plot_progress_curves(experiments=[myexperiment], plot_type="all", normalize=False)
plot_progress_curves(experiments=[myexperiment], plot_type="mean", normalize=False)
plot_progress_curves(experiments=[myexperiment], plot_type="quantile", beta=0.90, normalize=False)
plot_solvability_cdfs(experiments=[myexperiment], solve_tol=0.1)

# Plots will be saved in the folder experiments/plots.
print("Finished. Plots can be found in experiments/plots folder.")