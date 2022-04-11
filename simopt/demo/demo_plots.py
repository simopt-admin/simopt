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
from wrapper_base import Experiment, read_experiment_results, post_normalize, plot_terminal_progress

solver_names = {"RNDSRCH", "ASTRODF", "NELDMD"}  # Random search solver
# problem_name = "CNTNEWS-1"  # Continuous newsvendor problem
problem_name = "SAN-1"
# solver_name = <solver_name>
# problem_name = <problem_name>

myexperiments = []
for solver_name in solver_names:
    print(f"Testing solver {solver_name} on problem {problem_name}.")
    # Initialize an instance of the experiment class.
    myexperiment = Experiment(solver_name, problem_name)

    file_name_path = "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"

    # Run a fixed number of macroreplications of the solver on the problem.
    # myexperiment.run(n_macroreps=10)

    # If the solver runs have already been performed, uncomment the
    # following pair of lines (and uncommmen the myexperiment.run(...)
    # line above) to read in results from a .pickle file.
    myexperiment = read_experiment_results(file_name_path)

    print("Post-processing results.")
    # Run a fixed number of postreplications at all recommended solutions.
    myexperiment.post_replicate(n_postreps=200)

    myexperiments.append(myexperiment)

# Find an optimal solution x* for normalization.
post_normalize(myexperiments, n_postreps_init_opt=200)

print("Plotting results.")
# Produce basic plots of the solver on the problem
plot_terminal_progress(experiments=myexperiments, plot_type="box", normalize=False)
plot_terminal_progress(experiments=myexperiments, plot_type="box", normalize=True)
plot_terminal_progress(experiments=myexperiments, plot_type="violin", normalize=False)
plot_terminal_progress(experiments=myexperiments, plot_type="violin", normalize=True)

# Plots will be saved in the folder experiments/plots.
print("Finished. Plots can be found in experiments/plots folder.")
