"""
This script is intended to help with debugging problems and solvers.
It create problem-solver groups (using the directory) and runs multiple
macroreplications of each problem-solver pair.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

# Import the ProblemsSolvers class and other useful functions
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles

def main() -> None:
    # !! When testing a new solver/problem, first go to directory.py.
    # There you should add the import statement and an entry in the respective
    # dictionary (or dictionaries).
    # See directory.py for more details.

    # Specify the names of the solver and problem to test.
    # These names are strings and should match those input to directory.py.
    # Ex:
    solver_names = ["RNDSRCH", "ASTRODF", "NELDMD"]
    problem_names = ["CNTNEWS-1", "SAN-1"]


    # Initialize an instance of the experiment class.
    mymetaexperiment = ProblemsSolvers(solver_names = solver_names, problem_names = problem_names)

    # Write to log file.
    mymetaexperiment.log_group_experiment_results()

    # Run a fixed number of macroreplications of each solver on each problem.
    mymetaexperiment.run(n_macroreps=3)

    print("Post-processing results.")
    # Run a fixed number of postreplications at all recommended solutions.
    mymetaexperiment.post_replicate(n_postreps=50)
    # Find an optimal solution x* for normalization.
    mymetaexperiment.post_normalize(n_postreps_init_opt=50)

    print("Plotting results.")
    # Produce basic plots of the solvers on the problems.
    plot_solvability_profiles(experiments=mymetaexperiment.experiments, plot_type="cdf_solvability")

    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")

if __name__ == "__main__":
    main()