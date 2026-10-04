# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: simopt
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo script for the ProblemSolver class.
#
# This script is intended to help with debugging problems and solvers.
#
# It create a problem-solver pairing (using the directory) and runs multiple macroreplications of the solver on the problem.

# %% [markdown]
# ## Append SimOpt Path
#
# Since the notebook is stored in simopt/notebooks, we need to append the parent simopt directory to the system path to import the necessary modules later on.

# %%
import sys
from pathlib import Path

# Take the current directory, find the parent, and add it to the system path
sys.path.append(str(Path.cwd().parent))

# %% [markdown]
# ## Configuration Parameters
#
# This section defines the core parameters for the demo.
#
# To query model/problem/solver names, run `python scripts/list_directories.py`

# %%
# Specify the names of the solver and problem to test.
# Example with random search solver on continuous newsvendor problem:
# solver_abbr_name = "RNDSRCH"
# problem_abbr_name = "CNTNEWS-1"
solver_abbr_name = "RNDSRCH"
problem_abbr_name = "CNTNEWS-1"

num_macroreps = 10
num_postreps = 200
num_postreps_init_opt = 200

# %%
# If the solver runs have already been performed, specify the file name path.
# If no path is specified, a new run will be performed.
file_name_path = None

if file_name_path is None:
    # Import the ProblemSolver class and other useful functions
    from simopt.experiment_base import ProblemSolver

    # Initialize an instance of the experiment class.
    myexperiment = ProblemSolver(solver_abbr_name, problem_abbr_name)

    # Run a fixed number of macroreplications of the solver on the problem.
    myexperiment.run(n_macroreps=num_macroreps)
else:
    # following pair of lines (and uncomment the myexperiment.run(...)
    # line above) to read in results from a .pickle file.
    # file_name_path = ""
    from simopt.experiment_base import read_experiment_results

    myexperiment = read_experiment_results(file_name_path)

# %%
# Run a fixed number of postreplications at all recommended solutions.
myexperiment.post_replicate(n_postreps=num_postreps)

# %%
# Find an optimal solution x* for normalization.
from simopt.experiment_base import post_normalize

post_normalize([myexperiment], n_postreps_init_opt=num_postreps_init_opt)

# %%
# Log results.
myexperiment.log_experiment_results()

# %%
# Produce basic plots.

from simopt.experiment_base import PlotType
from simopt.analysis.progress_curve import ProgressCurve
from simopt.analysis.solvability_cdf import SolvabilityCdf
from simopt.experiment.plot import save_plots
from simopt.options import ConfidenceIntervalOptions

print("Plotting results...")


def _print_path(plot_path: list[Path]) -> None:
    print(f"Plot saved to {plot_path!s}")


_print_path(
    save_plots(
        [myexperiment],
        ProgressCurve(
            agg=PlotType.ALL.value,
            normalize=False,
            beta=0.5,
            ci_options=ConfidenceIntervalOptions(
                n_bootstraps=100, confidence_level=0.95, bias_correction=True
            ),
        ),
        print_max_hw=True,
        use_experiment_crn=True,
    )
)
_print_path(
    save_plots(
        [myexperiment],
        ProgressCurve(
            agg=PlotType.MEAN.value,
            normalize=False,
            beta=0.5,
            ci_options=ConfidenceIntervalOptions(
                n_bootstraps=100, confidence_level=0.95, bias_correction=True
            ),
        ),
        print_max_hw=True,
        use_experiment_crn=True,
    )
)
_print_path(
    save_plots(
        [myexperiment],
        ProgressCurve(
            agg=PlotType.QUANTILE.value,
            normalize=False,
            beta=0.9,
            ci_options=ConfidenceIntervalOptions(
                n_bootstraps=100, confidence_level=0.95, bias_correction=True
            ),
        ),
        print_max_hw=True,
        use_experiment_crn=True,
    )
)
_print_path(
    save_plots(
        [myexperiment],
        SolvabilityCdf(
            solve_tolerance=0.1,
            ci_options=ConfidenceIntervalOptions(
                n_bootstraps=100, confidence_level=0.95, bias_correction=True
            ),
        ),
        print_max_hw=True,
        use_experiment_crn=True,
    )
)

print("Plotting complete!")
