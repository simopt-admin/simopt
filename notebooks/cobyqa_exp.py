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
# # Demo for the ProblemsSolvers class.
#
# This script is intended to help with debugging problems and solvers.
#
# It create problem-solver groups (using the directory) and runs multiple macroreplications of each problem-solver pair.

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
# Specify the names of the solver(s) and problem(s) to test.
solver_abbr_names = ["COBYQA", "RNDSRCH"]
problem_abbr_names = ["NETWORK-1"]

adapt_factors = {"adaptive": True}
fixed_factors = {"adaptive": False, "n":1}
problem_factors = {"budget": 20000}
rnd_factors = {}
num_macroreps = 10
num_postreps = 50
num_postreps_init_opt = 50

# %%
# Initialize an instance of the experiment class.
from simopt.experiment_base import ProblemsSolvers

mymetaexperiment = ProblemsSolvers(
    solver_names=solver_abbr_names, problem_names=problem_abbr_names, problem_factors=[problem_factors], solver_factors = [adapt_factors, rnd_factors]
)

# Write to log file.
mymetaexperiment.log_group_experiment_results()

# %%
# Run a fixed number of macroreplications of each solver on each problem.
mymetaexperiment.run(n_macroreps=num_macroreps)

# %%
print("Post-processing results.")
# Run a fixed number of postreplications at all recommended solutions.
mymetaexperiment.post_replicate(n_postreps=num_postreps)

# %%
print("Post-normalizing results.")

# Find an optimal solution x* for normalization.
mymetaexperiment.post_normalize(n_postreps_init_opt=num_postreps_init_opt)

# %%
# Produce basic plots.

from simopt.experiment_base import PlotType, plot_solvability_profiles, plot_progress_curves, plot_terminal_scatterplots, plot_terminal_progress

print("Plotting results...")


def _print_path(plot_path: list[Path]) -> None:
    print(f"Plot saved to {plot_path!s}")
experiments = mymetaexperiment.experiments

for i in range(len(experiments[0])):
    plot_progress_curves(
        [experiments[solver_idx][i] for solver_idx in range(2)],
        plot_type=PlotType.ALL,
        all_in_one=True,
        print_max_hw=True,
    )
# for i in range(len(experiments[0])):
#     plot_progress_curves(
#         [experiments[solver_idx][i] for solver_idx in range(2)],
#         plot_type=PlotType.MEAN,
#         all_in_one=True,
#         print_max_hw=True,
#     )

print("Plotting complete!")
