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
# # Demo for running multiple problem-solver pairs.
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
solver_abbr_names = ["RNDSRCH", "ASTRODF", "NELDMD"]
problem_abbr_names = ["CNTNEWS-1", "SAN-1"]

num_macroreps = 3
num_postreps = 50
num_postreps_init_opt = 50

# %%
# Create one problem-solver pair for each solver on each problem.
from simopt.experiment_base import ProblemSolver, post_normalize

experiments = [
    [
        ProblemSolver(solver_name=solver_name, problem_name=problem_name)
        for problem_name in problem_abbr_names
    ]
    for solver_name in solver_abbr_names
]

# %%
# Run a fixed number of macroreplications of each solver on each problem.
for solver_experiments in experiments:
    for experiment in solver_experiments:
        experiment.run(n_macroreps=num_macroreps)

# %%
print("Post-processing results.")
# Run a fixed number of postreplications at all recommended solutions.
for solver_experiments in experiments:
    for experiment in solver_experiments:
        experiment.post_replicate(n_postreps=num_postreps)

# %%
print("Post-normalizing results.")

# Find an optimal solution x* for normalization.
for problem_idx in range(len(problem_abbr_names)):
    post_normalize(
        experiments=[solver_experiments[problem_idx] for solver_experiments in experiments],
        n_postreps_init_opt=num_postreps_init_opt,
    )

# %%
# Produce basic plots.

from simopt.experiment_base import PlotType, plot_solvability_profiles

print("Plotting results...")


def _print_path(plot_path: list[Path]) -> None:
    print(f"Plot saved to {plot_path!s}")


_print_path(
    plot_solvability_profiles(
        experiments=experiments, plot_type=PlotType.CDF_SOLVABILITY
    )
)

print("Plotting complete!")
