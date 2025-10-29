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
# # Demo for Data Farming over Solvers.
#
# This script is intended to help with running a data-farming experiment on a solver.
#
# It creates a design of solver factors and runs multiple macroreplications at each version of the solver.
#
# Outputs are printed to a file.

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
# ## Solver Configuration Parameters
#
# To query model/problem/solver names, run `python scripts/list_directories.py`

# %%
# Abbreviated name of the solver
solver_abbr_name = "ASTRODF"

# Name of each factor being data farmed
solver_factor_headers = ["eta_1", "eta_2", "lambda_min"]

# List of tuples defining the minimum, maximum, and # of decimals for each factor
# Each tuple corresponds to a factor in solver_factor_headers
solver_factor_settings = [(0.05, 0.15, 2), (0.6, 1.0, 2), (5, 6, 0)]

# Number of stacks for the solver
solver_n_stacks = 1

# Fixed factors for the solver (if any)
solver_fixed_factors = {}

# Cross design factors for the solver (if any)
solver_cross_design_factors = {"crn_across_solns": [True, False]}

# %% [markdown]
# ## Create Solver Design

# %%
from simopt.experiment_base import create_design

# Create DataFarmingExperiment object for solver design
solver_design_list = create_design(
    name=solver_abbr_name,
    factor_headers=solver_factor_headers,
    factor_settings=solver_factor_settings,
    n_stacks=solver_n_stacks,
    fixed_factors=solver_fixed_factors,
    cross_design_factors=solver_cross_design_factors,
)

# %% [markdown]
# ## Experiment Configuration Parameters

# %%
problem_names = ["SSCONT-1", "SAN-1"]

problem_fixed_factors = [
    {"budget": 2000, "demand_mean": 90.0, "fixed_cost": 25},
    {"budget": 500},
]

# Specify a common number of macroreplications of each unique solver/problem
# combination (i.e., the number of runs at each design point.)
n_macroreps = 3

# Specify the number of postreplications to take at each recommended solution
# from each macroreplication at each design point.
n_postreps = 100

# Specify the number of postreplications to take at x0 and x*.
n_postreps_init_opt = 200

# Specify the CRN control for postreplications.
crn_across_budget = True  # Default
crn_across_macroreps = False  # Default
crn_across_init_opt = True  # Default

# %% [markdown]
# ## Create Experiment using Specified Configuration

# %%
from simopt.experiment_base import ProblemsSolvers

# create solver name list
solver_names = [solver_abbr_name] * len(solver_design_list)

# Create ProblemsSolvers experiment with solver and model design
experiment = ProblemsSolvers(
    solver_factors=solver_design_list,
    problem_factors=problem_fixed_factors,
    solver_names=solver_names,
    problem_names=problem_names,
)

# check compatibility of selected solvers and problems
experiment.check_compatibility()

# %% [markdown]
# ## Run Experiment

# %%
# Run macroreplications at each design point.
experiment.run(n_macroreps)

# %%
# Postprocess the experimental results from each design point.
experiment.post_replicate(
    n_postreps=n_postreps,
    crn_across_budget=crn_across_budget,
    crn_across_macroreps=crn_across_macroreps,
)

# %%
experiment.post_normalize(
    n_postreps_init_opt=n_postreps_init_opt,
    crn_across_init_opt=crn_across_init_opt,
)

# %%
# Record and log results
experiment.record_group_experiment_results()
experiment.log_group_experiment_results()
experiment.report_group_statistics()
