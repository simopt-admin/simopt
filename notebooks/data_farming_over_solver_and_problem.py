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
# # Demo for Data Farming over Solvers and Problems.
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
# ## Configuration Parameters
#
# This section defines the core parameters for the data farming experiment.
#
# To query model/problem/solver names, run `python scripts/list_directories.py`

# %%
solver_abbr_name = "ASTRODF"
problem_abbr_name = "CNTNEWS-1"
model_abbr_name = "CNTNEWS"

# Specify the names of the solver factors (in order) that will be varied.
solver_factor_headers = ["eta_1", "eta_2", "lambda_min"]
# Specify the names of the model factors (in order) that will be varied.
model_factor_headers = ["purchase_price", "sales_price", "order_quantity"]

# OPTIONAL: factors chosen for cross design
# factor name followed by list containing factor values to cross design over
solver_cross_design_factors = {"crn_across_solns": [True, False]}
# model_cross_design_factors = {}

# OPTIONAL: Provide additional overrides for default factors.
# If empty, default factor settings are used.
solver_fixed_factors = {}
model_fixed_factors = {"salvage_price": 5, "Burr_c": 1}

# Provide the name of a file  .txt locatated in the datafarming_experiments folder
# containing the following:
#    - one row corresponding to each solver factor being varied
#    - three columns:
#         - first column: lower bound for factor value
#         - second column: upper bound for factor value
#         - third column: (integer) number of digits for discretizing values
#                         (e.g., 0 corresponds to integral values for the factor)
solver_factor_settings_filename = "astrodf_testing"
model_factor_settings_filename = "testing_model_cntnews_1"

# Specify the number stacks to use for ruby design creation
solver_n_stacks = 1
problem_n_stacks = 1

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

# %%
from simopt.experiment_base import create_design

# Create DataFarmingExperiment object for solver design
solver_design_list = create_design(
    name=solver_abbr_name,
    factor_headers=solver_factor_headers,
    factor_settings_filename=solver_factor_settings_filename,
    n_stacks=solver_n_stacks,
    fixed_factors=solver_fixed_factors,  # optional
    cross_design_factors=solver_cross_design_factors,  # optional
)
# Create DataFarmingExperiment object for model design
model_design_list = create_design(
    name=model_abbr_name,
    factor_headers=model_factor_headers,
    factor_settings_filename=model_factor_settings_filename,
    n_stacks=problem_n_stacks,
    fixed_factors=model_fixed_factors,  # optional
    # cross_design_factors=model_cross_design_factors, #optional
)

# %%
from simopt.experiment_base import ProblemsSolvers

# create solver name list for ProblemsSolvers
solver_names = []
for _ in range(len(solver_design_list)):
    solver_names.append(solver_abbr_name)

# create problem name list for ProblemsSolvers
problem_names = []
for _ in range(len(model_design_list)):
    problem_names.append(problem_abbr_name)

# Create ProblemsSolvers experiment with solver and model design
experiment = ProblemsSolvers(
    solver_factors=solver_design_list,
    problem_factors=model_design_list,
    solver_names=solver_names,
    problem_names=problem_names,
)

# check compatibility of selected solvers and problems
experiment.check_compatibility()

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
