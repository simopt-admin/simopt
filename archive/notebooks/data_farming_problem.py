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
# # Demo for Data Farming over Problems.
#
# This script is intended to help with running a data-farming experiment on a problem.
#
# It creates a design of problem factors and runs multiple macroreplications at each version of the problem.
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
# ## Problem/Model Configuration Parameters
#
# To query model/problem/solver names, run `python scripts/list_directories.py`

# %%
# Abbreviated name of the problem and model
problem_abbr_name = "CNTNEWS-1"
model_abbr_name = "CNTNEWS"

# Name of each factor being data farmed for the model
model_factor_headers = ["purchase_price", "sales_price", "order_quantity"]

# List of tuples defining the minimum, maximum, and # of decimals for each factor
# Each tuple corresponds to a factor in model_factor_headers
model_factor_settings = [(4.0, 6.0, 1), (8.0, 12.0, 1), (0.4, 0.6, 2)]

# Number of stacks for the model
model_n_stacks = 1

# Fixed factors for the model (if any)
model_fixed_factors = {"salvage_price": 3, "Burr_c": 1}

# Cross design factors for the model (if any)
model_cross_design_factors = {}

# %% [markdown]
# ## Create Problem/Model Design

# %%
from simopt.experiment_base import create_design

# Create DataFarmingExperiment object for model design
model_design_list = create_design(
    name=model_abbr_name,
    factor_headers=model_factor_headers,
    factor_settings=model_factor_settings,
    n_stacks=model_n_stacks,
    fixed_factors=model_fixed_factors,  # optional
    # cross_design_factors=model_cross_design_factors, #optional
)

# %% [markdown]
# ## Experiment Configuration Parameters

# %%
solver_names = ["ASTRODF", "RNDSRCH"]
solver_fixed_factors = [{"eta_1": 0.1, "eta_2": 0.8}, {"sample_size": 15}]

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
from simopt.directory import problem_directory, solver_directory
from simopt.experiment.api import SimulationConfig, run
from simopt.experiment.single import ProblemSolver
from simopt.options import CrnOptions
from simopt.problem import Problem

def make_problem(name: str, factors: dict) -> Problem:
    """Split design-point factors into problem and model factors."""
    default = problem_directory[name]()
    return problem_directory[name](
        fixed_factors={k: v for k, v in factors.items() if k in default.specifications},
        model_fixed_factors={
            k: v for k, v in factors.items() if k in default.model.specifications
        },
    )

# Create one solver per name and one problem per model design point
solvers = [
    solver_directory[name](fixed_factors=factors)
    for name, factors in zip(solver_names, solver_fixed_factors, strict=True)
]
problems = [make_problem(problem_abbr_name, dp) for dp in model_design_list]

# Create the cross product of solvers and problem design points
experiments = [
    ProblemSolver(solver=solver, problem=problem) for problem in problems for solver in solvers
]

# check compatibility of selected solvers and problems
for ps in experiments:
    if message := ps.check_compatibility():
        print(message)

simulation_config = SimulationConfig(
    n_mreps=n_macroreps,
    n_preps=n_postreps,
    n_preps_x0_xstar=n_postreps_init_opt,
)
crn_options = CrnOptions(
    across_budget=crn_across_budget,
    across_macroreps=crn_across_macroreps,
    across_x0_xstar=crn_across_init_opt,
)

# %% [markdown]
# ## Run Experiment

# %%
# Run macroreplications, postreplications and normalization at each design point.
analysis_inputs = run(experiments, simulation_config, crn_options)

# %%
# Inspect the results of the first problem-solver pair.
analysis_inputs[0].mean_df.head()
