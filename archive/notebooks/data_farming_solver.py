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

# Create one solver per solver design point and one problem per problem name
solvers = [
    solver_directory[solver_abbr_name](fixed_factors=dp) for dp in solver_design_list
]
problems = [
    make_problem(name, factors)
    for name, factors in zip(problem_names, problem_fixed_factors, strict=True)
]

# Create the cross product of solver design points and problems
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
