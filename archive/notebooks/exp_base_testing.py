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
# This section defines the core parameters for the demo.
#
# To query model/problem/solver names, run `python scripts/list_directories.py`

# %%
# Specify the name of the solver as it appears in directory.py

solver_names = ["RNDSRCH", "RNDSRCH", "ASTRODF"]

solver_renames = ["RND_test1", "RND_test2", "AST_test"]

problem_names = ["EXAMPLE-1", "CNTNEWS-1"]

problem_renames = ["EX_test", "NEWS_test"]

experiment_name = "test_exp"

solver_factors = [{}, {"sample_size": 2}, {}]

problem_factors = [{}, {}]

num_macroreps = 2
num_postreps = 10
num_postreps_init_opt = 10

# %%
# Create the cross product of solvers and problems
from simopt.experiment.api import SimulationConfig, run
from simopt.experiment.single import ProblemSolver

experiments = [
    ProblemSolver(
        solver_name=solver_name,
        problem_name=problem_name,
        solver_rename=solver_rename,
        problem_rename=problem_rename,
        solver_fixed_factors=solver_ff,
        problem_fixed_factors=problem_ff,
    )
    for solver_name, solver_rename, solver_ff in zip(
        solver_names, solver_renames, solver_factors, strict=True
    )
    for problem_name, problem_rename, problem_ff in zip(
        problem_names, problem_renames, problem_factors, strict=True
    )
]

# check compatibility of selected solvers and problems
for ps in experiments:
    if message := ps.check_compatibility():
        print(message)

# %%
simulation_config = SimulationConfig(
    n_mreps=num_macroreps,
    n_preps=num_postreps,
    n_preps_x0_xstar=num_postreps_init_opt,
)

# %%
# Run macroreplications, postreplications and normalization.
analysis_inputs = run(experiments, simulation_config)

# %%
# Inspect the results of the first problem-solver pair.
analysis_inputs[0].mean_df.head()
