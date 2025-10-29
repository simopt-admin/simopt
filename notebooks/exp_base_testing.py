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
# Create ProblemsSolvers experiment with solver and model design
from simopt.experiment_base import ProblemsSolvers

experiment = ProblemsSolvers(
    solver_factors=solver_factors,
    problem_factors=problem_factors,
    solver_names=solver_names,
    problem_names=problem_names,
    solver_renames=solver_renames,
    problem_renames=problem_renames,
    experiment_name=experiment_name,
    create_pair_pickles=True,
)

# check compatibility of selected solvers and problems
experiment.check_compatibility()

# %%
# Run macroreplications at each design point.
experiment.run(n_macroreps=num_macroreps)

# %%
# Postprocess the experimental results from each design point.
experiment.post_replicate(n_postreps=num_postreps)

# %%
experiment.post_normalize(n_postreps_init_opt=num_postreps_init_opt)

# %%
# Record and log results
experiment.record_group_experiment_results()
experiment.log_group_experiment_results()
experiment.report_group_statistics()
