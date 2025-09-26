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
# # Hello SimOpt!
#
# This script is intended to be an introduction to the world of SimOpt.
#
# The script creates a simple experiment and goes through the process of running, post-processing, post-normalizing, and recording the results.

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
# Create an instance of the ProblemSolver class for the given solver and problem
solver_name = "ASTRODF"
problem_name = "CNTNEWS-1"

num_macroreps = 10
num_postreps = 200
num_postreps_init_opt = 200

# %%
from simopt.experiment_base import ProblemSolver

myexperiment = ProblemSolver(solver_name, problem_name)
# Run the experiment with a specified number of macro-repetitions
myexperiment.run(n_macroreps=num_macroreps)

# %%
# Post-process the results
myexperiment.post_replicate(n_postreps=num_postreps)

# %%
# Normalize the results
from simopt.experiment_base import post_normalize

post_normalize(experiments=[myexperiment], n_postreps_init_opt=num_postreps_init_opt)

# %%
# Record the results and plot the mean progress curve.
from simopt.experiment_base import PlotType, plot_progress_curves

myexperiment.log_experiment_results()
plot_progress_curves(
    experiments=[myexperiment],
    plot_type=PlotType.MEAN,
    normalize=False,
)
