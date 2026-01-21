# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DASSO Solver
#
# This script uses the DASSO sovler to solve a discrete optimization problem.
#
# The script creates a simple experiment and goes through running,
# post-processing, post-normalizing, and recording the results.

# %%
import sys

sys.path.append("..")

# %%
from simopt.base import Problem, Solver
from simopt.experiment_base import (
    PlotType,
    ProblemSolver,
    plot_progress_curves,
    plot_terminal_progress,
    post_normalize,
)
from simopt.models.example import Example2Problem
from simopt.solvers.dasso import DASSO
from simopt.solvers.randomsearch import RandomSearch

# %%
problem = Example2Problem(fixed_factors={"budget": 10000})

# %% jupyter={"is_executing": true}
dasso = DASSO()
random_search = RandomSearch()


# %%
def run_experiment(
    solver: Solver,
    problem: Problem,
    n_macroreps: int,
    n_postreps: int,
) -> ProblemSolver:
    """Run and post-process a solver/problem experiment."""
    experiment = ProblemSolver(solver=solver, problem=problem)
    experiment.run(n_macroreps=n_macroreps)
    experiment.post_replicate(n_postreps=n_postreps)
    return experiment


# %%
n_macroreps = 10
n_postreps = 100
experiments = [
    run_experiment(solver, problem, n_macroreps, n_postreps)
    for solver in [dasso, random_search]
]
experiment1, experiment2 = experiments
post_normalize(experiments, n_postreps)

# %% [markdown]
# ## Plots

# %%
plot_progress_curves([experiment1, experiment2], PlotType.ALL, normalize=False)

# %%
plot_terminal_progress([experiment1, experiment2], PlotType.VIOLIN, normalize=False)
