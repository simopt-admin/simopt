# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 · Models and Problems
#
# SimOpt separates a **model** (a stochastic simulation that maps factors to responses)
# from a **problem** (a model plus an objective, constraints, decision variables and a
# budget). This notebook shows both layers without running any solver.
#
# 1. Run replications of a model directly.
# 2. Evaluate a problem at a fixed solution.
# 3. Inspect stochastic constraints on a constrained problem.
#
# To list every available model, problem and solver, run
# `uv run python scripts/list_directories.py`.

# %%
import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import Solution
from simopt.models.cntnv import CntNVMaxProfit
from simopt.models.facilitysizing import FacilitySizingTotalCost
from simopt.models.mm1queue import MM1Queue

# %% [markdown]
# ## 1. Model: replicate an M/M/1 queue
#
# A model owns `n_rngs` random number streams. Each replication consumes one
# substream from each of them; advancing the substream gives an independent
# replication.

# %%
model = MM1Queue(fixed_factors={"lambda": 3.0, "mu": 8.0})
print(model.name, "factors:", model.factors)

rngs = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(model.n_rngs)]
responses, gradients = model.replicate(model.config, rngs)

for key, value in responses.items():
    print(f"{key:>20}: {value:.4f}")

# %% [markdown]
# Gradients are reported per response with respect to each factor (``nan`` when the
# model does not provide that gradient).

# %%
print({factor: round(g, 4) for factor, g in gradients["avg_sojourn_time"].items()})

# %% [markdown]
# Running several replications only requires advancing the substreams between calls.

# %%
n_reps = 20
sojourn_times = []
for _ in range(n_reps):
    responses, _ = model.replicate(model.config, rngs)
    sojourn_times.append(responses["avg_sojourn_time"])
    for rng in rngs:
        rng.advance_subsubstream()

print(f"mean sojourn time over {n_reps} reps: {np.mean(sojourn_times):.4f}")

# %% [markdown]
# ## 2. Problem: evaluate the continuous newsvendor at a solution
#
# `Problem.simulate` runs replications of the underlying model at a `Solution` and
# stores the objective estimates (and gradients, when available) on the solution.

# %%
problem = CntNVMaxProfit(fixed_factors={"initial_solution": (2,), "budget": 500})
print(problem.name, "| dim:", problem.dim, "| minmax:", problem.minmax)
print("gradient available:", problem.gradient_available)

x = (0.3,)
rngs = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(problem.model.n_rngs)]
solution = Solution(x, rngs)

n_reps = 10
problem.simulate(solution, num_macroreps=n_reps)

print(
    f"objective at x={x}: {solution.objectives_mean[0]:.4f} ± {solution.objectives_stderr[0]:.4f}"
)
print("per-replication objectives:", np.round(solution.objectives[:n_reps, 0], 3))
if problem.gradient_available:
    print(
        "per-replication gradients:", np.round(solution.objectives_gradients[:n_reps, 0], 3).ravel()
    )

# %% [markdown]
# ## 3. Stochastic constraints: facility sizing
#
# Constraints of the form $E[\text{LHS}] \le 0$ are estimated alongside the
# objective. `FACSIZE-1` minimizes capacity cost subject to a chance constraint on
# meeting demand.

# %%
problem = FacilitySizingTotalCost(fixed_factors={"epsilon": 0.1})
print(problem.name, "| stochastic constraints:", problem.n_stochastic_constraints)

n_reps = 30
for capacity in (100, 150):
    x = (capacity,) * 3
    rngs = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(problem.model.n_rngs)]
    solution = Solution(x, rngs)
    problem.simulate(solution, num_macroreps=n_reps)

    print(f"x={x} objective: {solution.objectives_mean[0]:.1f}")
    for i in range(problem.n_stochastic_constraints):
        mean = solution.stoch_constraints_mean[i]
        stderr = solution.stoch_constraints_stderr[i]
        status = "satisfied" if mean <= 0 else "violated"
        print(f"  constraint {i}: E[LHS] = {mean:.4f} ± {stderr:.4f} ({status})")
