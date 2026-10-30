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
# # 06 · Customizing Experiments
#
# 1. Replace a model's input distribution with data (data-driven simulation).
# 2. Run a solver for discrete problems (DASSO) against random search.
# 3. Supply a known optimum so normalization does not have to estimate it.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simopt.analysis.progress_curve import ProgressCurve
from simopt.analysis.terminal_progress import TerminalProgress
from simopt.experiment.api import ProxyValues, SimulationConfig, run
from simopt.experiment.single import ProblemSolver
from simopt.input_models import InputModel
from simopt.models.cntnv import CntNVMaxProfit
from simopt.models.example import Example2Problem
from simopt.options import ConfidenceIntervalOptions
from simopt.solvers.astrodf import ASTRODF
from simopt.solvers.dasso import DASSO
from simopt.solvers.randomsearch import RandomSearch

simulation_config = SimulationConfig(n_mreps=5, n_preps=50, n_preps_x0_xstar=100)
ci_options = ConfidenceIntervalOptions(n_bootstraps=20)

# %% [markdown]
# ## 1. Data-driven input models
#
# Models draw random inputs through `InputModel` objects. The continuous newsvendor
# samples demand from `model.demand_model`, a Burr distribution by default. Swapping in
# an empirical distribution lets the same problem be optimized against observed data.
#
# Sample with the `rng` that is passed in (not `np.random`): it is an `MRG32k3a`
# stream, which keeps replications reproducible and common random numbers working.


# %%
class EmpiricalDemand(InputModel):
    """Resample demand uniformly from observed data."""

    def __init__(self, data):
        self.data = list(data)

    def random(self, rng, burr_c, burr_k):
        return rng.choice(self.data)


demand_data = np.load(Path("data") / "demand.npy")
print(f"{len(demand_data)} observations, mean demand {demand_data.mean():.2f}")

default_problem = CntNVMaxProfit(name="CNTNEWS-1 (Burr)", fixed_factors={"budget": 500})
data_problem = CntNVMaxProfit(name="CNTNEWS-1 (data)", fixed_factors={"budget": 500})
data_problem.model.demand_model = EmpiricalDemand(demand_data)  # ty: ignore[unresolved-attribute]

solver = ASTRODF()
experiments = [ProblemSolver(solver=solver, problem=p) for p in (default_problem, data_problem)]
results = run(experiments, simulation_config)

for experiment, result in zip(experiments, results, strict=True):
    xstar_value = result.xstar_sample.mean()
    xstar = np.round(result.xstar, 3)
    print(f"{experiment.problem.name:>18}: x* ≈ {xstar}, f(x*) ≈ {xstar_value:.3f}")

# %% [markdown]
# Alternatively, subclass `ProblemSolver` and override the `model_created` hook, which
# runs right after the model is constructed. This keeps the customization with the
# experiment, e.g. when the same setup is reused across many experiments.


# %%
class DataDrivenNewsvendor(ProblemSolver):
    def model_created(self, model):
        model.demand_model = EmpiricalDemand(demand_data)


experiment = DataDrivenNewsvendor(solver=ASTRODF(), problem=CntNVMaxProfit())
print(type(experiment.problem.model.demand_model).__name__)  # ty: ignore[unresolved-attribute]

# %% [markdown]
# ## 2. Discrete solvers
#
# `EXAMPLE-2` is a noisy quadratic over the integer lattice $\{-4, \dots, 4\}^4$. DASSO
# is designed for such discrete problems; random search is a baseline.

# %%
problem = Example2Problem(fixed_factors={"budget": 2000})
solvers = [DASSO(), RandomSearch()]
experiments = [ProblemSolver(solver=solver, problem=problem) for solver in solvers]
results = run(experiments, simulation_config)

ProgressCurve(agg="all", normalize=False).plot(results, experiments)
TerminalProgress(plot_type="violin", normalize=False).plot(results, experiments)
plt.show()

# %% [markdown]
# ## 3. Known optimal solutions
#
# Normalization needs $x^*$. Problems that define `optimal_solution` (such as
# `EXAMPLE-2`, where it is `(1, 2, 3, 4)`) use it automatically. Otherwise $x^*$ is
# approximated by the best solution any solver recommended, which can be wrong when
# solvers stop short. If the optimum is known from theory or a long reference run,
# pass it with `ProxyValues`.
#
# For the newsvendor, the optimal order quantity is the critical-fractile quantile of
# demand: $F(x^*) = (p - c)/(p - s)$. With Burr Type XII demand,
# $F(x) = 1 - (1 + x^{c})^{-k}$.

# %%
problem = CntNVMaxProfit(fixed_factors={"budget": 500})
factors = problem.model.factors
critical_ratio = (factors["sales_price"] - factors["purchase_price"]) / (
    factors["sales_price"] - factors["salvage_price"]
)
xstar = ((1 - critical_ratio) ** (-1 / factors["Burr_k"]) - 1) ** (1 / factors["Burr_c"])
print(f"analytic x* = {xstar:.4f}")

experiments = [
    ProblemSolver(solver=solver, problem=problem) for solver in (ASTRODF(), RandomSearch())
]
estimated = run(experiments, simulation_config)
known = run(experiments, simulation_config, proxy_values=ProxyValues(xstar=(xstar,)))
print(f"estimated x* = {estimated[0].xstar[0]:.4f}, known x* = {known[0].xstar[0]:.4f}")

# %% [markdown]
# With the estimated proxy the curves dip below 0: the solvers "beat" a proxy that is
# not actually optimal. With the true $x^*$ the curves settle at 0.

# %%
ProgressCurve(agg="mean", ci_options=ci_options).plot(estimated, experiments)
plt.title("x* estimated from solver output")
ProgressCurve(agg="mean", ci_options=ci_options).plot(known, experiments)
plt.title("x* known")
plt.show()
