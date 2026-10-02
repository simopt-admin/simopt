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
# # 02 · Running a Single Experiment
#
# An **experiment** pairs one solver with one problem (`ProblemSolver`). Running it
# has three stages, all handled by `simopt.experiment.api.run`:
#
# 1. **Macroreplications** – run the solver several times from scratch.
# 2. **Postreplications** – re-evaluate every recommended solution with fresh
#    replications, so the reported objective is not biased by the solver's own noise.
# 3. **Normalization** – estimate the objective at the initial solution $x_0$ and at a
#    proxy optimum $x^*$, so progress can be expressed as a fraction of the gap.
#
# The results come back as `AnalysisInput` objects, which every plot in
# `simopt.analysis` consumes.

# %%
import matplotlib.pyplot as plt

from simopt.analysis.auc import Auc
from simopt.analysis.progress_curve import ProgressCurve
from simopt.analysis.solvability_cdf import SolvabilityCdf
from simopt.analysis.terminal_progress import TerminalProgress
from simopt.experiment.api import SimulationConfig, run
from simopt.experiment.single import ProblemSolver
from simopt.models.cntnv import CntNVMaxProfit
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from simopt.solvers.astrodf import ASTRODF

# %% [markdown]
# ## Configuration
#
# These numbers control the run time. Increase them for publication-quality results.

# %%
simulation_config = SimulationConfig(
    n_mreps=5,  # solver runs
    n_preps=50,  # postreplications per recommended solution
    n_preps_x0_xstar=100,  # postreplications at x0 and x* for normalization
)
crn_options = CrnOptions(across_budget=True, across_macroreps=False, across_x0_xstar=True)
ci_options = ConfidenceIntervalOptions(n_bootstraps=50)

# %% [markdown]
# ## Build and run the experiment
#
# Import the solver and problem classes and customize them with fixed factors.

# %%
experiment = ProblemSolver(
    solver=ASTRODF(),
    problem=CntNVMaxProfit(fixed_factors={"budget": 500}),
)
if message := experiment.check_compatibility():
    print(message)

[result] = run([experiment], simulation_config, crn_options)

# %% [markdown]
# ## What is inside an `AnalysisInput`
#
# - `full_df` – every postreplication of every recommended solution.
# - `mean_df` – postreplications averaged per (macroreplication, step), plus
#   normalized budget and objective.
# - `x0`, `xstar` and their samples – the normalization reference points.

# %%
print("x0 =", result.x0, " x* =", result.xstar, " budget =", result.budget)
print(f"f(x0) ≈ {result.x0_sample.mean():.4f},  f(x*) ≈ {result.xstar_sample.mean():.4f}")
result.mean_df.head()

# %% [markdown]
# ## Progress curves
#
# Every plot is a small configuration object with a `.plot(analysis_inputs, experiments)`
# method. `agg` selects how macroreplications are combined.

# %%
inputs, experiments = [result], [experiment]

ProgressCurve(agg="all", normalize=False).plot(inputs, experiments)
ProgressCurve(agg="mean", normalize=False, ci_options=ci_options).plot(inputs, experiments)
ProgressCurve(agg="quantile", beta=0.9, ci_options=ci_options).plot(inputs, experiments)
plt.show()

# %% [markdown]
# ## Solvability, terminal progress and area under the curve
#
# - **Solvability CDF** – fraction of macroreplications that have closed all but
#   `solve_tolerance` of the optimality gap by a given budget.
# - **Terminal progress** – distribution of the final (normalized) objective.
# - **AUC** – mean and standard deviation of the area under each progress curve.

# %%
SolvabilityCdf(solve_tolerance=0.1, ci_options=ci_options).plot(inputs, experiments)
TerminalProgress(plot_type="box").plot(inputs, experiments)
Auc(ci_options=ci_options).plot(inputs, experiments)
plt.show()
