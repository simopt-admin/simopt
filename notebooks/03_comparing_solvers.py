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
# # 03 · Comparing Solvers Across Problems
#
# Benchmarking runs several solvers on a set of problem instances and summarizes
# their relative performance:
#
# - per problem: progress curves, terminal distributions, AUC;
# - across problems: solvability profiles, their differences against a
#   reference solver, and terminal scatter plots.
#
# This notebook uses a small grid of (s, S) inventory problems. The INFORMS paper
# experiments in `archive/notebooks/` used the same workflow with 60 problems.

# %%
import matplotlib.pyplot as plt

from simopt.analysis.auc import Auc
from simopt.analysis.diff import DiffSolvability
from simopt.analysis.progress_curve import ProgressCurve
from simopt.analysis.solvability_profile import SolvabilityProfile
from simopt.analysis.terminal_progress import TerminalProgress
from simopt.analysis.terminal_scatterplot import TerminalScatter
from simopt.experiment.api import SimulationConfig, run
from simopt.experiment.single import ProblemSolver
from simopt.models.sscont import SSContMinCost
from simopt.options import ConfidenceIntervalOptions
from simopt.solvers import ASTRODF, NelderMead, RandomSearch

# %% [markdown]
# ## Configuration

# %%
simulation_config = SimulationConfig(n_mreps=5, n_preps=30, n_preps_x0_xstar=50)
ci_options = ConfidenceIntervalOptions(n_bootstraps=20)
budget = 500

# %% [markdown]
# ## Solvers
#
# The same solver class can appear several times with different factors; give each
# version a distinct `name` so the plots can tell them apart.

# %%
solvers = [
    RandomSearch(name="RNDSRCH_ss=5", fixed_factors={"sample_size": 5}),
    RandomSearch(name="RNDSRCH_ss=20", fixed_factors={"sample_size": 20}),
    ASTRODF(),
    NelderMead(),
]

# %% [markdown]
# ## Problems
#
# Problem variants are created by changing model factors. **Create each problem once
# and share the object across solvers**: `run` groups experiments by problem
# instance, and solvers in the same group share the same $x_0$/$x^*$ normalization.

# %%
problems = [
    SSContMinCost(
        name=f"SSCONT-1_dm={demand_mean}_lm={lead_mean}",
        fixed_factors={"budget": budget},
        model_fixed_factors={"demand_mean": demand_mean, "lead_mean": lead_mean},
    )
    for demand_mean in (50.0, 200.0)
    for lead_mean in (1.0, 6.0)
]

experiments = [
    ProblemSolver(solver=solver, problem=problem) for problem in problems for solver in solvers
]
print(f"{len(solvers)} solvers × {len(problems)} problems = {len(experiments)} experiments")

# %%
results = run(experiments, simulation_config)

# %% [markdown]
# ## Per-problem plots
#
# Progress curves, terminal distributions and AUC compare solvers on a **single**
# problem, so select the experiments that share one problem.


# %%
def select(problem):
    """Return the results and experiments that use `problem`."""
    pairs = [(r, e) for r, e in zip(results, experiments, strict=True) if e.problem is problem]
    return [r for r, _ in pairs], [e for _, e in pairs]


inputs, exps = select(problems[0])
print(problems[0].name)
ProgressCurve(agg="mean", ci_options=ci_options).plot(inputs, exps)
TerminalProgress(plot_type="violin").plot(inputs, exps)
Auc(ci_options=ci_options).plot(inputs, exps)
plt.show()

# %% [markdown]
# ## Cross-problem plots
#
# These take every experiment at once and aggregate by solver name.
#
# - **CDF solvability profile** – average, over problems, of the fraction of
#   macroreplications that solved the problem to within `solve_tolerance` by each budget.
# - **Quantile solvability profile** – fraction of problems whose `beta`-quantile
#   solve time is below each budget.

# %%
SolvabilityProfile(plot_type="cdf", solve_tolerance=0.1, ci_options=ci_options).plot(
    results, experiments
)
SolvabilityProfile(plot_type="quantile", solve_tolerance=0.1, beta=0.5, ci_options=ci_options).plot(
    results, experiments
)
plt.show()

# %% [markdown]
# Difference profiles subtract a reference solver's profile, making it easy to see
# where each solver is ahead (above zero) or behind.

# %%
DiffSolvability(plot_type="cdf", reference_solver="ASTRODF", ci_options=ci_options).plot(
    results, experiments
)
plt.show()

# %% [markdown]
# The terminal scatter plot places each (solver, problem) pair by the mean and
# standard deviation of its terminal normalized objective.

# %%
TerminalScatter().plot(results, experiments)
plt.show()
