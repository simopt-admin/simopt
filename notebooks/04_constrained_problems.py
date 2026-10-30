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
# # 04 · Problems with Stochastic Constraints
#
# Some problems impose constraints of the form $E[g(x, \xi)] \le 0$ that can only be
# estimated by simulation. For these, the objective alone is not enough to judge a
# solver: we also need to know how (in)feasible its recommendations are.
#
# This notebook runs three variants of the FCSA solver on the stochastic activity
# network problem `SAN-2`, which bounds the expected path length to two nodes.

# %%
import matplotlib.pyplot as plt

from simopt.analysis.feasibility_progress import FeasibilityProgress
from simopt.analysis.progress_curve import ProgressCurve
from simopt.analysis.terminal_feasibility import TerminalFeasibility
from simopt.analysis.terminal_progress import TerminalProgress
from simopt.experiment.api import SimulationConfig, run
from simopt.experiment.single import ProblemSolver
from simopt.models.san import SANLongestPathStochastic
from simopt.options import ConfidenceIntervalOptions
from simopt.solvers.fcsa import FCSA

# %% [markdown]
# ## Configuration

# %%
simulation_config = SimulationConfig(n_mreps=5, n_preps=20, n_preps_x0_xstar=50)
ci_options = ConfidenceIntervalOptions(n_bootstraps=20)

problem = SANLongestPathStochastic(
    fixed_factors={
        "constraint_nodes": [6, 8],  # nodes with stochastic constraints
        "length_to_node_constraint": [5, 5],  # max expected path length to each node
        "initial_solution": (5,) * 13,
        "budget": 10000,
    }
)
print(problem.name, "| stochastic constraints:", problem.n_stochastic_constraints)

# %% [markdown]
# ## Solver variants
#
# FCSA can use the plain cooperative stochastic approximation direction (`CSA`), the
# same with normalized gradients (`CSA-N`), or its own feasibility-aware direction
# (`FCSA`). With `report_all_solns=True` every incumbent is reported, so the progress
# plots show the full search path rather than only accepted recommendations.


# %%
def fcsa(name, search_direction, normalize_grads, report_all_solns):
    """Build an FCSA solver variant."""
    return FCSA(
        name=name,
        fixed_factors={
            "search_direction": search_direction,
            "normalize_grads": normalize_grads,
            "report_all_solns": report_all_solns,
            "crn_across_solns": False,
        },
    )


solvers = [
    fcsa("CSA", "CSA", normalize_grads=False, report_all_solns=True),
    fcsa("CSA-N", "CSA", normalize_grads=True, report_all_solns=True),
    fcsa("FCSA", "FCSA", normalize_grads=True, report_all_solns=True),
]
experiments = [ProblemSolver(solver=solver, problem=problem) for solver in solvers]
results = run(experiments, simulation_config)

# %% [markdown]
# ## Objective progress
#
# Normalization is less meaningful when $x^*$ may be infeasible, so plot raw objectives.

# %%
ProgressCurve(agg="all", normalize=False).plot(results, experiments)
TerminalProgress(plot_type="violin", normalize=False).plot(results, experiments)
plt.show()

# %% [markdown]
# ## Feasibility
#
# The feasibility score summarizes the constraint values: with `score_type="inf_norm"`
# it is $-\max_i E[\text{LHS}_i]$, so **negative means infeasible**. Two-sided scores
# also reward slack (positive when strictly feasible); one-sided scores are 0 for any
# feasible solution and support mean/quantile aggregation with confidence intervals.

# %%
FeasibilityProgress(agg="all", two_sided=True).plot(results, experiments)
FeasibilityProgress(agg="mean", two_sided=False, ci_options=ci_options).plot(results, experiments)
plt.show()

# %% [markdown]
# Terminal feasibility pairs each macroreplication's final objective with its final
# feasibility score, as a violin per solver or as an objective-vs-feasibility scatter.

# %%
TerminalFeasibility(plot_type="violin", ci_options=None).plot(results, experiments)
TerminalFeasibility(plot_type="scatter", ci_options=ci_options).plot(results, experiments)
plt.show()
