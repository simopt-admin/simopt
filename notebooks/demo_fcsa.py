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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys

sys.path.append("..")

# %%
from simopt.experiment_base import (
    PlotType,
    ProblemSolver,
    plot_feasibility_progress,
    plot_progress_curves,
    plot_terminal_feasibility,
    plot_terminal_progress,
    post_normalize,
)
from simopt.models.san import SANLongestPathStochastic
from simopt.solvers.fcsa import FCSA

# %%
fixed_factors = {
    "constraint_nodes": [6, 8],  # nodes with stochastic constraints
    "length_to_node_constraint": [5, 5],  # max expected length to each constraint node
    "initial_solution": (5,) * 13,
    "budget": 10000,
}
problem = SANLongestPathStochastic(fixed_factors=fixed_factors)

# %%
csa = FCSA(
    fixed_factors={
        "search_direction": "CSA",
        "normalize_grads": False,
        "report_all_solns": True,
        "crn_across_solns": False,
    },
    name="CSA",
)
csa_n = FCSA(
    fixed_factors={
        "search_direction": "CSA",
        "normalize_grads": True,
        "report_all_solns": True,
        "crn_across_solns": False,
    },
    name="CSA-N",
)
fcsa = FCSA(
    fixed_factors={
        "search_direction": "FCSA",
        "normalize_grads": True,
        "report_all_solns": True,
        "crn_across_solns": False,
    },
    name="FCSA",
)


# %%
def run_experiment(solver, problem, n_macroreps, n_postreps):
    experiment = ProblemSolver(solver=solver, problem=problem)
    experiment.run(n_macroreps=n_macroreps)
    experiment.post_replicate(n_postreps=n_postreps)
    return experiment


# %%
n_macroreps = 10
n_postreps = 100
experiments = [
    run_experiment(solver, problem, n_macroreps, n_postreps)
    for solver in [csa, csa_n, fcsa]
]
experiment1, experiment2, experiment3 = experiments
post_normalize(experiments, n_postreps)

# %%
plot_terminal_progress([experiment1], PlotType.VIOLIN, normalize=False)

# %%
plot_terminal_feasibility(
    [[experiment1]], PlotType.FEASIBILITY_VIOLIN, all_in_one=True, two_sided=True
)

# %%
plot_progress_curves([experiment1], PlotType.ALL, normalize=False)

# %%
plot_feasibility_progress(
    [[experiment1]], PlotType.ALL_FEASIBILITY_PROGRESS, print_max_hw=False
)

# %%
plot_progress_curves([experiment2, experiment3], PlotType.ALL, normalize=False)

# %%
plot_feasibility_progress(
    [[experiment2], [experiment3]],
    PlotType.ALL_FEASIBILITY_PROGRESS,
    print_max_hw=False,
)

# %%
csa_n = FCSA(
    fixed_factors={
        "search_direction": "CSA",
        "normalize_grads": True,
        "report_all_solns": False,
        "crn_across_solns": False,
    },
    name="CSA-N",
)
fcsa = FCSA(
    fixed_factors={
        "search_direction": "FCSA",
        "normalize_grads": True,
        "report_all_solns": False,
        "crn_across_solns": False,
    },
    name="FCSA",
)

# %%
experiments = [
    run_experiment(solver, problem, n_macroreps, n_postreps) for solver in [csa_n, fcsa]
]
experiment2, experiment3 = experiments
post_normalize([experiment2, experiment3], 100)

# %%
plot_terminal_feasibility([[experiment2], [experiment3]], PlotType.FEASIBILITY_SCATTER)
