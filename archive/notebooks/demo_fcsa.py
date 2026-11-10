# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo for an experiment with FCSA on the SAN problem
# This script is intented to demonstrate an experiment with three different versions of the FCSA solver on the SAN problem.

# %% [markdown]
# ## Append SimOpt Path
#
# Since the notebook is stored in simopt/notebooks, we need to append the
# parent simopt directory to the system path to import the necessary modules
# later on.

# %%
import sys

sys.path.append("..")

# %%
from simopt.experiment_base import PlotType, ProblemSolver, post_normalize
from simopt.analysis.feasibility_progress import FeasibilityProgress
from simopt.analysis.progress_curve import ProgressCurve
from simopt.analysis.terminal_feasibility import TerminalFeasibility
from simopt.analysis.terminal_progress import TerminalProgress
from simopt.experiment.plot import save_plots
from simopt.options import ConfidenceIntervalOptions
from simopt.models.san import SANLongestPathStochastic
from simopt.solvers import FCSA

# %% [markdown]
# ## Experiment Configuration Parameters
#
# Configure 3 versions of the solver: CSA, CSA-N, and FCSA and set problem configuration. Set report_all_solutions = True meaning all incumbent solutions will be reported.

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
    run_experiment(solver, problem, n_macroreps, n_postreps) for solver in [csa, csa_n, fcsa]
]
experiment1, experiment2, experiment3 = experiments
post_normalize(experiments, n_postreps)

# %% [markdown]
# ## Plotting Settings
#
# Define the plotting settings for the experiments. Plot terminal objective progress, terminal feasibility progress, objective progress curve, and feasiblity progress curve for all incumbent solutions.

# %%
save_plots(
    [experiment1],
    TerminalProgress(plot_type=PlotType.VIOLIN.value, normalize=False),
    use_experiment_crn=True,
)

# %%
save_plots(
    [[experiment1]],
    TerminalFeasibility(
        plot_type=PlotType.FEASIBILITY_VIOLIN.value.removeprefix("feasibility_"),
        score_type="inf_norm",
        norm_degree=1,
        two_sided=True,
        ci_options=ConfidenceIntervalOptions(
            n_bootstraps=100, confidence_level=0.95, bias_correction=True
        ),
    ),
    all_in_one=True,
    plot_optimal=True,
    use_experiment_crn=True,
)

# %%
save_plots(
    [experiment1],
    ProgressCurve(
        agg=PlotType.ALL.value,
        normalize=False,
        beta=0.5,
        ci_options=ConfidenceIntervalOptions(
            n_bootstraps=100, confidence_level=0.95, bias_correction=True
        ),
    ),
    print_max_hw=True,
    use_experiment_crn=True,
)

# %%
save_plots(
    [[experiment1]],
    FeasibilityProgress(
        agg=PlotType.ALL_FEASIBILITY_PROGRESS.value.removesuffix("_feasibility_progress"),
        beta=0.5,
        score_type="inf_norm",
        norm_degree=1,
        two_sided=True,
        ci_options=ConfidenceIntervalOptions(
            n_bootstraps=100, confidence_level=0.95, bias_correction=True
        ),
    ),
    print_max_hw=False,
    use_experiment_crn=True,
)

# %%
save_plots(
    [experiment2, experiment3],
    ProgressCurve(
        agg=PlotType.ALL.value,
        normalize=False,
        beta=0.5,
        ci_options=ConfidenceIntervalOptions(
            n_bootstraps=100, confidence_level=0.95, bias_correction=True
        ),
    ),
    print_max_hw=True,
    use_experiment_crn=True,
)

# %%
save_plots(
    [[experiment2], [experiment3]],
    FeasibilityProgress(
        agg=PlotType.ALL_FEASIBILITY_PROGRESS.value.removesuffix("_feasibility_progress"),
        beta=0.5,
        score_type="inf_norm",
        norm_degree=1,
        two_sided=True,
        ci_options=ConfidenceIntervalOptions(
            n_bootstraps=100, confidence_level=0.95, bias_correction=True
        ),
    ),
    print_max_hw=False,
    use_experiment_crn=True,
)

# %% [markdown]
# ## Experiment Configuration Parameters
#
# Configure 2 versions of the solver: CSA-N, and FCSA and set problem configuration. Set report_all_solutions = False meaning only recommended solutions will be reported.

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
experiments = [run_experiment(solver, problem, n_macroreps, n_postreps) for solver in [csa_n, fcsa]]
experiment2, experiment3 = experiments
post_normalize([experiment2, experiment3], 100)

# %% [markdown]
# ## Plotting Settings
#
# Define the plotting settings for the experiments. Plot terminal objective vs feasibility scatter plot for recommended solutions.

# %%
save_plots(
    [[experiment2], [experiment3]],
    TerminalFeasibility(
        plot_type=PlotType.FEASIBILITY_SCATTER.value.removeprefix("feasibility_"),
        score_type="inf_norm",
        norm_degree=1,
        two_sided=True,
        ci_options=ConfidenceIntervalOptions(
            n_bootstraps=100, confidence_level=0.95, bias_correction=True
        ),
    ),
    plot_optimal=True,
    use_experiment_crn=True,
)
