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

# %% [markdown]
# # Run five solvers on 20 versions of the (s, S) inventory problem.
#
# Produces plots appearing in the INFORMS Journal on Computing submission.

# %% [markdown]
# ## Append SimOpt Path
#
# Since the notebook is stored in simopt/notebooks, we need to append the
# parent simopt directory to the system path to import the necessary modules
# later on.

# %%
import sys
from pathlib import Path

# Take the current directory, find the parent, and add it to the system path
sys.path.append(str(Path.cwd().parent))

from simopt.experiment_base import (
    PlotType,
    plot_area_scatterplots,
    plot_progress_curves,
    plot_solvability_profiles,
    plot_terminal_progress,
    plot_terminal_scatterplots,
)
from simopt.experimental import run_experiment

# %% [markdown]
# ## Experiment Configuration Parameters
#
# Define the options used in the experiments.

# %%
# Experiment Configuration
num_macroreps = 10
num_postreps = 100
num_postnorms = 200


# %% [markdown]
# ## Problem/Solver Configuration Parameters
#
# Define the problems and solvers used in the experiments.

# %%
# Solvers to use in the experiment.
# Includes two versions of random search with varying sample sizes.
# The rename will be used in the plots to differentiate them.
solvers = [
    {
        "name": "RNDSRCH",
        "rename": "RNDSRCH_ss=10",
        "fixed_factors": {"sample_size": 10},
    },
    {
        "name": "RNDSRCH",
        "rename": "RNDSRCH_ss=50",
        "fixed_factors": {"sample_size": 50},
    },
    {"name": "ASTRODF"},
    {"name": "NELDMD"},
    {"name": "STRONG"},
]

# %%
# Configure the problem
demand_means = [25.0, 50.0, 100.0, 200.0, 400.0]
lead_means = [1.0, 3.0, 6.0, 9.0]

# Create all the problem variants.
SSCONT_problems = [
    {
        "name": "SSCONT-1",
        "rename": f"SSCONT-1_dm={dm}_lm={lm}",
        "fixed_factors": {"budget": 1000},
        "model_fixed_factors": {
            "demand_mean": dm,
            "lead_mean": lm,
        },
    }
    for dm in demand_means
    for lm in lead_means
]

# Run the experiment for SSCONT problems.
SSCONT_experiments = run_experiment(
    problems=SSCONT_problems,
    solvers=solvers,
    num_macroreps=num_macroreps,
    num_postreps=num_postreps,
    num_postnorms=num_postnorms,
)

# %%
# Combine the experiments into a list of lists, where the outer list
# contains all the experiments for a single solver, and the inner lists
# contain the ProblemSolver instances associated with that solver.

experiment_dict = {}
for exp_problem_list in SSCONT_experiments:
    for experiment in exp_problem_list:
        # Use the solver name as the key and append the ProblemSolver instance.
        key = experiment.solver.name
        if key not in experiment_dict:
            experiment_dict[key] = []
        experiment_dict[key].append(experiment)
# Turn the dictionary into a list of lists.
experiments = list(experiment_dict.values())

# %% [markdown]
# ## Plotting Settings
#
# Define the plotting settings for the experiments.

# %%
enable_confidence_intervals = True
alpha = 0.2

# %%
plot_area_scatterplots(
    experiments, all_in_one=True, plot_conf_ints=True, print_max_hw=True
)

# %%
plot_solvability_profiles(
    experiments,
    plot_type=PlotType.CDF_SOLVABILITY,
    solve_tol=0.1,
    all_in_one=True,
    plot_conf_ints=True,
    print_max_hw=True,
)

# %%
plot_solvability_profiles(
    experiments,
    plot_type=PlotType.QUANTILE_SOLVABILITY,
    solve_tol=0.1,
    beta=0.5,
    all_in_one=True,
    plot_conf_ints=True,
    print_max_hw=True,
)

# %%
plot_solvability_profiles(
    experiments=experiments,
    plot_type=PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
    solve_tol=0.1,
    ref_solver="ASTRODF",
    all_in_one=True,
    plot_conf_ints=True,
    print_max_hw=True,
)

# %%
plot_solvability_profiles(
    experiments=experiments,
    plot_type=PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
    solve_tol=0.1,
    beta=0.5,
    ref_solver="ASTRODF",
    all_in_one=True,
    plot_conf_ints=True,
    print_max_hw=True,
)

# %%
plot_terminal_scatterplots(experiments, all_in_one=True)

# %%
n_problems = len(SSCONT_problems)
n_solvers = len(experiments)

for i in range(n_problems):
    plot_progress_curves(
        [experiments[solver_idx][i] for solver_idx in range(n_solvers)],
        plot_type=PlotType.MEAN,
        all_in_one=True,
        plot_conf_ints=True,
        print_max_hw=True,
        normalize=False,
    )
    plot_terminal_progress(
        [experiments[solver_idx][i] for solver_idx in range(n_solvers)],
        plot_type=PlotType.VIOLIN,
        normalize=True,
        all_in_one=True,
    )
