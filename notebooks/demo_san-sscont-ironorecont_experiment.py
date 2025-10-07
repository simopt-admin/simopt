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
# # Demo for an Experiment with SAN, SSCONT, and IRONORECONT problems.
#
# This script is intended to help with a large experiment with
# 5 solvers (two versions of random search, ASTRO-DF, STRONG, and Nelder-Mead)
# and 60 problems (20 unique instances of problems from (s, S) inventory,
# iron ore, and stochastic activity network).
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
    plot_solvability_cdfs,
    plot_solvability_profiles,
    plot_terminal_progress,
    plot_terminal_scatterplots,
)
from simopt.experimental import group_experiments_by_solver, run_experiment

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
# Problem 1 - SAN
# Configure the problem
all_random_costs = [
    (1, 2, 2, 7, 17, 7, 2, 13, 1, 9, 18, 16, 7),
    (2, 1, 10, 13, 15, 13, 12, 9, 12, 15, 5, 8, 10),
    (2, 6, 7, 11, 13, 5, 1, 2, 2, 3, 15, 16, 13),
    (3, 4, 18, 8, 10, 17, 14, 19, 15, 15, 7, 10, 6),
    (3, 6, 9, 15, 1, 19, 1, 13, 2, 19, 6, 7, 14),
    (4, 4, 2, 4, 5, 3, 19, 4, 17, 5, 16, 8, 8),
    (5, 14, 14, 7, 10, 14, 16, 16, 8, 7, 14, 11, 17),
    (7, 9, 17, 19, 1, 7, 4, 3, 9, 9, 13, 17, 14),
    (8, 14, 1, 10, 18, 10, 17, 1, 2, 11, 1, 16, 6),
    (8, 17, 5, 17, 4, 14, 2, 5, 5, 5, 8, 8, 16),
    (10, 3, 2, 7, 15, 12, 7, 9, 12, 17, 9, 1, 2),
    (10, 5, 17, 12, 13, 14, 6, 5, 19, 17, 1, 7, 17),
    (10, 16, 10, 13, 9, 1, 1, 16, 5, 7, 7, 12, 15),
    (11, 5, 15, 13, 15, 17, 12, 12, 16, 11, 18, 19, 2),
    (12, 11, 13, 4, 15, 11, 16, 2, 7, 7, 13, 8, 3),
    (13, 3, 14, 2, 15, 18, 17, 13, 5, 17, 17, 5, 18),
    (14, 8, 8, 14, 8, 8, 18, 16, 8, 18, 12, 6, 7),
    (14, 18, 7, 8, 13, 17, 10, 17, 19, 1, 13, 6, 12),
    (15, 1, 2, 6, 14, 18, 11, 19, 15, 18, 15, 1, 4),
    (18, 4, 19, 2, 13, 11, 9, 2, 17, 18, 11, 7, 14),
]
num_problems = len(all_random_costs)

# Create all the problem variants.
SAN_problems = [
    {
        "name": "SAN-1",
        "rename": f"SAN-1_rc={costs}",
        "fixed_factors": {"budget": 10000, "arc_costs": costs},
    }
    for costs in all_random_costs
]

# Run the experiment for SAN problems.
SAN_experiments = run_experiment(
    SAN_problems, solvers, num_macroreps, num_postreps, num_postnorms
)

# %%
# Problem 2 - SSCONT
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
    SSCONT_problems, solvers, num_macroreps, num_postreps, num_postnorms
)

# %%
# Problem 3 - IRONORECONT
# Configure the problem
st_devs = [1, 2, 3, 4, 5]
holding_costs = [1, 100]
inven_stops = [1000, 10000]

# Create all the problem variants.
IRONORECONT_problems = [
    {
        "name": "IRONORECONT-1",
        "rename": f"IRONORECONT-1_sd={sd}_hc={hc}_inv={inv}",
        "fixed_factors": {"budget": 1000},
        "model_fixed_factors": {
            "st_dev": sd,
            "holding_cost": hc,
            "inven_stop": inv,
        },
    }
    for sd in st_devs
    for hc in holding_costs
    for inv in inven_stops
]

# Run the experiment for IRONORECONT problems.
IRONORECONT_experiments = run_experiment(
    IRONORECONT_problems, solvers, num_macroreps, num_postreps, num_postnorms
)

# %%
# Combine the experiments into a list of lists, where the outer list
# contains all the experiments for a single solver, and the inner lists
# contain the ProblemSolver instances associated with that solver.
all_experiments = []
all_experiments.extend(SAN_experiments)
all_experiments.extend(SSCONT_experiments)
all_experiments.extend(IRONORECONT_experiments)
experiments = group_experiments_by_solver(all_experiments)


# %% [markdown]
# ## Plotting Settings
#
# Define the plotting settings for the experiments.

# %%
enable_confidence_intervals = True
alpha = 0.2

# %%
plot_solvability_profiles(
    experiments,
    plot_type=PlotType.CDF_SOLVABILITY,
    solve_tol=alpha,
    all_in_one=True,
    plot_conf_ints=enable_confidence_intervals,
    print_max_hw=enable_confidence_intervals,
)

# %%
plot_solvability_profiles(
    experiments,
    plot_type=PlotType.QUANTILE_SOLVABILITY,
    solve_tol=alpha,
    beta=0.5,
    all_in_one=True,
    plot_conf_ints=enable_confidence_intervals,
    print_max_hw=enable_confidence_intervals,
)

# %%
plot_solvability_profiles(
    experiments=experiments,
    plot_type=PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
    solve_tol=alpha,
    ref_solver="ASTRODF",
    all_in_one=True,
    plot_conf_ints=enable_confidence_intervals,
    print_max_hw=enable_confidence_intervals,
)

# %%
plot_solvability_profiles(
    experiments=experiments,
    plot_type=PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
    solve_tol=alpha,
    beta=0.5,
    ref_solver="ASTRODF",
    all_in_one=True,
    plot_conf_ints=enable_confidence_intervals,
    print_max_hw=enable_confidence_intervals,
)

# %%
plot_area_scatterplots(
    experiments,
    all_in_one=True,
    plot_conf_ints=enable_confidence_intervals,
    print_max_hw=enable_confidence_intervals,
)

# %%
plot_terminal_scatterplots(experiments, all_in_one=True)

# %%
n_solvers = len(solvers)

for i in range(len(experiments[0])):
    plot_progress_curves(
        [experiments[solver_idx][i] for solver_idx in range(n_solvers)],
        plot_type=PlotType.MEAN,
        all_in_one=True,
        plot_conf_ints=enable_confidence_intervals,
        print_max_hw=True,
    )
    plot_terminal_progress(
        [experiments[solver_idx][i] for solver_idx in range(n_solvers)],
        plot_type=PlotType.VIOLIN,
        normalize=True,
        all_in_one=True,
    )
    # plot_solvability_cdfs(
    #     [experiments[solver_idx][i] for solver_idx in range(n_solvers)],
    #     solve_tol=0.2,
    #     all_in_one=True,
    #     plot_CIs=True,
    #     print_max_hw=True,
    # )

# %%
# Plots for mu_D = 400 and mu_L = 6 (appreared in the paper)
plot_progress_curves(
    [experiments[solver_idx][0] for solver_idx in range(n_solvers)],
    plot_type=PlotType.ALL,
    all_in_one=True,
)

plot_progress_curves(
    [experiments[solver_idx][0] for solver_idx in range(3, 4)],
    plot_type=PlotType.ALL,
    all_in_one=True,
    normalize=False,
)

plot_progress_curves(
    [experiments[solver_idx][0] for solver_idx in range(n_solvers)],
    plot_type=PlotType.ALL,
    all_in_one=True,
    plot_conf_ints=True,
    print_max_hw=False,
    normalize=True,
)

plot_solvability_cdfs(
    experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
    solve_tol=0.2,
    all_in_one=True,
    plot_conf_ints=True,
    print_max_hw=False,
)

plot_terminal_progress(
    [experiments[solver_idx][0] for solver_idx in range(n_solvers)],
    plot_type=PlotType.VIOLIN,
    normalize=False,
    all_in_one=True,
)
