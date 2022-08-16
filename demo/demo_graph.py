"""
This script runs three versions of random search, plus ASTRO-DF, on 25
versions of the (s, S) inventory problem.
Produces plots appearing in the INFORMS Journal on Computing submission.
"""

import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from experiment_base import Experiment, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles

# Default values of the (s, S) model:
# "demand_mean": 100.0
# "lead_mean": 6.0
# "backorder_cost": 4.0
# "holding_cost": 1.0
# "fixed_cost": 36.0
# "variable_cost": 2.0

# Create 25 problem instances by varying two factors, five levels.
# demand_means = [25.0, 50.0, 100.0, 200.0, 400.0]
# lead_means = [1.0, 3.0, 6.0, 9.0, 12.0]

# Three versions of random search with varying sample sizes.
# rs_sample_sizes = [10, 50, 100]

# RUNNING AND POST-PROCESSING EXPERIMENTS

# Loop over problem instances.
# problem_fixed_factors = {"budget": 10000}
# problem_rename = f"SSCONT-1_dm={dm}_lm={lm}"

# PROBLEM = "PARAMESTI-1"
# PROBLEM_FULL_NAME = "Parameter Estimation"
# PROBLEM = "CNTNEWS-1"
# PROBLEM_FULL_NAME = "Continuous Newsvendor"
# PROBLEM = "SSCONT-1"
# PROBLEM_FULL_NAME = "(s,S) Inventory"
PROBLEM = "SAN-1"
PROBLEM_FULL_NAME = "Stochastic Activity Network"
# PROBLEM = "MM1-1"
# PROBLEM_FULL_NAME = "MM1MinMeanSojournTime"
# PROBLEM = "DYNAMNEWS-1"
# PROBLEM_FULL_NAME = "Dynamic Newsvendor"
# PROBLEM = "IRONORECONT-1"
# PROBLEM_FULL_NAME = "Continuous Iron Ore"


# Temporarily store experiments on the same problem for post-normalization.
experiments_same_problem = []

# Loop over random search solvers.
# solver_fixed_factors = {"sample_size": rs_ss}
# solver_rename = f"RNDSRCH_ss={rs_ss}"
# Create experiment for the problem-solver pair.
new_experiment = Experiment(solver_name="RNDSRCH",
                            problem_name=PROBLEM
                            )
# Run experiment with M = 10.
new_experiment.run(n_macroreps=10)
# Post replicate experiment with N = 100.
new_experiment.post_replicate(n_postreps=100)
experiments_same_problem.append(new_experiment)

# Setup and run ASTRO-DF.
solver_fixed_factors = {"delta_max": 200.0}
new_experiment = Experiment(solver_name="ASTRODF",
                            problem_name=PROBLEM,
                            solver_fixed_factors=solver_fixed_factors
                            )

# Run experiment with M = 10.
new_experiment.run(n_macroreps=10)
# Post replicate experiment with N = 100.
new_experiment.post_replicate(n_postreps=100)
experiments_same_problem.append(new_experiment)

# NELDMD
new_experiment = Experiment(solver_name="NELDMD",
                            problem_name=PROBLEM
                            )
# Run experiment with M = 10.
new_experiment.run(n_macroreps=10)
# Post replicate experiment with N = 100.
new_experiment.post_replicate(n_postreps=100)
experiments_same_problem.append(new_experiment)

# STRONG
new_experiment = Experiment(solver_name="STRONG",
                            problem_name=PROBLEM
                            )
# Run experiment with M = 10.
new_experiment.run(n_macroreps=10)
# Post replicate experiment with N = 100.
new_experiment.post_replicate(n_postreps=100)
experiments_same_problem.append(new_experiment)


# ADAM
new_experiment = Experiment(solver_name="ADAM",
                            problem_name=PROBLEM
                            )
# Run experiment with M = 10.
new_experiment.run(n_macroreps=10)
# Post replicate experiment with N = 100.
new_experiment.post_replicate(n_postreps=100)
experiments_same_problem.append(new_experiment)

# ALOE
new_experiment = Experiment(solver_name="ALOE",
                            problem_name=PROBLEM
                            )
# Run experiment with M = 10.
new_experiment.run(n_macroreps=10)
# Post replicate experiment with N = 100.
new_experiment.post_replicate(n_postreps=100)
experiments_same_problem.append(new_experiment)

# ADAM without true gradient
new_experiment = Experiment(solver_name="ADAM2",
                            problem_name=PROBLEM
                            )
# Run experiment with M = 10.
new_experiment.run(n_macroreps=10)
# Post replicate experiment with N = 100.
new_experiment.post_replicate(n_postreps=100)
experiments_same_problem.append(new_experiment)

# ALOE without true gradient
new_experiment = Experiment(solver_name="ALOE2",
                            problem_name=PROBLEM
                            )
# Run experiment with M = 10.
new_experiment.run(n_macroreps=10)
# Post replicate experiment with N = 100.
new_experiment.post_replicate(n_postreps=100)
experiments_same_problem.append(new_experiment)

# Post-normalize experiments with L = 200.
# Provide NO proxies for f(x0), f(x*), or f(x).
post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=200)

# LOAD DATA FROM .PICKLE FILES TO PREPARE FOR PLOTTING.

# For plotting, "experiments" will be a list of list of Experiment objects.
#   outer list - indexed by solver
#   inner list - index by problem
experiments = []

# Load .pickle files of past results.
# Load all experiments for a given solver, for all solvers.
solver_rename = "RNDSRCH"
experiments_same_solver = []
problem_rename = PROBLEM
file_name = f"{solver_rename}_on_{problem_rename}"
# Load experiment.
new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
# Rename problem and solver to produce nicer plot labels.
new_experiment.solver.name = "Random Search"
new_experiment.problem.name = PROBLEM_FULL_NAME
experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

# Load ASTRO-DF results.
solver_rename = "ASTRODF"
experiments_same_solver = []
problem_rename = PROBLEM
file_name = f"{solver_rename}_on_{problem_rename}"
# Load experiment.
new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
# Rename problem and solver to produce nicer plot labels.
new_experiment.solver.name = "ASTRO-DF"
new_experiment.problem.name = PROBLEM_FULL_NAME
experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

# Load NELDMD results.
solver_rename = "NELDMD"
experiments_same_solver = []
problem_rename = PROBLEM
file_name = f"{solver_rename}_on_{problem_rename}"
# Load experiment.
new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
# Rename problem and solver to produce nicer plot labels.
new_experiment.solver.name = "Nelder-Mead"
new_experiment.problem.name = PROBLEM_FULL_NAME
experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

# Load STRONG results.
solver_rename = "STRONG"
experiments_same_solver = []
problem_rename = PROBLEM
file_name = f"{solver_rename}_on_{problem_rename}"
# Load experiment.
new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
# Rename problem and solver to produce nicer plot labels.
new_experiment.solver.name = "STRONG"
new_experiment.problem.name = PROBLEM_FULL_NAME
experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

# Load ADAM results.
solver_rename = "ADAM"
experiments_same_solver = []
problem_rename = PROBLEM
file_name = f"{solver_rename}_on_{problem_rename}"
# Load experiment.
new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
# Rename problem and solver to produce nicer plot labels.
new_experiment.solver.name = "Adam"
new_experiment.problem.name = PROBLEM_FULL_NAME
experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

# Load ALOE results.
solver_rename = "ALOE"
experiments_same_solver = []
problem_rename = PROBLEM
file_name = f"{solver_rename}_on_{problem_rename}"
# Load experiment.
new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
# Rename problem and solver to produce nicer plot labels.
new_experiment.solver.name = "ALOE"
new_experiment.problem.name = PROBLEM_FULL_NAME
experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

# Load ADAM without true gradient results.
solver_rename = "ADAM2"
experiments_same_solver = []
problem_rename = PROBLEM
file_name = f"{solver_rename}_on_{problem_rename}"
# Load experiment.
new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
# Rename problem and solver to produce nicer plot labels.
new_experiment.solver.name = "Adam w/o IPA grad"
new_experiment.problem.name = PROBLEM_FULL_NAME
experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

# Load ALOE without true gradient results.
solver_rename = "ALOE2"
experiments_same_solver = []
problem_rename = PROBLEM
file_name = f"{solver_rename}_on_{problem_rename}"
# Load experiment.
new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
# Rename problem and solver to produce nicer plot labels.
new_experiment.solver.name = "ALOE w/o IPA grad"
new_experiment.problem.name = PROBLEM_FULL_NAME
experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)


# PLOTTING

n_solvers = len(experiments)
n_problems = len(experiments[0])

# All progress curves for one experiment. Problem instance 0.
plot_progress_curves([experiments[solver_idx][0] for solver_idx in range(n_solvers)], plot_type="all", all_in_one=True)

# # All progress curves for one experiment. Problem instance 22.
# plot_progress_curves([experiments[solver_idx][22] for solver_idx in range(n_solvers)], plot_type="all", all_in_one=True)

# Mean progress curves from all solvers on one problem. Problem instance 0.
plot_progress_curves(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
                     plot_type="mean",
                     all_in_one=True,
                     plot_CIs=True,
                     print_max_hw=False,
                     normalize=False
                     )

# # Mean progress curves from all solvers on one problem. Problem instance 22.
# plot_progress_curves(experiments=[experiments[solver_idx][22] for solver_idx in range(n_solvers)],
#                      plot_type="mean",
#                      all_in_one=True,
#                      plot_CIs=True,
#                      print_max_hw=False
#                      )

# Plot 0.9-quantile progress curves from all solvers on one problem. Problem instance 0.
plot_progress_curves(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
                     plot_type="quantile",
                     beta=0.9,
                     all_in_one=True,
                     plot_CIs=True,
                     print_max_hw=False,
                     normalize=False
                     )

# # Plot 0.9-quantile progress curves from all solvers on one problem. Problem instance 22.
# plot_progress_curves(experiments=[experiments[solver_idx][22] for solver_idx in range(n_solvers)],
#                      plot_type="quantile",
#                      beta=0.9,
#                      all_in_one=True,
#                      plot_CIs=True,
#                      print_max_hw=False
#                      )

# # Plot cdf of 0.2-solve times for all solvers on one problem. Problem instance 0.
# plot_solvability_cdfs(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
#                       solve_tol=0.2,
#                       all_in_one=True,
#                       plot_CIs=True,
#                       print_max_hw=False
#                       )

# # Plot cdf of 0.2-solve times for all solvers on one problem. Problem instance 22.
# plot_solvability_cdfs(experiments=[experiments[solver_idx][22] for solver_idx in range(n_solvers)],
#                       solve_tol=0.2,
#                       all_in_one=True,
#                       plot_CIs=True,
#                       print_max_hw=False
#                       )

# # Plot area scatterplots of all solvers on all problems.
# plot_area_scatterplots(experiments=experiments,
#                        all_in_one=True,
#                        plot_CIs=False,
#                        print_max_hw=False
#                        )

# # Plot cdf 0.1-solvability profiles of all solvers on all problems.
# plot_solvability_profiles(experiments=experiments,
#                           plot_type="cdf_solvability",
#                           all_in_one=True,
#                           plot_CIs=True,
#                           print_max_hw=False,
#                           solve_tol=0.1
#                           )

# # Plot 0.5-quantile 0.1-solvability profiles of all solvers on all problems.
# plot_solvability_profiles(experiments=experiments,
#                           plot_type="quantile_solvability",
#                           all_in_one=True,
#                           plot_CIs=True,
#                           print_max_hw=False,
#                           solve_tol=0.1,
#                           beta=0.5
#                           )

# Plot difference of cdf 0.1-solvability profiles of all solvers on all problems.
# Reference solver = ASTRO-DF.
# plot_solvability_profiles(experiments=experiments,
#                           plot_type="diff_cdf_solvability",
#                           all_in_one=True,
#                           plot_CIs=True,
#                           print_max_hw=False,
#                           solve_tol=0.1,
#                           ref_solver="ASTRO-DF"
#                           )

# # Plot difference of 0.5-quantile 0.1-solvability profiles of all solvers on all problems.
# # Reference solver = ASTRO-DF.
# plot_solvability_profiles(experiments=experiments,
#                           plot_type="diff_quantile_solvability",
#                           all_in_one=True,
#                           plot_CIs=True,
#                           print_max_hw=False,
#                           solve_tol=0.1,
#                           beta=0.5,
#                           ref_solver="ASTRO-DF"
#                           )
