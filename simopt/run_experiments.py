from wrapper_base import Experiment, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles
from rng.mrg32k3a import MRG32k3a

# 3 versions of random search
rs_sample_sizes = [10, 50, 100]

# Problem ranges: 5*5 = 25 problem instances
demand_means = [25.0, 50.0, 100.0, 200.0, 400.0] #, 800.0]
lead_means = [1.0, 3.0, 6.0, 9.0, 12.0] #, 15.0]

# default values
# "demand_mean": 100.0
# "lead_mean": 6.0
# "backorder_cost": 4.0
# "holding_cost": 1.0
# "fixed_cost": 36.0
# "variable_cost": 2.0

# # First Section: Running experiments.

# # Loop over problems.
# for dm in demand_means:
#     for lm in lead_means:
#         oracle_fixed_factors = {"demand_mean": dm,
#                                 "lead_mean": lm
#                                 }
#         # Budget = 1000 for (s,S) inventory problem.
#         # RS w/ sample size 100 will get through only 10 iterations.
#         problem_fixed_factors = {"budget": 1000}
#         problem_rename = f"SSCONT-1_dm={dm}_lm={lm}"
#         # Temporarily store experiments on the same problem for post-normalization.
#         # experiments_same_problem = []
#         # Loop over solvers.
#         # for rs_ss in rs_sample_sizes:
#         #     solver_fixed_factors = {"sample_size": rs_ss}
#         #     solver_rename = f"RNDSRCH_ss={rs_ss}"
#         #     # Create experiment.
#         #     new_experiment = Experiment(solver_name="RNDSRCH",
#         #                                 problem_name="SSCONT-1",
#         #                                 solver_rename=solver_rename,
#         #                                 problem_rename=problem_rename,
#         #                                 solver_fixed_factors=solver_fixed_factors,
#         #                                 problem_fixed_factors=problem_fixed_factors,
#         #                                 oracle_fixed_factors=oracle_fixed_factors
#         #                                 )
#         #     # Run experiment with M = 50.
#         #     new_experiment.run(n_macroreps=10)
#         #     # Post replicate experiment with N = 100.
#         #     new_experiment.post_replicate(n_postreps=100)
#         #     experiments_same_problem.append(new_experiment)

#         # Run ASTRO-DF. (COMMENTED OUT)
#         solver_fixed_factors = {"delta_max": 200.0}
#         new_experiment = Experiment(solver_name="ASTRODF",
#                                     problem_name="SSCONT-1",
#                                     problem_rename=problem_rename,
#                                     solver_fixed_factors=solver_fixed_factors,
#                                     problem_fixed_factors=problem_fixed_factors,
#                                     oracle_fixed_factors=oracle_fixed_factors
#                                     )
#         # Run experiment with M = 10.
#         new_experiment.run(n_macroreps=10)
#         # Post replicate experiment with N = 100.
#         new_experiment.post_replicate(n_postreps=100)
#         # experiments_same_problem.append(new_experiment)

# #         # Post-normalize experiments with L = 200.
# #         # Provide NO proxies for f(x0), f(x*), or f(x).
# #         post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=200)

# # STOPPING POINT.
# # If experiments have been run, comment out the First Section.

# Second Section: Plotting.

# For plotting, "experiments" will be a list of list of Experiment objects.
#   outer list - indexed by solver
#   inner list - index by problem
experiments = []

# Load .pickle files of past results.
# TODO: Concatenate file name strings.
# Load all experiments for a given solver, for all solvers.
for rs_ss in rs_sample_sizes:
    solver_rename = f"RNDSRCH_ss={rs_ss}"
    experiments_same_solver = []
    for dm in demand_means:
        for lm in lead_means:
            problem_rename = f"SSCONT-1_dm={dm}_lm={lm}"
            file_name = f"{solver_rename}_on_{problem_rename}"
            # Load experiment.
            new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
            # Rename problem and solver to produce nicer plot labels.
            new_experiment.solver.name = f"Random Search w/ s={rs_ss}"
            new_experiment.problem.name = f"SSCONT-1 with mu_D={round(dm)} and mu_L={round(lm)}"
            experiments_same_solver.append(new_experiment)
    experiments.append(experiments_same_solver)
# Load ASTRO-DF results
solver_rename = f"ASTRODF"
experiments_same_solver = []
for dm in demand_means:
    for lm in lead_means:
        problem_rename = f"SSCONT-1_dm={dm}_lm={lm}"
        file_name = f"{solver_rename}_on_{problem_rename}"
        # Load experiment.
        new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
        # Rename problem and solver to produce nicer plot labels.
        new_experiment.solver.name = "ASTRO-DF"
        new_experiment.problem.name = f"SSCONT-1 with mu_D={round(dm)} and mu_L={round(lm)}"
        #print(new_experiment.problem.name)
        experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)


# Plotting
n_solvers = len(experiments)
n_problems = len(experiments[0])

# # Post-normalize to incorporate ASTRO-DF results
# for problem_idx in range(n_problems):
#     experiments_same_problem = [experiments[solver_idx][problem_idx] for solver_idx in range(n_solvers)]
#     post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=200)

# # All progress curves for one experiment.
# plot_progress_curves([experiments[0][0], experiments[3][0]], plot_type="all", all_in_one=True)

# # All progress curves for one experiment.
# plot_progress_curves([experiments[solver_idx][0] for solver_idx in range(n_solvers)], plot_type="all", all_in_one=True)

# # All progress curves for one experiment.
# plot_progress_curves([experiments[solver_idx][22] for solver_idx in range(n_solvers)], plot_type="all", all_in_one=True)

# # All progress curves for one experiment.
# plot_progress_curves([experiments[0][22], experiments[3][22]], plot_type="all", all_in_one=True)

# # Mean progress curves from all solvers on one problem.
# plot_progress_curves(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
#                      plot_type="mean",
#                      all_in_one=True,
#                      plot_CIs=True,
#                      print_max_hw=False
#                      )

# # Mean progress curves from all solvers on one problem.
# plot_progress_curves(experiments=[experiments[solver_idx][22] for solver_idx in range(n_solvers)],
#                      plot_type="mean",
#                      all_in_one=True,
#                      plot_CIs=True,
#                      print_max_hw=False
#                      )

# # Plot 0.9-quantile progress curves from all solvers on one problem.
# plot_progress_curves(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
#                      plot_type="quantile",
#                      beta=0.9,
#                      all_in_one=True,
#                      plot_CIs=True,
#                      print_max_hw=False
#                      )

# # Plot 0.9-quantile progress curves from all solvers on one problem.
# plot_progress_curves(experiments=[experiments[solver_idx][22] for solver_idx in range(n_solvers)],
#                      plot_type="quantile",
#                      beta=0.9,
#                      all_in_one=True,
#                      plot_CIs=True,
#                      print_max_hw=False
#                      )


# # Plot cdf of 0.2-solve times for all solvers on one problem.
# plot_solvability_cdfs(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
#                       solve_tol=0.2,
#                       all_in_one=True,
#                       plot_CIs=True,
#                       print_max_hw=False
#                       )

# # Plot cdf of 0.2-solve times for all solvers on one problem.
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

# # Plot difference of cdf 0.1-solvability profiles of all solvers on all problems.
# # Reference solver = ASTRO-DF.
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
