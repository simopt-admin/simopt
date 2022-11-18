"""
This script runs five solvers on 20 versions of SAN problem.
Produces plots appearing in the INFORMS Journal on Computing submission.
"""

import sys
import os.path as o
import os
# sys.path.append('/Users/sarashashaani/Documents/GitHub/simopt/simopt/')

from wrapper_base import Experiment, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress

# Problems factors used in experiments 
## SAN
all_random_costs =  [(1, 2, 2, 7, 17, 7, 2, 13, 1, 9, 18, 16, 7), 
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
                     (18, 4, 19, 2, 13, 11, 9, 2, 17, 18, 11, 7, 14)]
num_problems = len(all_random_costs)

# Two versions of random search with varying sample sizes.
rs_sample_sizes = [10, 50]

# RUNNING AND POST-PROCESSING EXPERIMENTS
M = 10
N = 100
L = 200

# Loop over problem instances.
for i in range(num_problems):
    problem_fixed_factors = {"budget": 10000, "arc_costs":all_random_costs[i]}
    problem_rename = f"SAN-1_rc={all_random_costs[i]}"
    
    # Temporarily store experiments on the same problem for post-normalization.
    experiments_same_problem = []
    
    # Loop over random search solvers.
    for rs_ss in rs_sample_sizes:
        solver_fixed_factors = {"sample_size": rs_ss}
        solver_rename = f"RNDSRCH_ss={rs_ss}"
        # Create experiment for the problem-solver pair.
        new_experiment = Experiment(solver_name="RNDSRCH",
                                    problem_name="SAN-1",
                                    solver_rename=solver_rename,
                                    problem_rename=problem_rename,
                                    solver_fixed_factors=solver_fixed_factors,
                                    problem_fixed_factors=problem_fixed_factors
                                    )
        # Run experiment with M.
        new_experiment.run(n_macroreps=M)
        # Post replicate experiment with N.
        new_experiment.post_replicate(n_postreps=N)
        experiments_same_problem.append(new_experiment)

    solver_fixed_factors = {"delta_max": 200.0, "lambda_min" : 3}
    new_experiment = Experiment(solver_name="ASTRODF",
                                problem_name="SAN-1",
                                problem_rename=problem_rename,
                                solver_fixed_factors=solver_fixed_factors,
                                problem_fixed_factors=problem_fixed_factors
                                )
    # Run experiment with M.
    new_experiment.run(n_macroreps=M)
    # Post replicate experiment with N.
    new_experiment.post_replicate(n_postreps=N)
    experiments_same_problem.append(new_experiment)
    
    new_experiment = Experiment(solver_name="NELDMD",
                                problem_name="SAN-1",
                                problem_rename=problem_rename,
                                solver_fixed_factors=solver_fixed_factors,
                                problem_fixed_factors=problem_fixed_factors
                                )
    # Run experiment with M.
    new_experiment.run(n_macroreps=M)
    # Post replicate experiment with N.
    new_experiment.post_replicate(n_postreps=N)
    experiments_same_problem.append(new_experiment)
    
    new_experiment = Experiment(solver_name="STRONG",
                                problem_name="SAN-1",
                                problem_rename=problem_rename,
                                solver_fixed_factors=solver_fixed_factors,
                                problem_fixed_factors=problem_fixed_factors
                                )
    # Run experiment with M.
    new_experiment.run(n_macroreps=M)
    # Post replicate experiment with N.
    new_experiment.post_replicate(n_postreps=N)
    experiments_same_problem.append(new_experiment)
    
    # Post-normalize experiments with L.
    # Provide NO proxies for f(x0), f(x*), or f(x).
    post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=L)

# LOAD DATA FROM .PICKLE FILES TO PREPARE FOR PLOTTING.

# For plotting, "experiments" will be a list of list of Experiment objects.
#   outer list - indexed by solver
#   inner list - index by problem
experiments = []

# Load .pickle files of past results.
# Load all experiments for a given solver, for all solvers.

for rs_ss in rs_sample_sizes:
    solver_rename = f"RNDSRCH_ss={rs_ss}"
    experiments_same_solver = []
    for i in range(20):
        problem_rename = f"SAN-1_rc={all_random_costs[i]}"
        file_name = f"{solver_rename}_on_{problem_rename}"
        # Load experiment.
        new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
        # Rename problem and solver to produce nicer plot labels.
        new_experiment.solver.name = f"Random Search {rs_ss}"
        new_experiment.problem.name = f"SAN-1 with rc={all_random_costs[i]}"
        experiments_same_solver.append(new_experiment)
    experiments.append(experiments_same_solver)

solver_rename = "ASTRODF"
experiments_same_solver = []
for i in range(20):
    problem_rename = f"SAN-1_rc={all_random_costs[i]}"
    file_name = f"{solver_rename}_on_{problem_rename}"
    # Load experiment.
    new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
    # Rename problem and solver to produce nicer plot labels.
    new_experiment.solver.name = "ASTRO-DF"
    new_experiment.problem.name = f"SAN-1 with rc={all_random_costs[i]}"
    experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

solver_rename = "NELDMD"
experiments_same_solver = []
for i in range(20):
    problem_rename = f"SAN-1_rc={all_random_costs[i]}"
    file_name = f"{solver_rename}_on_{problem_rename}"
    # Load experiment.
    new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
    # Rename problem and solver to produce nicer plot labels.
    new_experiment.solver.name = "Nelder-Mead"
    new_experiment.problem.name = f"SAN-1 with rc={all_random_costs[i]}"
    experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

solver_rename = "STRONG"
experiments_same_solver = []
for i in range(20):
    problem_rename = f"SAN-1_rc={all_random_costs[i]}"
    file_name = f"{solver_rename}_on_{problem_rename}"
    # Load experiment.
    new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
    # Rename problem and solver to produce nicer plot labels.
    new_experiment.solver.name = "STRONG"
    new_experiment.problem.name = f"SAN-1 with rc={all_random_costs[i]}"
    experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

# PLOTTING

n_solvers = len(experiments)
n_problems = len(experiments[0])

plot_area_scatterplots(experiments, all_in_one=True, plot_CIs=True, print_max_hw=True)
plot_solvability_profiles(experiments, plot_type="cdf_solvability", solve_tol=0.1, all_in_one=True, plot_CIs=True, print_max_hw=True)
plot_solvability_profiles(experiments, plot_type="quantile_solvability", solve_tol=0.1, beta=0.5, all_in_one=True, plot_CIs=True, print_max_hw=True)
plot_solvability_profiles(experiments=experiments, plot_type="diff_cdf_solvability", solve_tol=0.1, ref_solver="ASTRO-DF", all_in_one=True, plot_CIs=True, print_max_hw=True)
plot_solvability_profiles(experiments=experiments, plot_type="diff_quantile_solvability", solve_tol=0.1, beta=0.5, ref_solver="ASTRO-DF", all_in_one=True, plot_CIs=True, print_max_hw=True)
plot_terminal_scatterplots(experiments, all_in_one=True)

                      
# for i in range(n_problems):
#     plot_progress_curves([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="mean", all_in_one=True, plot_CIs=True, print_max_hw=True, normalize=False)
#     plot_terminal_progress([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="violin", normalize=True, all_in_one=True)
