#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:22:20 2022

@author: sarashashaani
"""

import sys
import os.path as o
import os
#sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

sys.path.append('/Users/sarashashaani/Documents/GitHub/simopt/simopt/')

from wrapper_base import Experiment, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles


st_devs = [1,2,3,4,5]
holding_costs = [1,100]
inven_stops = [1000,10000]


# Three versions of random search with varying sample sizes.
rs_sample_sizes = [10,50]

for sd in st_devs:
    for hc in holding_costs:
        for inv in inven_stops:
            model_fixed_factors = {"st_dev": sd,
                                    "holding_cost": hc,
                                    "inven_stop": inv
                                    }
            # Default budget for (s,S) inventory problem = 1000 replications.
            # RS with sample size of 100 will get through only 10 iterations.
            problem_fixed_factors = {"budget": 1000}
            problem_rename = f"IRONORECONT-1_sd={sd}_hc={hc}_is={inv}"
            
            # Temporarily store experiments on the same problem for post-normalization.
            experiments_same_problem = []
            
            # Loop over random search solvers.
            for rs_ss in rs_sample_sizes:
                solver_fixed_factors = {"sample_size": rs_ss}
                solver_rename = f"RNDSRCH_ss={rs_ss}"
                # Create experiment for the problem-solver pair.
                new_experiment = Experiment(solver_name="RNDSRCH",
                                            problem_name="IRONORECONT-1",
                                            solver_rename=solver_rename,
                                            problem_rename=problem_rename,
                                            solver_fixed_factors=solver_fixed_factors,
                                            problem_fixed_factors=problem_fixed_factors,
                                            model_fixed_factors=model_fixed_factors
                                            )
                # Run experiment with M = 10.
                new_experiment.run(n_macroreps=10)
                # Post replicate experiment with N = 100.
                new_experiment.post_replicate(n_postreps=100)
                experiments_same_problem.append(new_experiment)
    
            solver_fixed_factors = {"delta_max": 200.0}
            new_experiment = Experiment(solver_name="ASTRODF",
                                        problem_name="IRONORECONT-1",
                                        problem_rename=problem_rename,
                                        solver_fixed_factors=solver_fixed_factors,
                                        problem_fixed_factors=problem_fixed_factors,
                                        model_fixed_factors=model_fixed_factors
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
for rs_ss in rs_sample_sizes:
    solver_rename = f"RNDSRCH_ss={rs_ss}"
    experiments_same_solver = []
    for sd in st_devs:
        for hc in holding_costs:
            for inv in inven_stops:
                problem_rename = f"IRONORECONT-1_sd={sd}_hc={hc}_is={inv}"
                file_name = f"{solver_rename}_on_{problem_rename}"
                # Load experiment.
                new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                # Rename problem and solver to produce nicer plot labels.
                new_experiment.solver.name = f"Random Search {rs_ss}"
                new_experiment.problem.name = fr"IRONORECONT-1 with $\sigma={sd}$ and hc={hc} and is={inv}"
                experiments_same_solver.append(new_experiment)
    experiments.append(experiments_same_solver)

solver_rename = "ASTRODF"
experiments_same_solver = []
for sd in st_devs:
        for hc in holding_costs:
            for inv in inven_stops:
                problem_rename = f"IRONORECONT-1_sd={sd}_hc={hc}_is={inv}"
                file_name = f"{solver_rename}_on_{problem_rename}"
                # Load experiment.
                new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                # Rename problem and solver to produce nicer plot labels.
                new_experiment.solver.name = "ASTRO-DF"
                new_experiment.problem.name = fr"IRONORECONT-1 with $\sigma={sd}$ and hc={hc} and is={inv}"
                experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)


# PLOTTING

n_solvers = len(experiments)
n_problems = len(experiments[0])


plot_area_scatterplots(experiments, all_in_one=True, plot_CIs=True, print_max_hw=True)
plot_solvability_profiles(experiments, plot_type="cdf_solvability", solve_tol=0.1, all_in_one=True, plot_CIs=True, print_max_hw=True)
plot_solvability_profiles(experiments, plot_type="quantile_solvability", solve_tol=0.1, beta=0.5, all_in_one=True, plot_CIs=True, print_max_hw=True)
                          
for i in range(n_problems+1):
    plot_progress_curves([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="mean", all_in_one=True, plot_CIs=True, print_max_hw=True)
    # plot_progress_curves([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="quantile", beta=0.9, all_in_one=True, plot_CIs=True, print_max_hw=True)
    # plot_solvability_cdfs([experiments[solver_idx][i] for solver_idx in range(n_solvers)], solve_tol=0.2,  all_in_one=True, plot_CIs=True, print_max_hw=True)


