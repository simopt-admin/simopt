"""
This script runs five solvers on 20 versions of the (s, S) inventory problem.
Produces plots appearing in the INFORMS Journal on Computing submission.
"""

import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress

def main():
    # Default values of the (s, S) model:
    # "demand_mean": 100.0
    # "lead_mean": 6.0
    # "backorder_cost": 4.0
    # "holding_cost": 1.0
    # "fixed_cost": 36.0
    # "variable_cost": 2.0

    # Create 20 problem instances by varying two factors, five levels.
    demand_means = [25.0, 50.0, 100.0, 200.0, 400.0]
    lead_means = [1.0, 3.0, 6.0, 9.0]

    demand_means = [400.0]
    lead_means = [6.0]

    # Two versions of random search with varying sample sizes.
    rs_sample_sizes = [10, 50]

    # RUNNING AND POST-PROCESSING EXPERIMENTS
    M = 10
    N = 100
    L = 200
    # Loop over problem instances.
    for dm in demand_means:
        for lm in lead_means:
            model_fixed_factors = {"demand_mean": dm,
                                "lead_mean": lm}
            # Default budget for (s,S) inventory problem = 1000 replications.
            # RS with sample size of 100 will get through only 10 iterations.
            problem_fixed_factors = {"budget": 1000}
            problem_rename = f"SSCONT-1_dm={dm}_lm={lm}"

            # Temporarily store experiments on the same problem for post-normalization.
            experiments_same_problem = []

            # Loop over random search solvers.
            for rs_ss in rs_sample_sizes:
                solver_fixed_factors = {"sample_size": rs_ss}
                solver_rename = f"RNDSRCH_ss={rs_ss}"
                # Create experiment for the problem-solver pair.
                new_experiment = ProblemSolver(solver_name="RNDSRCH",
                                            problem_name="SSCONT-1",
                                            solver_rename=solver_rename,
                                            problem_rename=problem_rename,
                                            solver_fixed_factors=solver_fixed_factors,
                                            problem_fixed_factors=problem_fixed_factors,
                                            model_fixed_factors=model_fixed_factors
                                            )
                # Run experiment with M.
                new_experiment.run(n_macroreps=M)
                # Post replicate experiment with N.
                new_experiment.post_replicate(n_postreps=N)
                experiments_same_problem.append(new_experiment)

            # Setup and run ASTRO-DF.
            solver_fixed_factors = {"delta_max": 200.0}
            new_experiment = ProblemSolver(solver_name="ASTRODF",
                                        problem_name="SSCONT-1",
                                        problem_rename=problem_rename,
                                        solver_fixed_factors=solver_fixed_factors,
                                        problem_fixed_factors=problem_fixed_factors,
                                        model_fixed_factors=model_fixed_factors
                                        )
            # Run experiment with M.
            new_experiment.run(n_macroreps=M)
            # Post replicate experiment with N.
            new_experiment.post_replicate(n_postreps=N)
            experiments_same_problem.append(new_experiment)

            # Setup and run Nelder-Mead.
            new_experiment = ProblemSolver(solver_name="NELDMD",
                                        problem_name="SSCONT-1",
                                        problem_rename=problem_rename,
                                        problem_fixed_factors=problem_fixed_factors,
                                        model_fixed_factors=model_fixed_factors
                                        )
            # Run experiment withM.
            new_experiment.run(n_macroreps=M)
            # Post replicate experiment with N.
            new_experiment.post_replicate(n_postreps=N)
            experiments_same_problem.append(new_experiment)

            # Setup and run STRONG.=
            new_experiment = ProblemSolver(solver_name="STRONG",
                                        problem_name="SSCONT-1",
                                        problem_rename=problem_rename,
                                        problem_fixed_factors=problem_fixed_factors,
                                        model_fixed_factors=model_fixed_factors
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
        for dm in demand_means:
            for lm in lead_means:
                problem_rename = f"SSCONT-1_dm={dm}_lm={lm}"
                file_name = f"{solver_rename}_on_{problem_rename}"
                # Load experiment.
                new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                # Rename problem and solver to produce nicer plot labels.
                new_experiment.solver.name = f"RS{rs_ss}"
                new_experiment.problem.name = fr"SSCONT-1 with $\mu_D={round(dm)}$ and $\mu_L={round(lm)}$"
                experiments_same_solver.append(new_experiment)
        experiments.append(experiments_same_solver)
    # Load ASTRO-DF results.
    solver_rename = "ASTRODF"
    experiments_same_solver = []
    for dm in demand_means:
        for lm in lead_means:
            problem_rename = f"SSCONT-1_dm={dm}_lm={lm}"
            file_name = f"{solver_rename}_on_{problem_rename}"
            # Load experiment.
            new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
            # Rename problem and solver to produce nicer plot labels.
            new_experiment.solver.name = "ASTRO-DF"
            new_experiment.problem.name = fr"SSCONT-1 with $\mu_D={round(dm)}$ and $\mu_L={round(lm)}$"
            experiments_same_solver.append(new_experiment)
    experiments.append(experiments_same_solver)
    # Load Nelder-Mead results.
    solver_rename = "NELDMD"
    experiments_same_solver = []
    for dm in demand_means:
        for lm in lead_means:
            problem_rename = f"SSCONT-1_dm={dm}_lm={lm}"
            file_name = f"{solver_rename}_on_{problem_rename}"
            # Load experiment.
            new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
            # Rename problem and solver to produce nicer plot labels.
            new_experiment.solver.name = "Nelder-Mead"
            new_experiment.problem.name = fr"SSCONT-1 with $\mu_D={round(dm)}$ and $\mu_L={round(lm)}$"
            experiments_same_solver.append(new_experiment)
    experiments.append(experiments_same_solver)
    # Load STRONG results.
    solver_rename = "STRONG"
    experiments_same_solver = []
    for dm in demand_means:
        for lm in lead_means:
            problem_rename = f"SSCONT-1_dm={dm}_lm={lm}"
            file_name = f"{solver_rename}_on_{problem_rename}"
            # Load experiment.
            new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
            # Rename problem and solver to produce nicer plot labels.
            new_experiment.solver.name = "STRONG"
            new_experiment.problem.name = fr"SSCONT-1 with $\mu_D={round(dm)}$ and $\mu_L={round(lm)}$"
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


    for i in range(n_problems):
        plot_progress_curves([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="mean", all_in_one=True, plot_CIs=True, print_max_hw=True, normalize=False)
        plot_terminal_progress([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="violin", normalize=True, all_in_one=True)

    # from math import exp, log, sqrt
    # import numpy as np
    # import matplotlib.pyplot as plt
    # for mu_d in demand_means:
    #     for mu_l in lead_means:
    #         lq = mu_d*mu_l/3
    #         uq = mu_d*mu_l+2*sqrt(2*mu_d**2*mu_l)
    #         mu = (log(lq)+log(uq))/2
    #         sigma = (log(uq)-mu)/1.96
    #         print(round(mu_d,0), round(mu_l,0), round(lq,2), round(uq,2), round(mu,2), round(sigma,2))
    #         s = np.random.lognormal(mu, sigma, 1000)
    #         plt.hist(s, density=True, alpha=0.5, label=str(round(mu_d,0))+','+str(round(mu_l,0)), bins=50, color='blue')
    #         plt.axvline(lq,color='red')
    #         plt.axvline(uq,color='red')
    #         plt.axis('tight')
    #         plt.legend()
    #         plt.show()

    # # Mean progress curves from all solvers on one problem. Problem instance 0.
    # plot_progress_curves(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
    #                      plot_type="mean",
    #                      all_in_one=True,
    #                      plot_CIs=True,
    #                      print_max_hw=False
    #                      )

    # # Mean progress curves from all solvers on one problem. Problem instance 22.
    # plot_progress_curves(experiments=[experiments[solver_idx][22] for solver_idx in range(n_solvers)],
    #                      plot_type="mean",
    #                      all_in_one=True,
    #                      plot_CIs=True,
    #                      print_max_hw=False
    #                      )

    # # Plot 0.9-quantile progress curves from all solvers on one problem. Problem instance 0.
    # plot_progress_curves(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
    #                      plot_type="quantile",
    #                      beta=0.9,
    #                      all_in_one=True,
    #                      plot_CIs=True,
    #                      print_max_hw=False
    #                      )

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

if __name__ == "__main__":
    main()