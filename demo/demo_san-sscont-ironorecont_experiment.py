"""
This script is intended to help with a large experiment with
5 solvers (two versions of random search, ASTRO-DF, STRONG, and Nelder-Mead)
and 60 problems (20 unique instances of problems from
(s, S) inventory, iron ore, and stochastic activity network).
Produces plots appearing in the INFORMS Journal on Computing submission.
"""
import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress

def main():
    # Problems factors used in experiments
    # SAN
    all_random_costs = [(1, 2, 2, 7, 17, 7, 2, 13, 1, 9, 18, 16, 7),
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


    # SSCONT
    demand_means = [25.0, 50.0, 100.0, 200.0, 400.0]
    lead_means = [1.0, 3.0, 6.0, 9.0]

    # IRONORECONT
    st_devs = [1, 2, 3, 4, 5]
    holding_costs = [1, 100]
    inven_stops = [1000, 10000]

    # RUNNING AND POST-PROCESSING EXPERIMENTS
    M = 10
    N = 100
    L = 200


    # Five solvers.
    solvers = ["RNDSRCH_ss=10", "RNDSRCH_ss=50", "ASTRODF", "NELDMD", "STRONG"]
    # Two versions of random search with varying sample sizes.
    rs_sample_sizes = [10, 50]
    # ASTRODF factors
    delta_max = 200.0


    # First problem: SAN
    # Loop over problem instances.
    for i in range(num_problems):
        model_fixed_factors = {}
        problem_fixed_factors = {"budget": 10000, "arc_costs": all_random_costs[i]}
        problem_rename = f"SAN-1_rc={all_random_costs[i]}"

        # Temporarily store experiments on the same problem for post-normalization.
        experiments_same_problem = []
        solver_fixed_factors = {}

        for solver in solvers:
            solver_name = solver
            if solver == "RNDSRCH_ss=10":
                solver_name = "RNDSRCH"
                solver_fixed_factors = {"sample_size": 10}
            elif solver == "RNDSRCH_ss=50":
                solver_name = "RNDSRCH"
                solver_fixed_factors = {"sample_size": 50}
            elif solver == "ASTRODF":
                solver_fixed_factors = {"delta_max": delta_max}

            # Loop over solvers:
            new_experiment = ProblemSolver(solver_name=solver_name,
                                        solver_rename=solver,
                                        solver_fixed_factors=solver_fixed_factors,
                                        problem_name="SAN-1",
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

    # Second problem: SSCONT
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
            solver_fixed_factors = {}
            for solver in solvers:
                solver_name = solver
                if solver == "RNDSRCH_ss=10":
                    solver_name = "RNDSRCH"
                    solver_fixed_factors = {"sample_size": 10}
                elif solver == "RNDSRCH_ss=50":
                    solver_name = "RNDSRCH"
                    solver_fixed_factors = {"sample_size": 50}
                elif solver == "ASTRODF":
                    solver_fixed_factors = {"delta_max": delta_max}

                # Loop over solvers:
                new_experiment = ProblemSolver(solver_name=solver_name,
                                            solver_rename=solver,
                                            solver_fixed_factors=solver_fixed_factors,
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

    # Third problem: IRONORECONT
    for sd in st_devs:
        for hc in holding_costs:
            for inv in inven_stops:
                model_fixed_factors = {"st_dev": sd,
                                    "holding_cost": hc,
                                    "inven_stop": inv}
                problem_fixed_factors = {"budget": 1000}
                problem_rename = f"IRONORECONT-1_sd={sd}_hc={hc}_inv={inv}"

                # Temporarily store experiments on the same problem for post-normalization.
                experiments_same_problem = []
                solver_fixed_factors = {}
                for solver in solvers:
                    solver_name = solver
                    if solver == "RNDSRCH_ss=10":
                        solver_name = "RNDSRCH"
                        solver_fixed_factors = {"sample_size": 10}
                    elif solver == "RNDSRCH_ss=50":
                        solver_name = "RNDSRCH"
                        solver_fixed_factors = {"sample_size": 50}
                    elif solver == "ASTRODF":
                        solver_fixed_factors = {"delta_max": delta_max}

                    # Loop over solvers:
                    new_experiment = ProblemSolver(solver_name=solver_name,
                                                solver_rename=solver,
                                                solver_fixed_factors=solver_fixed_factors,
                                                problem_name="IRONORECONT-1",
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

    # For plotting, "experiments" will be a list of list of ProblemSolver objects.
    #   outer list - indexed by solver
    #   inner list - index by problem
    experiments = []


    # Load .pickle files of past results.
    # Load all experiments for a given solver, for all solvers.
    # Load experiments belonging to the problems in:
    problems = ["SAN", "SSCONT", "IRONORECONT"]
    # problems = ["SAN"]

    for solver in solvers:
        experiments_same_solver = []

        solver_display = solver
        if solver == "RNDSRCH_ss=10":
            solver_display = "RS10"
        elif solver == "RNDSRCH_ss=50":
            solver_display = "RS50"
        elif solver == "ASTRODF":
            solver_display = "ASTRO-DF"
        elif solver == "NELDMD":
            solver_display = "Nelder-Mead"

        for problem in problems:
            if problem == "SAN":
                # Load SAN .pickle files
                for i in range(num_problems):
                    problem_rename = f"{problem}-1_rc={all_random_costs[i]}"
                    file_name = f"{solver}_on_{problem_rename}"
                    # Load experiment.
                    new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                    # Rename problem to produce nicer plot labels.
                    new_experiment.problem.name = f"{problem}-1 with rc={all_random_costs[i]}"
                    new_experiment.solver.name = solver_display
                    experiments_same_solver.append(new_experiment)

            elif problem == "SSCONT":
                # Load SSCONT .pickle files
                for dm in demand_means:
                    for lm in lead_means:
                        problem_rename = f"{problem}-1_dm={dm}_lm={lm}"
                        file_name = f"{solver}_on_{problem_rename}"
                        # Load experiment.
                        new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                        # Rename problem to produce nicer plot labels.
                        new_experiment.problem.name = fr"{problem}-1 with $\mu_D={round(dm)}$ and $\mu_L={round(lm)}$"
                        new_experiment.solver.name = solver_display
                        experiments_same_solver.append(new_experiment)

            elif problem == "IRONORECONT":
                # Load IRONORECONT .pickle files
                for sd in st_devs:
                    for hc in holding_costs:
                        for inv in inven_stops:
                            problem_rename = f"{problem}-1_sd={sd}_hc={hc}_inv={inv}"
                            file_name = f"{solver}_on_{problem_rename}"
                            # Load experiment.
                            new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                            # Rename problem to produce nicer plot labels.
                            new_experiment.problem.name = fr"{problem}-1 with $\sigma={sd}$ and hc={hc} and inv={inv}"
                            new_experiment.solver.name = solver_display
                            experiments_same_solver.append(new_experiment)

        experiments.append(experiments_same_solver)

    # PLOTTING

    n_solvers = len(experiments)
    n_problems = len(experiments[0])

    CI_param = True
    alpha = 0.2

    plot_solvability_profiles(experiments, plot_type="cdf_solvability", solve_tol=alpha, all_in_one=True, plot_CIs=CI_param, print_max_hw=CI_param)
    plot_solvability_profiles(experiments, plot_type="quantile_solvability", solve_tol=alpha, beta=0.5, all_in_one=True, plot_CIs=CI_param, print_max_hw=CI_param)
    plot_solvability_profiles(experiments=experiments, plot_type="diff_cdf_solvability", solve_tol=alpha, ref_solver="ASTRO-DF", all_in_one=True, plot_CIs=CI_param, print_max_hw=CI_param)
    plot_solvability_profiles(experiments=experiments, plot_type="diff_quantile_solvability", solve_tol=alpha, beta=0.5, ref_solver="ASTRO-DF", all_in_one=True, plot_CIs=CI_param, print_max_hw=CI_param)
    plot_area_scatterplots(experiments, all_in_one=True, plot_CIs=CI_param, print_max_hw=CI_param)
    plot_terminal_scatterplots(experiments, all_in_one=True)

    for i in range(n_problems):
        plot_progress_curves([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="mean", all_in_one=True, plot_CIs=CI_param, print_max_hw=True)
        plot_terminal_progress([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="violin", normalize=True, all_in_one=True)
        # plot_solvability_cdfs([experiments[solver_idx][i] for solver_idx in range(n_solvers)], solve_tol=0.2,  all_in_one=True, plot_CIs=True, print_max_hw=True)

    # Plots for mu_D = 400 and mu_L = 6 (appreared in the paper)
    plot_progress_curves([experiments[solver_idx][0] for solver_idx in range(n_solvers)], plot_type="all", all_in_one=True)

    plot_progress_curves([experiments[solver_idx][0] for solver_idx in range(3, 4)], plot_type="all", all_in_one=True, normalize=False)

    plot_progress_curves([experiments[solver_idx][0] for solver_idx in range(n_solvers)], plot_type="mean", all_in_one=True, plot_CIs=True, print_max_hw=False, normalize=True)

    plot_solvability_cdfs(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)], solve_tol=0.2, all_in_one=True, plot_CIs=True, print_max_hw=False)

    plot_terminal_progress([experiments[solver_idx][0] for solver_idx in range(n_solvers)], plot_type="violin", normalize=False, all_in_one=True)

if (__name__ == "__main__"):
    main()