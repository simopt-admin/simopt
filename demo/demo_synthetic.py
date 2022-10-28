from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress

# Problem dimension
Dim = [5,10,20,30]

num_problems = len(Dim)
solvers = ["ASTRODF", "NELDMD", "ADAM", "ALOE"]

for i in range(num_problems):
    model_fixed_factors = {"dim":Dim[i]}
    problem_fixed_factors = {"budget": 5000}
    problem_rename = f"SYN-1_dim={Dim[i]}"

    # Temporarily store experiments on the same problem for post-normalization.
    experiments_same_problem = []
    solver_fixed_factors = {}

    for solver in solvers:
        solver_name = solver
        if solver_name == "ASTRODF":
            solver_fixed_factors = {"delta_max": 10.0}
        new_experiment = ProblemSolver(solver_name=solver_name,
                                        problem_name="SYN-1",
                                        problem_rename=problem_rename,
                                        model_fixed_factors=model_fixed_factors,
                                        solver_fixed_factors=solver_fixed_factors,
                                        problem_fixed_factors=problem_fixed_factors
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
experiments = []

for solver in solvers:
    solver_rename = solver
    experiments_same_solver = []
    for i in range(num_problems):
        problem_rename = f"SYN-1_dim={Dim[i]}"
        file_name = f"{solver_rename}_on_{problem_rename}"
        # Load experiment.
        new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
        # Rename problem and solver to produce nicer plot labels.
        new_experiment.solver.name = solver_rename
        new_experiment.problem.name = f"SYN-1 with dim={Dim[i]}"
        experiments_same_solver.append(new_experiment)
    experiments.append(experiments_same_solver)

# PLOTTING
n_solvers = len(experiments)
n_problems = len(experiments[0])

CI_param = True
alpha = 0.2

plot_solvability_profiles(experiments, plot_type="cdf_solvability", solve_tol=alpha, all_in_one=True, plot_CIs=CI_param, print_max_hw=CI_param)
plot_solvability_profiles(experiments, plot_type="quantile_solvability", solve_tol=alpha, beta=0.5, all_in_one=True, plot_CIs=CI_param, print_max_hw=CI_param)
#plot_solvability_profiles(experiments=experiments, plot_type="diff_cdf_solvability", solve_tol=alpha, ref_solver="ASTRO-DF", all_in_one=True, plot_CIs=CI_param, print_max_hw=CI_param)
#plot_solvability_profiles(experiments=experiments, plot_type="diff_quantile_solvability", solve_tol=alpha, beta=0.5, ref_solver="ASTRO-DF", all_in_one=True, plot_CIs=CI_param, print_max_hw=CI_param)
plot_area_scatterplots(experiments, all_in_one=True, plot_CIs=CI_param, print_max_hw=CI_param)
plot_terminal_scatterplots(experiments, all_in_one=True)

for i in range(n_problems):
    plot_progress_curves([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="mean", all_in_one=True, plot_CIs=CI_param, print_max_hw=True)
    plot_terminal_progress([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="violin", normalize=True, all_in_one=True)
