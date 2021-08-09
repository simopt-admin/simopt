from wrapper_base import Experiment, read_experiment_results, post_normalize, plot_progress_curves

# new_experiment = Experiment(solver_name="RNDSRCH",
#                             problem_name="CNTNEWS-1")
# # Run experiment with M = 100.
# new_experiment.run(n_macroreps=100)
# # Post replicate experiment with N = 100.
# new_experiment.post_replicate(n_postreps=100)
# # Post normalize.
# post_normalize([new_experiment], n_postreps_init_opt=200)

new_experiment = read_experiment_results("experiments/outputs/RNDSRCH_on_CNTNEWS-1.pickle")
# Mean progress curves from all solvers on one problem.
plot_progress_curves(experiments=[new_experiment],
                     plot_type="mean",
                     all_in_one=True,
                     plot_CIs=True,
                     print_max_hw=False
                     )
