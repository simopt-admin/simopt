from wrapper_base import Experiment, read_experiment_results

solver_name = "RNDSRCH" # random search solver
problem_name = "RMITD-1" # mm1 queueing problem
myexperiment = Experiment(solver_name, problem_name, solver_fixed_factors={"sample_size": 20})

#myexperiment = Experiment(solver_name, problem_name, solver_fixed_factors={"sample_size": 10})
# myexperiment.run(n_macroreps=10, crn_across_solns=True)
# print("Here")
# myexperiment = read_experiment_results(solver_name + "_on_" + problem_name)
# myexperiment.post_replicate(n_postreps=100, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)
# print("Now here.")
# myexperiment.plot_progress_curves(plot_type="all", normalize=False)
# myexperiment.plot_progress_curves(plot_type="all", normalize=True)
# print("Finally here.")
# myexperiment.plot_progress_curves(plot_type="mean", normalize=False)
# myexperiment.plot_progress_curves(plot_type="mean", normalize=True)
