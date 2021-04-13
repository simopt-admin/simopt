from wrapper_base import Experiment

# solver_name = "RNDSRCH" # random search solver
# problem_name = "CNTNEWS-1" # mm1 queueing problem
# myexperiment = Experiment(solver_name, problem_name)
# myexperiment.run(n_macroreps=10, crn_across_solns=True)
# myexperiment.post_replicate(n_postreps=20, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)
# myexperiment.plot_progress_curves(plot_type="all", normalize=False)

solver_name = "SANE" # random search solver
problem_name = "CNTNEWS-1" # mm1 queueing problem
myexperiment = Experiment(solver_name, problem_name)
myexperiment.run(n_macroreps=5, crn_across_solns=True)
myexperiment.post_replicate(n_postreps=20, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)
myexperiment.plot_progress_curves(plot_type="all", normalize=False)