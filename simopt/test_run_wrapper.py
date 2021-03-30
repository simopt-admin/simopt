import numpy as np
from rng.mrg32k3a import MRG32k3a
from base import Solver, Problem, Oracle, Solution
from wrapper_base import Experiment, read_run_results, read_post_replicate_results
# from solvers.randomsearch import RandomSearch
# from problems.mm1_min_mean_sojourn_time import MM1MinMeanSojournTime
# from oracles.mm1queue import MM1Queue
# from problems.cntnv_max_profit import CntNVMaxProfit

# myexperiment = Experiment(solver_name="RNDSRCH", problem_name="MM1-1")
myexperiment = Experiment(solver_name="RNDSRCH", problem_name="CNTNEWS-1")
myexperiment.run(n_macroreps=10, crn_across_solns=True)
myexperiment.record_run_results(file_name="run_newsvendor")
myexperiment2 = read_run_results(file_name="run_newsvendor")
myexperiment2.post_replicate(n_postreps=20, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)
myexperiment2.record_post_replicate_results(file_name="post_newsvendor")
myexperiment3 = read_post_replicate_results(file_name="post_newsvendor")
#myexperiment3.compute_area_stats()
#myexperiment3.compute_solvability_quantile()
myexperiment3.plot_solvability_curve(solve_tol=0.1, plot_CIs=False)
# myexperiment3.plot_progress_curves(plot_type="all")
# myexperiment3.plot_progress_curves(plot_type="mean")
# myexperiment3.plot_progress_curves(plot_type="quantile")
# myexperiment3.plot_progress_curves(plot_type="all", normalize=False)
# myexperiment3.plot_progress_curves(plot_type="mean", normalize=False)
# myexperiment3.plot_progress_curves(plot_type="quantile", normalize=False)

print('I ran this.')
