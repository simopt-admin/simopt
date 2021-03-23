import numpy as np
from rng.mrg32k3a import MRG32k3a
from base import Solver, Problem, Oracle, Solution
from wrapper_base import Experiment, read_run_results, read_post_replicate_results
from solvers.randomsearch import RandomSearch
from problems.mm1_min_mean_sojourn_time import MM1MinMeanSojournTime
from oracles.mm1queue import MM1Queue
from experiments.rs_on_mm1 import RandomSearchOnMM1
from problems.cntnv_max_profit import CntNVMaxProfit
from experiments.rs_on_cntnv import RandomSearchOnCNTNV

# myexperiment = RandomSearchOnMM1()
myexperiment = RandomSearchOnCNTNV()
myexperiment.run(n_macroreps=10, crn_across_solns=True)
myexperiment.record_run_results(file_name="run_newsvendor")
myexperiment2 = read_run_results(file_name="run_newsvendor")
myexperiment2.post_replicate(n_postreps=20, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)
myexperiment2.record_post_replicate_results(file_name="post_newsvendor")
myexperiment3 = read_post_replicate_results(file_name="post_newsvendor")
myexperiment3.make_plots(plot_type="all")
myexperiment3.make_plots(plot_type="mean")
myexperiment3.make_plots(plot_type="quantile")
myexperiment3.make_plots(plot_type="all", normalize=False)
myexperiment3.make_plots(plot_type="mean", normalize=False)
myexperiment3.make_plots(plot_type="quantile", normalize=False)

# bootstrap_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
# new_est_obj, new_conv_curves = myexperiment.bootstrap_sample(bootstrap_rng, crn_across_budget=True, crn_across_macroreps=False)
# print(new_est_obj)
# print(new_conv_curves)

# conv_curve = myexperiment.all_conv_curves[0]
# print(conv_curve)
# frac_inter_budgets = myexperiment.unique_frac_budgets
# print(frac_inter_budgets)
# solve_tol = 0.1
# print(solve_time_of_conv_curve(conv_curve, frac_inter_budgets, solve_tol))

print('I ran this.')
