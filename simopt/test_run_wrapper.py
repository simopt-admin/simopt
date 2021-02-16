import numpy as np
from rng.mrg32k3a import MRG32k3a
from base import Solver, Problem, Oracle, Solution
from wrapper_base import Experiment
from solvers.randomsearch import RandomSearch
from problems.mm1_min_mean_sojourn_time import MM1MinMeanSojournTime
from oracles.mm1queue import MM1Queue
from experiments.rs_on_mm1 import RandomSearchOnMM1

myexperiment = RandomSearchOnMM1()
myexperiment.run(n_macroreps=5, crn_across_solns=True)
myexperiment.post_replicate(n_postreps=10, n_postreps_init_opt=10) # handle different numbers later
#myexperiment.make_plots(plot_type="mean")
#myexperiment.make_plots(plot_type="quantile")

print('I ran this.')