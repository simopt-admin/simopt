import numpy as np
from rng.mrg32k3a import MRG32k3a
from base import Solver, Problem, Oracle, Solution
from wrapper_base import Experiment
from solvers.randomsearch import RandomSearch
from problems.mm1_min_mean_sojourn_time import MM1MinMeanSojournTime
from oracles.mm1queue import MM1Queue

class RandomSearchOnMM1(Experiment):
    """
    Base class to implement wrappers for running experiments.
    Random Search solver on M/M/1 Queueing problem

    Attributes
    ----------
    solver : base.Solver object
        simulation-optimization solver
    problem : base.Problem object
        simulation-optimization problem
    solver_fixed_factors : dict
        dictionary of user-specified solver factors
    problem_fixed_factors : dict
        dictionary of user-specified problem factors  
    oracle_fixed_factors : dict
        dictionary of user-specified oracle factors
    all_recommended_xs : list of lists of tuples
        sequences of recommended solutions from each macroreplication
    all_intermediate_budgets : list of lists
        sequences of intermediate budgets from each macroreplication
    all_reevaluated_solns : list of Solution objects
        reevaluated solutions recommended by the solver
        """
    def __init__(self):
        self.solver_fixed_factors = {}
        self.problem_fixed_factors = {} # unused
        self.oracle_fixed_factors = {}
        self.solver = RandomSearch(fixed_factors=self.solver_fixed_factors)
        self.problem = MM1MinMeanSojournTime(oracle_fixed_factors=self.oracle_fixed_factors)
        self.all_recommended_xs = []
        self.all_intermediate_budgets = []
        self.all_reevaluated_solns = []