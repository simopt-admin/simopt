import numpy as np
from rng.mrg32k3a import MRG32k3a
from base import Solver, Problem, Oracle, Solution
from wrapper_base import Experiment
from solvers.randomsearch import RandomSearch
from problems.cntnv_max_profit import CntNVMaxProfit
from oracles.cntnv import CntNV


class RandomSearchOnCNTNV(Experiment):
    """
    Base class to implement wrappers for running experiments.
    Example: Random Search solver on continuous newsvendor problem

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
        self.problem_fixed_factors = {}  # unused
        self.oracle_fixed_factors = {}
        self.solver = RandomSearch(fixed_factors=self.solver_fixed_factors)
        self.problem = CntNVMaxProfit(oracle_fixed_factors=self.oracle_fixed_factors)
        super().__init__()
