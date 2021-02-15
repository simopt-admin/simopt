#!/usr/bin/env python
"""
Summary
-------
Provide base classes for experiments.

Listing
-------
Experiment : class
"""

import numpy as np
from rng.mrg32k3a import MRG32k3a
from base import Solver, Problem, Oracle, Solution
import matplotlib.pyplot as plt

class Experiment(object):
    """
    Base class to implement wrappers for running experiments.

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
    """
    def __init__(self):
        pass

    def run(self, n_macroreps, crn_across_solns):
        """
        Run n_macroreps of the solver on the problem.

        Arguments
        ---------
        n_macroreps : int
            number of macroreplications of the solver to run on the problem
        crn_across_solns : bool
            indicates if CRN are used when simulating different solutions
        """

        # create, initialize, and attach random number generators
        # Stream 0 is reserved for overhead
        # Substream 0: rng for random problem instance
        rng0 = MRG32k3a(s_ss_sss_index=[0, 0, 0]) # Stream 0, Substream 0, Subsubstream 0
        # Substream 1: rng for random initial solution x0 and restart solutions 
        rng1 = MRG32k3a(s_ss_sss_index=[0, 1, 0]) # Stream 0, Substream 1, Subsubstream 0
        # Substream 2: rng for selecting random feasible solutions
        self.solver.attach_rngs([MRG32k3a(s_ss_sss_index=[0, 2, 0])]) # Stream 0, Substream 2, Subsubstream 0
        # Substream 3: rng for solver's internal randomness
        rng3 = MRG32k3a(s_ss_sss_index=[0, 3, 0]) # Stream 0, Substream 3, Subsubstream 0

        # run n_macroreps of the solver on the problem
        # report the recommended solutions and corresponding intermediate budgets
        for mrep in range(n_macroreps):
            # create, initialize, and attach random number generators for oracle
            oracle_rngs = [MRG32k3a(s_ss_sss_index=[mrep + 1, ss, 0]) for ss in range(self.problem.oracle.n_rngs)]
            self.problem.oracle.attach_rngs(oracle_rngs)

            # run the solver on the problem
            recommended_solns, intermediate_budgets = self.solver.solve(problem=self.problem, crn_across_solns=crn_across_solns)
            # extract x values from recommended_solns and record
            self.all_recommended_xs.append([solution.x for solution in recommended_solns])
            # record intermediate solutions
            self.all_intermediate_budgets.append(intermediate_budgets)

    def post_replicate(self, n_postreps, n_postreps_init_opt):
        """
        Run postreplications at solutions recommended by the solver.

        Arguments
        ---------
        n_postreps : int
            number of postreplications to take at each recommended solution
        n_postreps_init_opt : int
            number of postreplications to take at initial x0 and optimal x*
        """

    def make_plots(self):
        pass