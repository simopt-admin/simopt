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
        # initialize
        self.n_macroreps = n_macroreps
        # create, initialize, and attach random number generators
        # Stream 0 is reserved for taking post-replications
        # Stream 1 is reserved for overhead ...
        # Substream 0: rng for random problem instance
        rng0 = MRG32k3a(s_ss_sss_index=[1, 0, 0]) # Stream 1, Substream 0, Subsubstream 0
        # Substream 1: rng for random initial solution x0 and restart solutions 
        rng1 = MRG32k3a(s_ss_sss_index=[1, 1, 0]) # Stream 1, Substream 1, Subsubstream 0
        # Substream 2: rng for selecting random feasible solutions
        self.solver.attach_rngs([MRG32k3a(s_ss_sss_index=[1, 2, 0])]) # Stream 1, Substream 2, Subsubstream 0
        # Substream 3: rng for solver's internal randomness
        rng3 = MRG32k3a(s_ss_sss_index=[1, 3, 0]) # Stream 1, Substream 3, Subsubstream 0

        # run n_macroreps of the solver on the problem
        # report the recommended solutions and corresponding intermediate budgets
        # Streams 2, 3, ..., n_macroreps + 1 are used for the macroreplications
        for mrep in range(self.n_macroreps):
            # create, initialize, and attach random number generators for oracle
            oracle_rngs = [MRG32k3a(s_ss_sss_index=[mrep + 2, ss, 0]) for ss in range(self.problem.oracle.n_rngs)]
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
        # initialize
        self.n_postreps = n_postreps
        self.n_postreps_init_opt = n_postreps_init_opt
        # create, initialize, and attach random number generators for oracle
        # Stream 0 is reserved for post-replications
        oracle_rngs = [MRG32k3a(s_ss_sss_index=[0, rng_index, 0]) for rng_index in range(self.problem.oracle.n_rngs)]
        self.problem.oracle.attach_rngs(oracle_rngs)
        # simulate common initial solution x0
        x0 = self.problem.initial_solution
        initial_soln = Solution(x0, self.problem)
        self.problem.simulate(solution=initial_soln, m=self.n_postreps_init_opt)
        # reset each rng to start of its current substream
        for rng in self.problem.oracle.rng_list:
            rng.reset_substream()  

        # simulate "reference" optimal solution x*
        #xstar = self.problem.ref_opt_solution ## YET UNDEFINED
        #ref_opt_soln = Solution(xstar, self.problem)
        #self.problem.simulate(solution=ref_opt_soln, m=n_postreps_init_opt)

        for mrep in range(self.n_macroreps):            
            evaluated_solns = []
            for x in self.all_recommended_xs[mrep]:
                # treat initial solution differently
                if x == x0:
                    evaluated_solns.append(initial_soln)
                else:
                    fresh_soln = Solution(x, self.problem)
                    self.problem.simulate(solution=fresh_soln, m=self.n_postreps)
                    evaluated_solns.append(fresh_soln)
                    # reset each rng to start of its current substream
                    for rng in self.problem.oracle.rng_list:
                        rng.reset_substream()  
            # record sequence of reevaluated solutions
            self.all_reevaluated_solns.append(evaluated_solns)
            # advance each rng to start of the substream = current substream + # of oracle RNGs 
            for rng in self.problem.oracle.rng_list:
                for _ in range(self.problem.oracle.n_rngs):
                    rng.advance_substream()  
        # preprocessing for subsequent call to make_plots()
        # extract all unique budget points
        repeat_budgets = [budget for budget_list in self.all_intermediate_budgets for budget in budget_list]
        unique_budgets = np.unique(repeat_budgets)
        n_inter_budgets = len(unique_budgets)
        # initialize matrix for storing all replicates of objective for each macroreplication for each budget
        self.all_post_replicates = [[[] for _ in range(n_inter_budgets)] for _ in range(self.n_macroreps)]
        # fill matrix (CAN MAKE THIS MORE PYTHONIC)
        for mrep in range(self.n_macroreps):
            for budget_index in range(n_inter_budgets):
                mrep_budget_index = np.max(np.where(np.array(self.all_intermediate_budgets[mrep]) <= unique_budgets[budget_index]))
                lookup_solution = self.all_reevaluated_solns[mrep][mrep_budget_index]
                self.all_post_replicates[mrep][budget_index] = list(lookup_solution.objectives[:lookup_solution.n_reps][0]) # 0 <- assuming only one objective
        # store point estimates of objective for each macroreplication for each budget 
        self.all_est_objective = [[np.mean(self.all_post_replicates[mrep][budget_index]) for budget_index in range(n_inter_budgets)] for mrep in range(self.n_macroreps)]      

    def make_plots(self):
        """
        Produce of the solver's performance on the problem.

        Arguments
        ---------
        beta : float
            quantile to plot, e.g., beta quantile
        """
