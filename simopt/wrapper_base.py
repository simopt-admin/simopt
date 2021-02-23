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
    all_reevaluated_solns : list of Solution objects
        reevaluated solutions recommended by the solver
    """
    def __init__(self):
        self.all_recommended_xs = []
        self.all_intermediate_budgets = []
        self.all_reevaluated_solns = []

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
        xstar = self.problem.ref_optimal_solution
        ref_opt_soln = Solution(xstar, self.problem)
        self.problem.simulate(solution=ref_opt_soln, m=n_postreps_init_opt)
        # reset each rng to start of its current substream
        for rng in self.problem.oracle.rng_list:
            rng.reset_substream()
        # simulate intermediate solutions
        for mrep in range(self.n_macroreps):            
            evaluated_solns = []
            for x in self.all_recommended_xs[mrep]:
                # treat initial solution and reference solution differently
                if x == x0:
                    evaluated_solns.append(initial_soln)
                elif x == xstar:
                    evaluated_solns.append(ref_opt_soln)
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
        self.unique_budgets = np.unique(repeat_budgets)
        self.unique_frac_budgets = self.unique_budgets/self.problem.budget
        n_inter_budgets = len(self.unique_budgets)
        # initialize matrix for storing all replicates of objective for each macroreplication for each budget
        self.all_post_replicates = [[[] for _ in range(n_inter_budgets)] for _ in range(self.n_macroreps)]
        # initialize matrix for storing all convergence curve values for each macroreplication for each budget
        self.all_conv_curves = [[[] for _ in range(n_inter_budgets)] for _ in range(self.n_macroreps)]
        # compute signed initial optimality gap = f(x0) - f(x*)
        initial_obj_val = initial_soln.objectives[:initial_soln.n_reps][0] # 0 <- assuming only one objective
        ref_opt_obj_val = ref_opt_soln.objectives[:ref_opt_soln.n_reps][0] # 0 <- assuming only one objective
        initial_opt_gap = initial_obj_val - ref_opt_obj_val
        # fill matrix (CAN MAKE THIS MORE PYTHONIC)
        for mrep in range(self.n_macroreps):
            for budget_index in range(n_inter_budgets):
                mrep_budget_index = np.max(np.where(np.array(self.all_intermediate_budgets[mrep]) <= self.unique_budgets[budget_index]))
                lookup_solution = self.all_reevaluated_solns[mrep][mrep_budget_index]
                lookup_solution_obj_val = lookup_solution.objectives[:lookup_solution.n_reps][0] # 0 <- assuming only one objective
                self.all_post_replicates[mrep][budget_index] = list(lookup_solution_obj_val)
                current_opt_gap = lookup_solution_obj_val - ref_opt_obj_val 
                self.all_conv_curves[mrep][budget_index] = list(current_opt_gap/initial_opt_gap)
        # store point estimates of objective for each macroreplication for each budget 
        self.all_est_objective = [[np.mean(self.all_post_replicates[mrep][budget_index]) for budget_index in range(n_inter_budgets)] for mrep in range(self.n_macroreps)]      

    def make_plots(self, plot_type, beta=0.95, normalize=True):
        """
        Produce plots of the solver's performance on the problem.

        Arguments
        ---------
        plot_type : string
            indicates which type of plot to produce
                "all" : all estimated convergence curves
                "mean" : estimated mean convergence curve
                "quantile" : estimated beta quantile convergence curve
                "all" : all estimated convergence curves
                "mean" : estimated mean convergence curve
                "quantile" : estimated beta quantile convergence curve
        beta : float
            quantile to plot, e.g., beta quantile
        """
        if plot_type == "all":
            # plot all estimated convergence curves
            if normalize == True:
                plt.figure()
                for mrep in range(self.n_macroreps):
                    plt.step(self.unique_frac_budgets, self.all_conv_curves[mrep], where='post')
                self.stylize_plot(
                    xlabel = "Fraction of Budget",
                    ylabel = "Fraction of Initial Optimality Gap",
                    title = "Solver Name on Problem Name \n" + "Estimated Convergence Curves",
                    xlim = (0, 1),
                    ylim = (0, 1.1)
                )
                plt.savefig('experiments/plots/all_conv_curves.png', bbox_inches='tight')
            else: # unnormalized
                plt.figure()
                for mrep in range(self.n_macroreps):
                    plt.step(self.unique_budgets, self.all_est_objective[mrep], where='post')
                self.stylize_plot(
                    xlabel = "Budget",
                    ylabel = "Objective Function Value",
                    title = "Solver Name on Problem Name \n" + "Unnormalized Estimated Convergence Curves",
                    xlim = (0, self.problem.budget),
                )
                plt.savefig('experiments/plots/all_conv_curves_unnorm.png', bbox_inches='tight')
        elif plot_type == "mean":
            # plot estimated mean convergence curve
            if normalize == True:
                plt.figure()
                plt.step(self.unique_frac_budgets, np.mean(self.all_conv_curves, axis=0))
                self.stylize_plot(
                    xlabel = "Fraction of Budget",
                    ylabel = "Fraction of Initial Optimality Gap",
                    title = "Solver Name on Problem Name \n" + "Estimated Mean Convergence Curve",
                    xlim = (0, 1),
                    ylim = (0, 1.1)
                )
                plt.savefig('experiments/plots/mean_conv_curve.png', bbox_inches='tight')
            else: # unnormalized
                plt.figure()
                plt.step(self.unique_budgets, np.mean(self.all_est_objective, axis=0))
                self.stylize_plot(
                    xlabel = "Budget",
                    ylabel = "Objective Function Value",
                    title = "Solver Name on Problem Name \n" + "Unnormalized Estimated Mean Convergence Curve",
                    xlim = (0, self.problem.budget)
                )
                plt.savefig('experiments/plots/mean_conv_curve_unnorm.png', bbox_inches='tight')
        elif plot_type == "quantile":
            # plot estimated beta quantile convergence curve
            if normalize == True:
                plt.figure()
                plt.step(self.unique_frac_budgets, np.quantile(self.all_conv_curves, q=beta, axis=0))
                self.stylize_plot(
                    xlabel = "Fraction of Budget",
                    ylabel = "Fraction of Initial Optimality Gap",
                    title = "Solver Name on Problem Name \n" + "Estimated Quantile Convergence Curve",
                    xlim = (0, 1),
                    ylim = (0, 1.1)
                )
                plt.savefig('experiments/plots/quantile_conv_curve.png', bbox_inches='tight')
            else: # unnormalized
                plt.figure()
                plt.step(self.unique_budgets, np.quantile(self.all_est_objective, q=beta, axis=0))
                self.stylize_plot(
                    xlabel = "Budget",
                    ylabel = "Objective Function Value",
                    title = "Solver Name on Problem Name \n" + "Unnormalized Estimated Quantile Convergence Curve",
                    xlim = (0, self.problem.budget)
                )
                plt.savefig('experiments/plots/quantile_conv_curve_unnorm.png', bbox_inches='tight')
        else:
            print("Not a valid plot type.")

    def stylize_plot(self, xlabel, ylabel, title, xlim, ylim=None):
        """
        Add labels to plots and reformat axes.

        Arguments
        ---------
        xlabel : string
            label for x axis
        ylabel : string
            label for y axis
        title : string
            title for plot
        xlim : 2-tuple
            (lower x limit, upper x limit)
        ylim : 2-tuple
            (lower y limit, upper y limit)
        """
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)