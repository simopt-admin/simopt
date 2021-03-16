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
from rng.mrg32k3a import MRG32k3a, bsm
from base import Solver, Problem, Oracle, Solution
import matplotlib.pyplot as plt
from scipy.stats import norm

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
    all_post_replicates : list of lists of lists
        all post-replicates from all solutions from all macroreplications
    all_est_objective : numpy array of arrays
        estimated objective values of all solutions from all macroreplications
    all_conv_curves : numpy array of arrays
        estimated convergence curves from all macroreplications
    initial_soln : Solution object
        initial solution (w/ postreplicates) used for normalization
    ref_opt_soln : Solution object
        reference optimal solution (w/ postreplicates) used for normalization
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
        # Stream 1 is reserved for bootstrapping
        # Stream 2 is reserved for overhead ...
        # Substream 0: rng for random problem instance
        rng0 = MRG32k3a(s_ss_sss_index=[2, 0, 0]) # Stream 2, Substream 0, Subsubstream 0  # UNUSED
        # Substream 1: rng for random initial solution x0 and restart solutions 
        rng1 = MRG32k3a(s_ss_sss_index=[2, 1, 0]) # Stream 2, Substream 1, Subsubstream 0  # UNUSED
        # Substream 2: rng for selecting random feasible solutions
        self.solver.attach_rngs([MRG32k3a(s_ss_sss_index=[2, 2, 0])]) # Stream 2, Substream 2, Subsubstream 0
        # Substream 3: rng for solver's internal randomness
        rng3 = MRG32k3a(s_ss_sss_index=[2, 3, 0]) # Stream 2, Substream 3, Subsubstream 0 # UNUSED

        # run n_macroreps of the solver on the problem
        # report the recommended solutions and corresponding intermediate budgets
        # Streams 3,  4, ..., n_macroreps + 2 are used for the macroreplications
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

    def post_replicate(self, n_postreps, n_postreps_init_opt, crn_across_budget=True, crn_across_macroreps=False):
        """
        Run postreplications at solutions recommended by the solver.

        Arguments
        ---------
        n_postreps : int
            number of postreplications to take at each recommended solution
        n_postreps_init_opt : int
            number of postreplications to take at initial x0 and optimal x*
        crn_across_budget : bool
            use CRN for post-replications at solutions recommended at different times?
        crn_across_macroreps : bool
            use CRN for post-replications at solutions recommended on different macroreplications?
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
        self.initial_soln = Solution(x0, self.problem)
        self.problem.simulate(solution=self.initial_soln, m=self.n_postreps_init_opt)
        if crn_across_budget==True:
            # reset each rng to start of its current substream
            for rng in self.problem.oracle.rng_list:
                rng.reset_substream()  
        # simulate "reference" optimal solution x*
        xstar = self.problem.ref_optimal_solution
        self.ref_opt_soln = Solution(xstar, self.problem)
        self.problem.simulate(solution=self.ref_opt_soln, m=self.n_postreps_init_opt)
        if crn_across_budget==True:
            # reset each rng to start of its current substream
            for rng in self.problem.oracle.rng_list:
                rng.reset_substream()
        # simulate intermediate solutions
        for mrep in range(self.n_macroreps):            
            evaluated_solns = []
            for x in self.all_recommended_xs[mrep]:
                # treat initial solution and reference solution differently
                if x == x0:
                    evaluated_solns.append(self.initial_soln)
                elif x == xstar:
                    evaluated_solns.append(self.ref_opt_soln)
                else:
                    fresh_soln = Solution(x, self.problem)
                    self.problem.simulate(solution=fresh_soln, m=self.n_postreps)
                    evaluated_solns.append(fresh_soln)
                    if crn_across_budget==True:
                        # reset each rng to start of its current substream
                        for rng in self.problem.oracle.rng_list:
                            rng.reset_substream()  
            # record sequence of reevaluated solutions
            self.all_reevaluated_solns.append(evaluated_solns)
            if crn_across_macroreps==False:
                # advance each rng to start of the substream = current substream + # of oracle RNGs 
                for rng in self.problem.oracle.rng_list:
                    for _ in range(self.problem.oracle.n_rngs):
                        rng.advance_substream()
            else: # if using CRN across macroreplications ...
                # reset each rng to start of its current substream
                for rng in self.problem.oracle.rng_list:
                    rng.reset_substream()
        # preprocessing for subsequent call to make_plots()
        # extract all unique budget points
        repeat_budgets = [budget for budget_list in self.all_intermediate_budgets for budget in budget_list]
        self.unique_budgets = np.unique(repeat_budgets)
        self.unique_frac_budgets = self.unique_budgets/self.problem.budget
        n_inter_budgets = len(self.unique_budgets)
        # initialize matrix for storing all replicates of objective for each macroreplication for each budget
        self.all_post_replicates = [[[] for _ in range(n_inter_budgets)] for _ in range(self.n_macroreps)]
        # compute signed initial optimality gap = f(x0) - f(x*)
        initial_obj_val = np.mean(self.initial_soln.objectives[:self.initial_soln.n_reps][:,0]) # 0 <- assuming only one objective
        ref_opt_obj_val = np.mean(self.ref_opt_soln.objectives[:self.ref_opt_soln.n_reps][:,0]) # 0 <- assuming only one objective
        initial_opt_gap = initial_obj_val - ref_opt_obj_val
        # fill matrix (CAN MAKE THIS MORE PYTHONIC)
        for mrep in range(self.n_macroreps):
            for budget_index in range(n_inter_budgets):
                mrep_budget_index = np.max(np.where(np.array(self.all_intermediate_budgets[mrep]) <= self.unique_budgets[budget_index]))
                lookup_solution = self.all_reevaluated_solns[mrep][mrep_budget_index]
                self.all_post_replicates[mrep][budget_index] = list(lookup_solution.objectives[:lookup_solution.n_reps][:,0]) # 0 <- assuming only one objective
        # store point estimates of objective for each macroreplication for each budget 
        self.all_est_objective = [[np.mean(self.all_post_replicates[mrep][budget_index]) for budget_index in range(n_inter_budgets)] for mrep in range(self.n_macroreps)]      
        # store convergence curve values for each macroreplication for each budget
        self.all_conv_curves = [[(self.all_est_objective[mrep][budget_index] - ref_opt_obj_val)/initial_opt_gap for budget_index in range(n_inter_budgets)] for mrep in range(self.n_macroreps)]

    def make_plots(self, plot_type, beta=0.50, normalize=True, plot_CIs=True):
        """
        Produce plots of the solver's performance on the problem.

        Arguments
        ---------
        plot_type : string
            indicates which type of plot to produce
                "all" : all estimated convergence curves
                "mean" : estimated mean convergence curve
                "quantile" : estimated beta quantile convergence curve
        beta : float
            quantile to plot, e.g., beta quantile
        normalize : Boolean
            normalize convergence curves w.r.t. optimality gaps?
        plot_CIs : Boolean
            plot bootstrapping confidence intervals?
        """
        # set up plot
        stylize_plot(plot_type=plot_type, normalize=normalize, budget=self.problem.budget, beta=beta)
        # make the plot
        if plot_type == "all":
            # plot all estimated convergence curves
            if normalize == True:
                for mrep in range(self.n_macroreps):
                    plt.step(self.unique_frac_budgets, self.all_conv_curves[mrep], where='post')
            else: # unnormalized
                for mrep in range(self.n_macroreps):
                    plt.step(self.unique_budgets, self.all_est_objective[mrep], where='post')
        elif plot_type == "mean":
            # plot estimated mean convergence curve
            if normalize == True:
                estimator = np.mean(self.all_conv_curves, axis=0)
                plt.step(self.unique_frac_budgets, estimator, 'b-', where='post')
            else: # unnormalized
                estimator = np.mean(self.all_est_objective, axis=0)
                plt.step(self.unique_budgets, estimator, 'b-', where='post')
        elif plot_type == "quantile":
            # plot estimated beta quantile convergence curve
            if normalize == True:
                estimator = np.quantile(self.all_conv_curves, q=beta, axis=0)
                plt.step(self.unique_frac_budgets, estimator, 'b-', where='post')
            else: # unnormalized
                estimator = np.quantile(self.all_est_objective, q=beta, axis=0)
                plt.step(self.unique_budgets, estimator, 'b-', where='post')
        else:
            print("Not a valid plot type.")
        if plot_type=="mean" or plot_type=="quantile":
            # report bootstrapping error estimation and optionally plot bootstrap CIs
            self.plot_bootstrap_CIs(plot_type, normalize, estimator, plot_CIs, beta)
        save_plot(plot_type=plot_type, normalize=normalize)

    def bootstrap_sample(self, bootstrap_rng, crn_across_budget=True, crn_across_macroreps=False):
        """
        Generate a bootstrap sample of estimated convergence curves (normalized and unnormalized).

        Arguments
        ---------
        bootstrap_rng : MRG32k3a object
            random number generator to use for bootstrapping
        crn_across_budget : bool
            use CRN for resampling postreplicates at solutions recommended at different times?
        crn_across_macroreps : bool
            use CRN for resampling postreplicates at solutions recommended on different macroreplications?

        Returns
        -------
        bootstrap_est_objective : numpy array of arrays
            bootstrapped estimated objective values of all solutions from all macroreplications
        bootstrap_conv_curves : numpy array of arrays
            bootstrapped estimated convergence curves from all macroreplications
        """
        # initialize matrix for bootstrap estimated convergence curves
        bootstrap_est_objective = np.empty((self.n_macroreps, len(self.unique_budgets)))
        bootstrap_conv_curves = np.empty((self.n_macroreps, len(self.unique_budgets)))
        # uniformly resample M macroreplications (with replacement) from 0, 1, ..., M-1
        # subsubstream 0 is reserved for this outer-level bootstrapping
        mreps = bootstrap_rng.choices(range(self.n_macroreps), k=self.n_macroreps)
        # advance random number generator subsubstream to prepare for inner-level bootstrapping
        bootstrap_rng.advance_subsubstream()
        # subsubstream 1 is reserved for bootstrapping at initial solution x0 and reference optimal solution x*
        # bootstrap sample postreplicates at common initial solution x0
        # uniformly resample L postreps (with replacement) from 0, 1, ..., L
        postreps = bootstrap_rng.choices(range(self.n_postreps_init_opt), k=self.n_postreps_init_opt)
        # compute the mean of the resampled postreplications
        bs_initial_obj_val = np.mean([self.initial_soln.objectives[postrep,0] for postrep in postreps])
        # reset subsubstream if using CRN across budgets
        # this means the same postreplication indices will be used for resampling at x0 and x*
        if crn_across_budget == True:
            bootstrap_rng.reset_subsubstream()
        # bootstrap sample postreplicates at reference optimal solution x*
        # uniformly resample L postreps (with replacement) from 0, 1, ..., L
        postreps = bootstrap_rng.choices(range(self.n_postreps_init_opt), k=self.n_postreps_init_opt)
        # compute the mean of the resampled postreplications
        bs_ref_opt_obj_val = np.mean([self.ref_opt_soln.objectives[postrep,0] for postrep in postreps])
        # compute initial optimality gap
        bs_initial_opt_gap = bs_initial_obj_val - bs_ref_opt_obj_val
        # advance random number generator subsubstream to prepare for inner-level bootstrapping
        # will now be at start of subsubstream 2
        bootstrap_rng.advance_subsubstream()
        # bootstrap within each bootstrapped macroreplication    
        for bs_mrep in range(self.n_macroreps):
            mrep = mreps[bs_mrep]
            # inner-level bootstrapping over intermediate recommended solutions
            for budget in range(len(self.unique_budgets)):
                # if solution is x0
                if np.array_equal(self.initial_soln.objectives[0:self.n_postreps_init_opt,0], self.all_post_replicates[mrep][budget]):
                    # plug in fixed bootstrapped f(x0)
                    bootstrap_est_objective[bs_mrep][budget] = bs_initial_obj_val
                # elif solution is x*
                elif np.array_equal(self.ref_opt_soln.objectives[0:self.n_postreps_init_opt,0], self.all_post_replicates[mrep][budget]):
                    # plug in fixed bootstrapped f(x*)
                    bootstrap_est_objective[bs_mrep][budget] = bs_ref_opt_obj_val
                else: # else solution other than x0 or x*
                    # uniformly resample N postreps (with replacement) from 0, 1, ..., N-1
                    postreps = bootstrap_rng.choices(range(self.n_postreps), k=self.n_postreps)
                    # compute the mean of the resampled postreplications
                    bootstrap_est_objective[bs_mrep][budget] = np.mean([self.all_post_replicates[mrep][budget][postrep] for postrep in postreps])
                # normalize the estimated objective function value
                bootstrap_conv_curves[bs_mrep][budget] = (bootstrap_est_objective[bs_mrep][budget] - bs_ref_opt_obj_val)/bs_initial_opt_gap
                # reset subsubstream if using CRN across budgets
                if crn_across_budget == True:
                    bootstrap_rng.reset_subsubstream()
            # advance subsubstream if not using CRN across macroreps
            if crn_across_macroreps == False:
                bootstrap_rng.advance_subsubstream()
            else: # if using CRN across macroreplications
                # reset subsubstream
                bootstrap_rng.reset_subsubstream()
        # advance substream of random number generator to prepare for next bootstrap sample
        bootstrap_rng.advance_substream()
        return bootstrap_est_objective, bootstrap_conv_curves

    def bootstrap_CI(self, plot_type, normalize, estimator, n_bootstraps=100, conf_level=0.95, bias_correction=True, beta=0.50):
        """
        Construct bootstrap confidence intervals and compute max half-width.

        Arguments
        ---------
        plot_type : string
            indicates which type of plot to produce
                "all" : all estimated convergence curves
                "mean" : estimated mean convergence curve
                "quantile" : estimated beta quantile convergence curve
        normalize : Boolean
            normalize convergence curves w.r.t. optimality gaps?
        estimator : numpy array
            estimated mean or quantile convergence curve
        n_bootstraps : int > 0
            number of times to generate a bootstrap sample of estimated convergence curves
        conf_level : float in (0,1)
            confidence level for confidence intervals, i.e., 1-alpha
        bias_correction : bool
            use bias-corrected bootstrap CIs (via percentile method)?
        beta : float
            quantile for quantile aggregate convergence curve, e.g., beta quantile 
        
        Returns
        -------
        bs_CI_lower_bounds : numpy array
            lower bounds of bootstrap CIs at all budgets
        bs_CI_upper_bounds : numpy array
            upper bounds of bootstrap CIs at all budgets
        max_halfwidth : float
            maximum halfwidth of all bootstrap confidence intervals constructed
        """
        # create random number generator for bootstrap sampling
        # Stream 1 dedicated for bootstrapping
        bootstrap_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
        # initialize storage for bootstrap aggregate curves
        bs_aggregate_curves = np.zeros((n_bootstraps, len(self.unique_budgets)))
        for bs_index in range(n_bootstraps):
            # generate bootstrap sample of estimated convergence curves
            bootstrap_est_objective, bootstrap_conv_curves = self.bootstrap_sample(bootstrap_rng=bootstrap_rng, crn_across_budget=True, crn_across_macroreps=False)
            # apply the functional of the bootstrap sample, e.g., mean/quantile (aggregate) convergence curve
            if plot_type == "mean": # mean convergence curve
                if normalize == True: 
                    bs_aggregate_curves[bs_index] = np.mean(bootstrap_conv_curves, axis=0)
                else:
                    bs_aggregate_curves[bs_index] = np.mean(bootstrap_est_objective, axis=0)
            elif plot_type == "quantile": # quantile convergence curve
                if normalize == True:
                    bs_aggregate_curves[bs_index] = np.quantile(bootstrap_conv_curves, q=beta, axis=0)
                else:
                    bs_aggregate_curves[bs_index] = np.quantile(bootstrap_est_objective, q=beta, axis=0)
        # compute bootstrapping confidence intervals via percentile method
        # see Efron and Gong (1983) "A leisurely look at the bootstrap, the jackknife, and cross-validation"
        if bias_correction == True: # if bias-corrected CIs
            # see equation (17) on page 41
            z0s = [norm.ppf(np.mean(bs_aggregate_curves[:,budget] < estimator[budget])) for budget in range(len(self.unique_budgets))]
            zconflvl = norm.ppf(conf_level)
            q_lowers = [norm.cdf(2*z0 - zconflvl) for z0 in z0s]
            q_uppers = [norm.cdf(2*z0 + zconflvl) for z0 in z0s]
            bs_CI_lower_bounds = np.array([np.quantile(bs_aggregate_curves[:,budget], q=q_lowers[budget]) for budget in range(len(self.unique_budgets))])
            bs_CI_upper_bounds = np.array([np.quantile(bs_aggregate_curves[:,budget], q=q_uppers[budget]) for budget in range(len(self.unique_budgets))])
        else: # if not bias-corrected CIs
            # see equation (16) on page 41
            q_lower = (1-conf_level)/2
            q_upper = 1-(1-conf_level)/2
            bs_CI_lower_bounds = np.quantile(bs_aggregate_curves, q=q_lower, axis=0)
            bs_CI_upper_bounds = np.quantile(bs_aggregate_curves, q=q_upper, axis=0)
        max_halfwidth = np.max((bs_CI_upper_bounds - bs_CI_lower_bounds)/2)
        return bs_CI_lower_bounds, bs_CI_upper_bounds, max_halfwidth

    def plot_bootstrap_CIs(self, plot_type, normalize, estimator, plot_CIs, beta):
        """
        Optionally plot bootstrap confidence intervals and report max half-width.

        Arguments
        ---------
        plot_type : string
            indicates which type of plot to produce
                "all" : all estimated convergence curves
                "mean" : estimated mean convergence curve
                "quantile" : estimated beta quantile convergence curve
        normalize : Boolean
            normalize convergence curves w.r.t. optimality gaps?
        estimator : numpy array
            estimated mean or quantile convergence curve
        plot_CIs : Boolean
            plot bootstrapping confidence intervals?
        beta : float (optional)
            quantile for quantile aggregate convergence curve, e.g., beta quantile   
        """
        # construct bootstrap confidence intervals, plot, and print caption
        bs_CI_lower_bounds, bs_CI_upper_bounds, max_halfwidth = self.bootstrap_CI(plot_type=plot_type, normalize=normalize, estimator=estimator, beta=beta)
        if normalize == True:
            budgets = self.unique_frac_budgets
            xloc = 0.05
            yloc = -0.35
        else:
            budgets = self.unique_budgets
            xloc = 0.05*self.problem.budget
            yloc = min(bs_CI_lower_bounds) - 0.25*(max(bs_CI_upper_bounds) - min(bs_CI_lower_bounds))
        if plot_CIs == True:
            # plot bootstrap confidence intervals
            plt.step(budgets, bs_CI_lower_bounds, 'b--', where='post')
            plt.step(budgets, bs_CI_upper_bounds, 'b--', where='post')
        # print caption about max halfwidth
        txt = "The max halfwidth of the bootstrap CIs is " + str(round(max_halfwidth,2)) + "."
        plt.text(x=xloc, y=yloc, s=txt)

def stylize_plot(plot_type, normalize, budget=None, beta=None):
    """
    Create new figure. Add labels to plot and reformat axes.

    Arguments
    ---------
    plot_type : string
        indicates which type of plot to produce
            "all" : all estimated convergence curves
            "mean" : estimated mean convergence curve
            "quantile" : estimated beta quantile convergence curve
    normalize : Boolean
        normalize convergence curves w.r.t. optimality gaps?
    budget : int
        budget of problem, measured in function evaluations
    beta : float (optional)
        quantile for quantile aggregate convergence curve, e.g., beta quantile   
    """
    plt.figure()
    # format axes and axis labels
    if normalize==True:
        xlabel = "Fraction of Budget"
        ylabel = "Fraction of Initial Optimality Gap"
        xlim = (0, 1)
        ylim = (-0.1, 1.1)
        title = "Solver Name on Problem Name \n"
    elif normalize==False:
        xlabel = "Budget"
        ylabel = "Objective Function Value"
        xlim = (0, budget)
        ylim = None
        title = "Solver Name on Problem Name \n" + "Unnormalized "
    # format title
    if plot_type=="all":
        title = title + "Estimated Convergence Curves"
    elif plot_type=="mean":
        title = title + "Mean Convergence Curve"
    elif plot_type=="quantile":
        title = title + str(round(beta,2)) + "-Quantile Convergence Curve"
    # add axis labels
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    # add title
    plt.title(title, size=14)
    # format axes and tick marks
    plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tick_params(axis='both', which='major', labelsize=12) 

def save_plot(plot_type, normalize):
    """
    Create new figure. Add labels to plot and reformat axes.

    Arguments
    ---------
    plot_type : string
        indicates which type of plot to produce
            "all" : all estimated convergence curves
            "mean" : estimated mean convergence curve
            "quantile" : estimated beta quantile convergence curve
    normalize : Boolean
        normalize convergence curves w.r.t. optimality gaps? 
    """
    # form string name for plot
    if plot_type == "all":
        plot_name = "all_conv_curves"
    elif plot_type == "mean":
        plot_name = "mean_conv_curve"
    elif plot_type == "quantile":
        plot_name = "quantile_conv_curve"
    if normalize==False:
        plot_name = plot_name + "_unnorm"
    path_name = "experiments/plots/" + plot_name + ".png"
    # save figure to folder as .png
    plt.savefig(path_name, bbox_inches="tight")

def areas_under_conv_curve(conv_curve, frac_inter_budgets):
    """
    Compute the area under a normalized estimated convergence curve.

    Arguments
    ---------
    conv_curve : numpy array
        normalized estimated convergence curves for a macroreplication
    frac_inter_budgets : numpy array
        fractions of budget at which the convergence curve is defined

    Returns
    -------
    area : float
        area under each estimated convergence curve
    """
    area = np.dot(conv_curve[:-1], np.diff(frac_inter_budgets))
    return area

def solve_time_of_conv_curve(conv_curve, frac_inter_budgets, solve_tol):
    """
    Compute the solve time of a normalized estimated convergence curve.

    Arguments
    ---------
    conv_curve : numpy array
        normalized estimated convergence curves for a macroreplication
    frac_inter_budgets : numpy array
        fractions of budget at which the convergence curve is defined
    solve_tol : float in (0,1)
        tolerance for a problem to be solved, relative to initial optimality gap
        
    Returns
    -------
    solve_time : float
        time at which the normalized convergence curve first drops below solve_tol
        i.e., the "alpha" solve time
    """
    # solve_time defined as infinity if the problem is not solved to within solve_tol
    solve_time = np.inf
    # pass over convergence curve to find first solve_tol crossing time
    for i in range(len(conv_curve)):
        if conv_curve[i] < solve_tol:
            solve_time = frac_inter_budgets[i]
            break
    return solve_time