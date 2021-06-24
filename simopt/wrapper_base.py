#!/usr/bin/env python
"""
Summary
-------
Provide base classes for experiments and meta experiments.
Plus helper functions for reading/writing data and plotting.

Listing
-------
Experiment : class
read_experiment_results : function
stylize_plot : function
stylize_solvability_plot : function
stylize_difference_plot : function
stylize_area_plot : function
save_plot : function
area_under_prog_curve : function
solve_time_of_prog_curve : function
MetaExperiment : class
compute_difference_solvability_profile : function
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import endswith
from scipy.stats import norm
import pickle
import importlib
from copy import deepcopy


from rng.mrg32k3a import MRG32k3a
from base import Solution
from directory import solver_directory, problem_directory


class Experiment(object):
    """
    Base class for running one solver on one problem.

    Attributes
    ----------
    solver : base.Solver object
        simulation-optimization solver
    problem : base.Problem object
        simulation-optimization problem
    n_macroreps : int > 0
        number of macroreplications run
    file_name_path : str
        path of .pickle file for saving wrapper_base.Experiment object
    all_recommended_xs : list of lists of tuples
        sequences of recommended solutions from each macroreplication
    all_intermediate_budgets : list of lists
        sequences of intermediate budgets from each macroreplication
    n_postreps : int
        number of postreplications to take at each recommended solution
    n_postreps_init_opt : int
        number of postreplications to take at initial x0 and optimal x*
    crn_across_budget : bool
        use CRN for post-replications at solutions recommended at different times?
    crn_across_macroreps : bool
        use CRN for post-replications at solutions recommended on different macroreplications?
    all_reevaluated_solns : list of Solution objects
        reevaluated solutions recommended by the solver
    all_post_replicates : list of lists of lists
        all post-replicates from all solutions from all macroreplications
    all_est_objective : numpy array of arrays
        estimated objective values of all solutions from all macroreplications
    all_prog_curves : numpy array of arrays
        estimated progress curves from all macroreplications
    initial_soln : base.Solution object
        initial solution (w/ postreplicates) used for normalization
    ref_opt_soln : base.Solution object
        reference optimal solution (w/ postreplicates) used for normalization
    areas : list of floats
        areas under each estimated progress curve
    area_mean : float
        sample mean area under estimated progress curves
    area_std_dev : float
        sample standard deviation of area under estimated progress curves
    area_mean_CI : numpy array of length 2
        bootstrap CI of the form [lower bound, upper bound] for mean area
    area_std_dev_CI : numpy array of length 2
        bootstrap CI of the form [lower_bound, upper_bound] for std dev of area
    solve_tols : list of floats in (0,1]
        relative optimality gap(s) definining when a problem is solved
    solve_times = list of lists of floats
        solve_tol solve times for each estimated progress curve for each solve_tol
    solve_time_quantiles : list of floats
        beta quantile of solve times for each solve_tole
    solve_time_quantiles_CIs : list of numpy arrays of length 2
        bootstrap CI of the form [lower bound, upper bound] for quantile of solve time
        for each solve_tol

    Arguments
    ---------
    solver_name : string
        name of solver
    problem_name : string
        name of problem
    solver_fixed_factors : dict
        dictionary of user-specified solver factors
    problem_fixed_factors : dict
        dictionary of user-specified problem factors
    oracle_fixed_factors : dict
        dictionary of user-specified oracle factors
    file_name_path : str
        path of .pickle file for saving wrapper_base.Experiment object
    """
    def __init__(self, solver_name, problem_name, solver_fixed_factors={}, problem_fixed_factors={}, oracle_fixed_factors={}, file_name_path=None):
        self.solver = solver_directory[solver_name](fixed_factors=solver_fixed_factors)
        self.problem = problem_directory[problem_name](fixed_factors=problem_fixed_factors, oracle_fixed_factors=oracle_fixed_factors)
        if file_name_path is None:
            self.file_name_path = "./experiments/outputs/" + self.solver.name + "_on_" + self.problem.name + ".pickle"
        else:
            self.file_name_path = file_name_path

    def run(self, n_macroreps):
        """
        Run n_macroreps of the solver on the problem.

        Arguments
        ---------
        n_macroreps : int
            number of macroreplications of the solver to run on the problem
        """
        self.n_macroreps = n_macroreps
        self.all_recommended_xs = []
        self.all_intermediate_budgets = []
        # Create, initialize, and attach random number generators
        #     Stream 0: reserved for taking post-replications
        #     Stream 1: reserved for bootstrapping
        #     Stream 2: reserved for overhead ...
        #         Substream 0: rng for random problem instance
        #         Substream 1: rng for random initial solution x0 and
        #                      restart solutions
        #         Substream 2: rng for selecting random feasible solutions
        #         Substream 3: rng for solver's internal randomness
        #     Streams 3, 4, ..., n_macroreps + 2: reserved for
        #                                         macroreplications
        rng0 = MRG32k3a(s_ss_sss_index=[2, 0, 0])  # unused
        rng1 = MRG32k3a(s_ss_sss_index=[2, 1, 0])  # unused
        rng2 = MRG32k3a(s_ss_sss_index=[2, 2, 0])
        # self.solver.attach_rngs([MRG32k3a(s_ss_sss_index=[2, 2, 0])])
        rng3 = MRG32k3a(s_ss_sss_index=[2, 3, 0])  # unused
        self.solver.attach_rngs([rng1, rng2, rng3])
        # Run n_macroreps of the solver on the problem.
        # Report recommended solutions and corresponding intermediate budgets.
        for mrep in range(self.n_macroreps):
            print("Running macroreplication " + str(mrep + 1) + " of " + str(self.n_macroreps) + ".")
            # Create, initialize, and attach RNGs used for simulating solutions.
            progenitor_rngs = [MRG32k3a(s_ss_sss_index=[mrep + 2, ss, 0]) for ss in range(self.problem.oracle.n_rngs)]
            self.solver.solution_progenitor_rngs = progenitor_rngs
            # print([rng.s_ss_sss_index for rng in progenitor_rngs])
            # Run the solver on the problem.
            recommended_solns, intermediate_budgets = self.solver.solve(problem=self.problem)
            # Trim solutions recommended after final budget
            recommended_solns, intermediate_budgets = trim_solver_results(problem=self.problem, recommended_solns=recommended_solns, intermediate_budgets=intermediate_budgets)
            # Extract decision-variable vectors (x) from recommended solutions.
            # Record recommended solutions and intermediate budgets.
            self.all_recommended_xs.append([solution.x for solution in recommended_solns])
            self.all_intermediate_budgets.append(intermediate_budgets)
        # Save Experiment object to .pickle file.
        self.record_experiment_results()

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
        self.n_postreps = n_postreps
        self.n_postreps_init_opt = n_postreps_init_opt
        self.crn_across_budget = crn_across_budget
        self.crn_across_macroreps = crn_across_macroreps
        self.all_reevaluated_solns = []
        # Create, initialize, and attach RNGs for oracle.
        # Stream 0: reserved for post-replications.
        baseline_rngs = [MRG32k3a(s_ss_sss_index=[0, rng_index, 0]) for rng_index in range(self.problem.oracle.n_rngs)]
        # Copy rngs to use for later simulating intial solution x0 and
        # reference optimal solution x*.
        copied_baseline_rngs = [deepcopy(rng) for rng in baseline_rngs]
        # Skip over first set of substreams dedicated for sampling x0 and x*.
        # Advance each rng to start of
        #     substream = current substream + # of oracle RNGs.
        for rng in baseline_rngs:
            for _ in range(self.problem.oracle.n_rngs):
                rng.advance_substream()
        # Simulate intermediate recommended solutions.
        for mrep in range(self.n_macroreps):
            evaluated_solns = []
            for x in self.all_recommended_xs[mrep]:
                fresh_soln = Solution(x, self.problem)
                fresh_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
                self.problem.simulate(solution=fresh_soln, m=self.n_postreps)
                evaluated_solns.append(fresh_soln)
                if crn_across_budget:
                    # Reset each rng to start of its current substream.
                    for rng in baseline_rngs:
                        rng.reset_substream()
            # Record sequence of reevaluated solutions.
            self.all_reevaluated_solns.append(evaluated_solns)
            if crn_across_macroreps:
                # Reset each rng to start of its current substream.
                for rng in baseline_rngs:
                    rng.reset_substream()
            else:
                # Advance each rng to start of
                #     substream = current substream + # of oracle RNGs.
                for rng in baseline_rngs:
                    for _ in range(self.problem.oracle.n_rngs):
                        rng.advance_substream()
        # Simulate common initial solution x0.
        x0 = self.problem.factors["initial_solution"]
        self.initial_soln = Solution(x0, self.problem)
        self.initial_soln.attach_rngs(rng_list=copied_baseline_rngs, copy=False)
        self.problem.simulate(solution=self.initial_soln, m=self.n_postreps_init_opt)
        if crn_across_budget:
            # Reset each rng to start of its current substream.
            for rng in copied_baseline_rngs:
                rng.reset_substream()
        # Determine reference optimal solution x*.
        if self.problem.ref_optimal_solution is not None:
            xstar = self.problem.ref_optimal_solution
        else:  # Look up estimated best solution recommended over all macroreplications.
            # TO DO: Simplify this block of code.
            # TO DO: Handle duplicate argmaxs.
            # TO DO: Currently 0 <- assuming only one objective. Generalize.
            best_est_objectives = np.zeros(self.n_macroreps)
            for mrep in range(self.n_macroreps):
                est_objectives = np.array([np.mean(rec_soln.objectives[:self.n_postreps][:, 0]) for rec_soln in self.all_reevaluated_solns[mrep]])
                best_est_objectives[mrep] = np.max(self.problem.minmax[0]*est_objectives)
            best_mrep = np.argmax(self.problem.minmax[0]*best_est_objectives)
            best_mrep_est_objectives = np.array([np.mean(rec_soln.objectives[:self.n_postreps][:, 0]) for rec_soln in self.all_reevaluated_solns[best_mrep]])
            best_soln_index = np.argmax(self.problem.minmax[0]*best_mrep_est_objectives)
            xstar = self.all_reevaluated_solns[best_mrep][best_soln_index].x
        # Simulate reference optimal solution x*.
        self.ref_opt_soln = Solution(xstar, self.problem)
        self.ref_opt_soln.attach_rngs(rng_list=copied_baseline_rngs, copy=False)
        self.problem.simulate(solution=self.ref_opt_soln, m=self.n_postreps_init_opt)
        # Replace recommended solutions corresponding to x0 and x* with resimulated versions.
        for mrep in range(self.n_macroreps):
            for soln_index in range(len(self.all_reevaluated_solns[mrep])):
                if self.all_reevaluated_solns[mrep][soln_index].x == x0:
                    self.all_reevaluated_solns[mrep][soln_index] = self.initial_soln
                elif self.all_reevaluated_solns[mrep][soln_index].x == xstar:
                    self.all_reevaluated_solns[mrep][soln_index] = self.ref_opt_soln
        # Preprocessing in anticipation of plotting.
        # Extract all unique budget points.
        repeat_budgets = [budget for budget_list in self.all_intermediate_budgets for budget in budget_list]
        self.unique_budgets = np.unique(repeat_budgets)
        self.unique_frac_budgets = self.unique_budgets / self.problem.factors["budget"]
        n_inter_budgets = len(self.unique_budgets)
        # Compute signed initial optimality gap = f(x0) - f(x*);
        initial_obj_val = np.mean(self.initial_soln.objectives[:self.initial_soln.n_reps][:, 0])  # 0 <- assuming only one objective
        ref_opt_obj_val = np.mean(self.ref_opt_soln.objectives[:self.ref_opt_soln.n_reps][:, 0])  # 0 <- assuming only one objective
        initial_opt_gap = initial_obj_val - ref_opt_obj_val
        # Populate matrix containing
        #     all replicates of objective,
        #     for each macroreplication,
        #     for each budget.
        self.all_post_replicates = [[[] for _ in range(n_inter_budgets)] for _ in range(self.n_macroreps)]
        for mrep in range(self.n_macroreps):
            for budget_index in range(n_inter_budgets):
                mrep_budget_index = np.max(np.where(np.array(self.all_intermediate_budgets[mrep]) <= self.unique_budgets[budget_index]))
                lookup_solution = self.all_reevaluated_solns[mrep][mrep_budget_index]
                self.all_post_replicates[mrep][budget_index] = list(lookup_solution.objectives[:lookup_solution.n_reps][:, 0])  # 0 <- assuming only one objective
        # Store estimated objective and progress curve values
        # for each macrorep for each budget.
        self.all_est_objective = [[np.mean(self.all_post_replicates[mrep][budget_index]) for budget_index in range(n_inter_budgets)] for mrep in range(self.n_macroreps)]
        self.all_prog_curves = [[(self.all_est_objective[mrep][budget_index] - ref_opt_obj_val) / initial_opt_gap for budget_index in range(n_inter_budgets)] for mrep in range(self.n_macroreps)]
        # Save Experiment object to .pickle file.
        self.record_experiment_results()

    def plot_progress_curves(self, plot_type, beta=0.50, normalize=True, plot_CIs=True):
        """
        Produce plots of the solver's performance on the problem.

        Arguments
        ---------
        plot_type : string
            indicates which type of plot to produce
                "all" : all estimated progress curves
                "mean" : estimated mean progress curve
                "quantile" : estimated beta quantile progress curve
        beta : float in (0,1)
            quantile to plot, e.g., beta quantile
        normalize : Boolean
            normalize progress curves w.r.t. optimality gaps?
        plot_CIs : Boolean
            plot bootstrapping confidence intervals?
        """
        # Set up plot.
        stylize_plot(plot_type=plot_type, solver_name=self.solver.name, problem_name=self.problem.name, normalize=normalize, budget=self.problem.factors["budget"], beta=beta)
        if plot_type == "all":
            # Plot all estimated progress curves.
            if normalize:
                for mrep in range(self.n_macroreps):
                    plt.step(self.unique_frac_budgets, self.all_prog_curves[mrep], where='post')
            else:
                for mrep in range(self.n_macroreps):
                    plt.step(self.unique_budgets, self.all_est_objective[mrep], where='post')
        elif plot_type == "mean":
            # Plot estimated mean progress curve.
            if normalize:
                estimator = np.mean(self.all_prog_curves, axis=0)
                plt.step(self.unique_frac_budgets, estimator, where='post')
            else:
                estimator = np.mean(self.all_est_objective, axis=0)
                plt.step(self.unique_budgets, estimator, where='post')
        elif plot_type == "quantile":
            # Plot estimated beta-quantile progress curve.
            if normalize:
                estimator = np.quantile(self.all_prog_curves, q=beta, axis=0)
                plt.step(self.unique_frac_budgets, estimator, where='post')
            else:
                estimator = np.quantile(self.all_est_objective, q=beta, axis=0)
                plt.step(self.unique_budgets, estimator, where='post')
        else:
            print("Not a valid plot type.")
        if plot_type == "mean" or plot_type == "quantile":
            # Report bootstrapping error estimation and optionally plot bootstrap CIs.
            self.plot_bootstrap_CIs(plot_type, normalize, estimator, plot_CIs, beta)
        save_plot(solver_name=self.solver.name, problem_name=self.problem.name, plot_type=plot_type, normalize=normalize)

    def plot_solvability_curves(self, solve_tols=[0.10], plot_CIs=True):
        """
        Plot the solvability curve(s) for a single solver-problem pair.
        Optionally plot bootstrap CIs.

        Arguments
        ---------
        solve_tols : list of floats in (0,1]
            relative optimality gap(s) definining when a problem is solved
        plot_CIs : Boolean
            plot bootstrapping confidence intervals?
        """
        # Compute solve times.
        self.compute_solvability(solve_tols=solve_tols)
        for tol_index in range(len(self.solve_tols)):
            solve_tol = solve_tols[tol_index]
            stylize_solvability_plot(solver_name=self.solver.name, problem_name=self.problem.name, solve_tol=solve_tol, plot_type="single")
            # Construct matrix showing when macroreplications are solved.
            solve_matrix = np.zeros((self.n_macroreps, len(self.unique_frac_budgets)))
            # Pass over progress curves to find first solve_tol crossing time.
            for mrep in range(self.n_macroreps):
                for budget_index in range(len(self.unique_frac_budgets)):
                    if self.solve_times[tol_index][mrep] <= self.unique_frac_budgets[budget_index]:
                        solve_matrix[mrep][budget_index] = 1
            # Compute proportion of macroreplications "solved" by intermediate budget.
            estimator = np.mean(solve_matrix, axis=0)
            # Plot solvability curve.
            plt.step(self.unique_frac_budgets, estimator, where='post')
            if plot_CIs:
                # Report bootstrapping error estimation and optionally plot bootstrap CIs.
                self.plot_bootstrap_CIs(plot_type="solvability", normalize=True, estimator=estimator, plot_CIs=plot_CIs, tol_index=tol_index)
            save_plot(solver_name=self.solver.name, problem_name=self.problem.name, plot_type="cdf solve times", normalize=True, extra=solve_tol)

    def compute_area_stats(self, compute_CIs=True):
        """
        Compute average and standard deviation of areas under progress curves.
        Optionally compute bootstrap confidence intervals.

        Arguments
        ---------
        compute_CIs : Boolean
            compute bootstrap confidence invervals for average and std dev?
        """
        # Compute areas under each estimated progress curve.
        self.areas = [area_under_prog_curve(prog_curve, self.unique_frac_budgets) for prog_curve in self.all_prog_curves]
        self.area_mean = np.mean(self.areas)
        self.area_std_dev = np.std(self.areas, ddof=1)
        # (Optional) Compute bootstrap CIs.
        if compute_CIs:
            lower_bound, upper_bound, _ = self.bootstrap_CI(plot_type="area_mean", normalize=True, estimator=[self.area_mean], n_bootstraps=100, conf_level=0.95, bias_correction=True)
            self.area_mean_CI = [lower_bound[0], upper_bound[0]]
            lower_bound, upper_bound, _ = self.bootstrap_CI(plot_type="area_std_dev", normalize=True, estimator=[self.area_std_dev], n_bootstraps=100, conf_level=0.95, bias_correction=True)
            self.area_std_dev_CI = [lower_bound[0], upper_bound[0]]

    def compute_solvability(self, solve_tols=[0.10]):
        """
        Compute alpha-solve times for all macroreplications.
        Can specify multiple values of alpha.

        Arguments
        ---------
        solve_tols : list of floats in (0,1]
            relative optimality gap(s) definining when a problem is solved
        """
        self.solve_tols = solve_tols
        self.solve_times = [[solve_time_of_prog_curve(prog_curve, self.unique_frac_budgets, solve_tol) for prog_curve in self.all_prog_curves] for solve_tol in solve_tols]

    def compute_solvability_quantiles(self, beta=0.50, compute_CIs=True):
        """
        Compute beta quantile of solve times, for each solve tolerance.
        Optionally compute bootstrap confidence intervals.

        Arguments
        ---------
        beta : float in (0,1)
            quantile to compute, e.g., beta quantile
        compute_CIs : Boolean
            compute bootstrap confidence invervals for quantile?
        """
        self.solve_time_quantiles = [np.quantile(self.solve_times[tol_index], q=beta, interpolation="higher") for tol_index in range(len(self.solve_tols))]
        # The default method for np.quantile is a *linear* interpolation.
        # Linear interpolation will throw error if a breakpoint is +/- infinity.
        if compute_CIs:
            lower_bounds, upper_bounds, _ = self.bootstrap_CI(plot_type="solve_time_quantile", normalize=True, estimator=self.solve_time_quantiles, beta=beta)
            self.solve_time_quantiles_CIs = [[lower_bounds[tol_index], upper_bounds[tol_index]] for tol_index in range(len(self.solve_tols))]

    def bootstrap_sample(self, bootstrap_rng, crn_across_budget=True, crn_across_macroreps=False):
        """
        Generate a bootstrap sample of estimated progress curves (normalized and unnormalized).

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
        bootstrap_prog_curves : numpy array of arrays
            bootstrapped estimated progress curves from all macroreplications
        """
        # Initialize matrices for bootstrap estimated objective and progress curves.
        bootstrap_est_objective = np.empty((self.n_macroreps, len(self.unique_budgets)))
        bootstrap_prog_curves = np.empty((self.n_macroreps, len(self.unique_budgets)))
        # Uniformly resample M macroreplications (with replacement) from 0, 1, ..., M-1.
        # Subsubstream 0: reserved for this outer-level bootstrapping.
        mreps = bootstrap_rng.choices(range(self.n_macroreps), k=self.n_macroreps)
        # Advance RNG subsubstream to prepare for inner-level bootstrapping.
        bootstrap_rng.advance_subsubstream()
        # Subsubstream 1: reserved for bootstrapping at initial solution x0 and reference optimal solution x*.
        # Bootstrap sample postreplicates at common initial solution x0.
        # Uniformly resample L postreps (with replacement) from 0, 1, ..., L.
        postreps = bootstrap_rng.choices(range(self.n_postreps_init_opt), k=self.n_postreps_init_opt)
        # Compute the mean of the resampled postreplications.
        bs_initial_obj_val = np.mean([self.initial_soln.objectives[postrep, 0] for postrep in postreps])
        # Reset subsubstream if using CRN across budgets.
        # This means the same postreplication indices will be used for resampling at x0 and x*.
        if crn_across_budget:
            bootstrap_rng.reset_subsubstream()
        # Bootstrap sample postreplicates at reference optimal solution x*.
        # Uniformly resample L postreps (with replacement) from 0, 1, ..., L.
        postreps = bootstrap_rng.choices(range(self.n_postreps_init_opt), k=self.n_postreps_init_opt)
        # Compute the mean of the resampled postreplications.
        bs_ref_opt_obj_val = np.mean([self.ref_opt_soln.objectives[postrep, 0] for postrep in postreps])
        # Compute initial optimality gap.
        bs_initial_opt_gap = bs_initial_obj_val - bs_ref_opt_obj_val
        # Advance RNG subsubstream to prepare for inner-level bootstrapping.
        # Will now be at start of subsubstream 2.
        bootstrap_rng.advance_subsubstream()
        # Bootstrap within each bootstrapped macroreplication.
        for bs_mrep in range(self.n_macroreps):
            mrep = mreps[bs_mrep]
            # Inner-level bootstrapping over intermediate recommended solutions.
            for budget in range(len(self.unique_budgets)):
                # If solution is x0...
                if np.array_equal(self.initial_soln.objectives[0:self.n_postreps_init_opt, 0], self.all_post_replicates[mrep][budget]):
                    # ...plug in fixed bootstrapped f(x0);
                    bootstrap_est_objective[bs_mrep][budget] = bs_initial_obj_val
                # else if solution is x*...
                elif np.array_equal(self.ref_opt_soln.objectives[0:self.n_postreps_init_opt, 0], self.all_post_replicates[mrep][budget]):
                    # ...plug in fixed bootstrapped f(x*);
                    bootstrap_est_objective[bs_mrep][budget] = bs_ref_opt_obj_val
                else:  # else solution other than x0 or x*...
                    # ...uniformly resample N postreps (with replacement) from 0, 1, ..., N-1 and ...
                    postreps = bootstrap_rng.choices(range(self.n_postreps), k=self.n_postreps)
                    # ...compute the mean of the resampled postreplications.
                    bootstrap_est_objective[bs_mrep][budget] = np.mean([self.all_post_replicates[mrep][budget][postrep] for postrep in postreps])
                # Normalize the estimated objective function value.
                bootstrap_prog_curves[bs_mrep][budget] = (bootstrap_est_objective[bs_mrep][budget] - bs_ref_opt_obj_val) / bs_initial_opt_gap
                # Reset subsubstream if using CRN across budgets.
                if crn_across_budget:
                    bootstrap_rng.reset_subsubstream()
            # Advance subsubstream if not using CRN across macroreps.
            if not crn_across_macroreps:
                bootstrap_rng.advance_subsubstream()
            else:
                # Reset subsubstream if using CRN across macroreplications.
                bootstrap_rng.reset_subsubstream()
        # Advance substream of random number generator to prepare for next bootstrap sample.
        bootstrap_rng.advance_substream()
        return bootstrap_est_objective, bootstrap_prog_curves

    def bootstrap_CI(self, plot_type, normalize, estimator, n_bootstraps=100, conf_level=0.95, bias_correction=True, beta=0.50, tol_index=0):
        """
        Construct bootstrap confidence intervals and compute max half-width.

        Arguments
        ---------
        plot_type : string
            indicates which type of plot to produce
                "mean" : estimated mean progress curve
                "quantile" : estimated beta quantile progress curve
                "area_mean" : mean of area under convergence curve
                "area_std_dev" : standard deviation of area under progress curve
                "solve_time_quantile" : beta quantile of solve time
                "solvability" : estimated solvability curve
        normalize : Boolean
            normalize progress curves w.r.t. optimality gaps?
        estimator : numpy array
            estimated mean or quantile progress curve
        n_bootstraps : int > 0
            number of times to generate a bootstrap sample of estimated progress curves
        conf_level : float in (0,1)
            confidence level for confidence intervals, i.e., 1-alpha
        bias_correction : bool
            use bias-corrected bootstrap CIs (via percentile method)?
        beta : float in (0,1)
            quantile for quantile aggregate progress curve, e.g., beta quantile
        tol_index : int >= 0
            index of solve tolerance

        Returns
        -------
        bs_CI_lower_bounds : numpy array
            lower bounds of bootstrap CIs at all budgets
        bs_CI_upper_bounds : numpy array
            upper bounds of bootstrap CIs at all budgets
        max_halfwidth : float
            maximum halfwidth of all bootstrap confidence intervals constructed
        """
        # Create random number generator for bootstrap sampling.
        # Stream 1 dedicated for bootstrapping.
        bootstrap_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
        if plot_type == "mean" or plot_type == "quantile" or plot_type == "solvability":
            n_intervals = len(self.unique_budgets)
        elif plot_type == "area_mean" or plot_type == "area_std_dev":
            n_intervals = 1
        elif plot_type == "solve_time_quantile":
            n_intervals = len(self.solve_tols)
        bs_aggregate_objects = np.zeros((n_bootstraps, n_intervals))
        for bs_index in range(n_bootstraps):
            # Generate bootstrap sample of estimated progress curves.
            bootstrap_est_objective, bootstrap_prog_curves = self.bootstrap_sample(bootstrap_rng=bootstrap_rng, crn_across_budget=True, crn_across_macroreps=False)
            # Apply the functional of the bootstrap sample,
            # e.g., mean/quantile (aggregate) progress curve.
            if plot_type == "mean":
                if normalize:
                    bs_aggregate_objects[bs_index] = np.mean(bootstrap_prog_curves, axis=0)
                else:
                    bs_aggregate_objects[bs_index] = np.mean(bootstrap_est_objective, axis=0)
            elif plot_type == "quantile":
                if normalize:
                    bs_aggregate_objects[bs_index] = np.quantile(bootstrap_prog_curves, q=beta, axis=0)
                else:
                    bs_aggregate_objects[bs_index] = np.quantile(bootstrap_est_objective, q=beta, axis=0)
            elif plot_type == "area_mean":
                areas = [area_under_prog_curve(prog_curve, self.unique_frac_budgets) for prog_curve in bootstrap_prog_curves]
                bs_aggregate_objects[bs_index] = np.mean(areas)
            elif plot_type == "area_std_dev":
                areas = [area_under_prog_curve(prog_curve, self.unique_frac_budgets) for prog_curve in bootstrap_prog_curves]
                bs_aggregate_objects[bs_index] = np.std(areas, ddof=1)
            elif plot_type == "solve_time_quantile":
                solve_times = [[solve_time_of_prog_curve(prog_curve, self.unique_frac_budgets, solve_tol) for prog_curve in bootstrap_prog_curves] for solve_tol in self.solve_tols]
                bs_aggregate_objects[bs_index] = [np.quantile(solve_times[tol_index], q=beta) for tol_index in range(len(self.solve_tols))]
            elif plot_type == "solvability":
                solve_times = [solve_time_of_prog_curve(prog_curve, self.unique_frac_budgets, self.solve_tols[tol_index]) for prog_curve in bootstrap_prog_curves]
                # Construct full matrix showing when macroreplications are solved.
                solve_matrix = np.zeros((self.n_macroreps, len(self.unique_frac_budgets)))
                # Pass over progress curve to find first solve_tol crossing time.
                for mrep in range(self.n_macroreps):
                    for budget_index in range(len(self.unique_frac_budgets)):
                        if solve_times[mrep] <= self.unique_frac_budgets[budget_index]:
                            solve_matrix[mrep][budget_index] = 1
                bs_aggregate_objects[bs_index] = np.mean(solve_matrix, axis=0)

        # Compute bootstrapping confidence intervals via percentile method.
        # See Efron and Gong (1983) "A leisurely look at the bootstrap,
        #     the jackknife, and cross-validation."
        if bias_correction:
            # For biased-corrected CIs, see equation (17) on page 41.
            z0s = [norm.ppf(np.mean(bs_aggregate_objects[:, interval_id] < estimator[interval_id])) for interval_id in range(n_intervals)]
            zconflvl = norm.ppf(conf_level)
            q_lowers = [norm.cdf(2 * z0 - zconflvl) for z0 in z0s]
            q_uppers = [norm.cdf(2 * z0 + zconflvl) for z0 in z0s]
            bs_CI_lower_bounds = np.array([np.quantile(bs_aggregate_objects[:, interval_id], q=q_lowers[interval_id]) for interval_id in range(n_intervals)])
            bs_CI_upper_bounds = np.array([np.quantile(bs_aggregate_objects[:, interval_id], q=q_uppers[interval_id]) for interval_id in range(n_intervals)])
        else:
            # For uncorrected CIs, see equation (16) on page 41.
            q_lower = (1 - conf_level) / 2
            q_upper = 1 - (1 - conf_level) / 2
            bs_CI_lower_bounds = np.quantile(bs_aggregate_objects, q=q_lower, axis=0)
            bs_CI_upper_bounds = np.quantile(bs_aggregate_objects, q=q_upper, axis=0)
        max_halfwidth = np.max((bs_CI_upper_bounds - bs_CI_lower_bounds) / 2)
        return bs_CI_lower_bounds, bs_CI_upper_bounds, max_halfwidth

    def plot_bootstrap_CIs(self, plot_type, normalize, estimator, plot_CIs,
                           beta=None, tol_index=None):
        """
        Optionally plot bootstrap confidence intervals and report max
        half-width.

        Arguments
        ---------
        plot_type : string
            indicates which type of plot to produce
                "all" : all estimated progress curves
                "mean" : estimated mean progress curve
                "quantile" : estimated beta quantile progress curve
                "solvability" : estimated solvability curve
        normalize : Boolean
            normalize progress curves w.r.t. optimality gaps?
        estimator : numpy array
            estimated mean or quantile progress curve
        plot_CIs : Boolean
            plot bootstrapping confidence intervals?
        beta : float in (0,1) (optional)
            quantile for quantile aggregate progress curve, e.g., beta quantile
        tol_index : int >= 0
            index of solve tolerance
        """
        # Construct bootstrap confidence intervals.
        bs_CI_lower_bounds, bs_CI_upper_bounds, max_halfwidth = self.bootstrap_CI(plot_type=plot_type, normalize=normalize, estimator=estimator, beta=beta, tol_index=tol_index)
        if normalize:
            budgets = self.unique_frac_budgets
            xloc = 0.05
            yloc = -0.35
        else:
            budgets = self.unique_budgets
            xloc = 0.05 * self.problem.factors["budget"]
            yloc = (min(bs_CI_lower_bounds)
                    - 0.25 * (max(bs_CI_upper_bounds) - min(bs_CI_lower_bounds)))
        if plot_CIs:
            # Optionally plot bootstrap confidence intervals.
            plt.step(budgets, bs_CI_lower_bounds, 'b--', where='post')
            plt.step(budgets, bs_CI_upper_bounds, 'b--', where='post')
        # Print caption about max halfwidth of bootstrap confidence intervals.
        txt = ("The max halfwidth of the bootstrap CIs is "
               + str(round(max_halfwidth, 2)) + ".")
        plt.text(x=xloc, y=yloc, s=txt)

    def clear_runs(self):
        """
        Delete results from run() method and any downstream results.
        """
        attributes = ["n_macroreps",
                      "all_recommended_xs",
                      "all_intermediate_budgets"]
        for attribute in attributes:
            try:
                delattr(self, attribute)
            except Exception:
                pass
        self.clear_postreps()
        self.clear_stats()

    def clear_postreps(self):
        """
        Delete results from post_replicate() method and any downstream results.
        """
        attributes = ["n_postreps",
                      "n_postreps_init_opt",
                      "crn_across_budget",
                      "crn_across_macroreps",
                      "all_reevaluated_solns",
                      "all_post_replicates",
                      "all_est_objective",
                      "all_prog_curves",
                      "initial_soln",
                      "ref_opt_soln"]
        for attribute in attributes:
            try:
                delattr(self, attribute)
            except Exception:
                pass
        self.clear_stats()

    def clear_stats(self):
        """
        Delete summary statistics associated with experiment.
        """
        attributes = ["areas",
                      "area_mean",
                      "area_std_dev",
                      "area_mean_CI",
                      "area_std_dev_CI",
                      "solve_tol",
                      "solve_times",
                      "solve_time_quantile",
                      "solve_time_quantile_CI"]
        for attribute in attributes:
            try:
                delattr(self, attribute)
            except Exception:
                pass

    def record_experiment_results(self):
        """
        Save wrapper_base.Experiment object to .pickle file.
        """
        with open(self.file_name_path, "wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


def trim_solver_results(problem, recommended_solns, intermediate_budgets):
    """
    Trim solutions recommended by solver after problem's max budget.

    Arguments
    ---------
    problem : base.Problem object
        Problem object on which the solver was run
    recommended_solutions : list of base.Solution objects
        solutions recommended by the solver
    intermediate_budgets : list of ints >= 0
        intermediate budgets at which solver recommended different solutions
    """
    # Remove solutions corresponding to intermediate budgets exceeding max budget.
    invalid_idxs = [idx for idx, element in enumerate(intermediate_budgets) if element > problem.factors["budget"]]
    for invalid_idx in sorted(invalid_idxs, reverse=True):
        del recommended_solns[invalid_idx]
        del intermediate_budgets[invalid_idx]
    # If no solution is recommended at the final budget,
    # re-recommended the latest recommended solution.
    # Necessary for clean plotting of progress curves.
    if intermediate_budgets[-1] < problem.factors["budget"]:
        recommended_solns.append(recommended_solns[-1])
        intermediate_budgets.append(problem.factors["budget"])
    return recommended_solns, intermediate_budgets


def read_experiment_results(file_name_path):
    """
    Read in wrapper_base.Experiment object from .pickle file.

    Arguments
    ---------
    file_name_path : string
        path of .pickle file for reading wrapper_base.Experiment object

    Returns
    -------
    experiment : wrapper_base.Experiment object
        experiment that has been run or has been post-processed
    """
    with open(file_name_path, "rb") as file:
        experiment = pickle.load(file)
    return experiment


def stylize_plot(plot_type, solver_name, problem_name, normalize, budget=None,
                 beta=None):
    """
    Create new figure. Add labels to plot and reformat axes.

    Arguments
    ---------
    plot_type : string
        indicates which type of plot to produce
            "all" : all estimated progress curves
            "mean" : estimated mean progress curve
            "quantile" : estimated beta quantile progress curve
    solver_name : string
        name of solver
    problem_name : string
        name of problem
    normalize : Boolean
        normalize progress curves w.r.t. optimality gaps?
    budget : int
        budget of problem, measured in function evaluations
    beta : float in (0,1) (optional)
        quantile for quantile aggregate progress curve, e.g., beta quantile
    """
    plt.figure()
    # Format axes, axis labels, title, and tick marks.
    if normalize:
        xlabel = "Fraction of Budget"
        ylabel = "Fraction of Initial Optimality Gap"
        xlim = (0, 1)
        ylim = (-0.1, 1.1)
        title = solver_name + " on " + problem_name + "\n"
    elif not normalize:
        xlabel = "Budget"
        ylabel = "Objective Function Value"
        xlim = (0, budget)
        ylim = None
        title = solver_name + " on " + problem_name + "\n" + "Unnormalized "
    if plot_type == "all":
        title = title + "Estimated Progress Curves"
    elif plot_type == "mean":
        title = title + "Mean Progress Curve"
    elif plot_type == "quantile":
        title = title + str(round(beta, 2)) + "-Quantile Progress Curve"
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.title(title, size=14)
    plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tick_params(axis='both', which='major', labelsize=12)


def stylize_solvability_plot(solver_name, problem_name, solve_tol, plot_type, beta=0.5):
    """
    Create new figure. Add labels to plot and reformat axes.

    Arguments
    ---------
    solver_name : string
        name of solver
    problem_name : string
        name of problem
    solve_tol : float in (0,1]
        relative optimality gap definining when a problem is solved
    plot_type : string
        type of plot
            - "single"
            - "cdf"
            - "quantile"
    beta : float in (0,1)
        quantile to compute, e.g., beta quantile
    """
    plt.figure()
    # Format axes, axis labels, title, and tick marks.
    xlabel = "Fraction of Budget"
    xlim = (0, 1)
    ylim = (0, 1.05)
    if plot_type == "single":
        ylabel = "Fraction of Macroreplications Solved"
        title = solver_name + " on " + problem_name + "\n"
        title = title + "CDF of " + str(round(solve_tol, 2)) + "-Solve Times"
    elif plot_type == "cdf":
        ylabel = "Mean Solve Percentage"
        title = "CDF Solvability Profile for " + solver_name + "\n"
        title = title + "Profile of CDF of " + str(round(solve_tol, 2)) + "-Solve Times"
    elif plot_type == "quantile":
        ylabel = "Proportion of Problems Solved"
        title = "Quantile Solvability Profile for " + solver_name + "\n"
        title = title + "Profile of " + str(round(beta, 2)) + "-Quantiles of " + str(round(solve_tol, 2)) + "-Solve Times"
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.title(title, size=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tick_params(axis='both', which='major', labelsize=12)


def stylize_difference_plot(solve_tol):
    """
    Create new figure. Add labels to plot and reformat axes.

    Parameters
    ----------
    solve_tol : float in (0,1]
        relative optimality gap definining when a problem is solved
    """
    plt.figure()
    # Format axes, axis labels, title, and tick marks.
    xlabel = "Fraction of Budget"
    xlim = (0, 1)
    ylabel = "Difference in Fraction of Macroreplications Solved"
    title = "SOLVERSET" + " on " + "PROBLEMSET" + "\n"
    title = title + "Difference of " + str(round(solve_tol, 2)) + "-Solvability Curves"
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.title(title, size=14)
    plt.xlim(xlim)
    plt.tick_params(axis='both', which='major', labelsize=12)


def stylize_area_plot(solver_name):
    """
    Create new figure for area plots. Add labels to plot and reformat axes.

    Arguments
    ---------
    solver_name : string
        name of solver
    """
    plt.figure()
    # Format axes, axis labels, title, and tick marks.
    xlabel = "Mean Area"
    ylabel = "Std Dev of Area"
    xlim = (0, 1)
    ylim = (0, 0.5)
    title = solver_name + "\n"
    title = title + "Areas Under Progress Curves"
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.title(title, size=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tick_params(axis='both', which='major', labelsize=12)


def save_plot(solver_name, problem_name, plot_type, normalize, extra=None):
    """
    Create new figure. Add labels to plot and reformat axes.

    Arguments
    ---------
    solver_name : string
        name of solver
    problem_name : string
        name of problem
    plot_type : string
        indicates which type of plot to produce
            "all" : all estimated progress curves
            "mean" : estimated mean progress curve
            "quantile" : estimated beta quantile progress curve
            "cdf solve times" : cdf of solve times
            "cdf solvability" : cdf solvability profile
            "quantile solvability" : quantile solvability profile
            "area" : area scatterplot
            "difference" : difference profile
    normalize : Boolean
        normalize progress curves w.r.t. optimality gaps?
    extra : float (or list of floats)
        extra number(s) specifying quantile (e.g., beta) and/or solve tolerance
    """
    # Form string name for plot filename.
    if plot_type == "all":
        plot_name = "all_prog_curves"
    elif plot_type == "mean":
        plot_name = "mean_prog_curve"
    elif plot_type == "quantile":
        plot_name = "quantile_prog_curve"
    elif plot_type == "cdf solve times":
        plot_name = "cdf_" + str(extra) + "_solve_times"
    elif plot_type == "cdf solvability":
        plot_name = "profile_cdf_" + str(extra) + "_solve_times"
    elif plot_type == "quantile solvability":
        plot_name = "profile_" + str(extra[1]) + "_quantile_" + str(extra[0]) + "_solve_times"
    elif plot_type == "area":
        plot_name = "area_scatterplot"
    elif plot_type == "difference":
        plot_name = "difference_profile"
    if not normalize:
        plot_name = plot_name + "_unnorm"
    path_name = "experiments/plots/" + str(solver_name) + "_on_" + str(problem_name) + "_" + plot_name + ".png"
    plt.savefig(path_name, bbox_inches="tight")


def area_under_prog_curve(prog_curve, frac_inter_budgets):
    """
    Compute the area under a normalized estimated progress curve.

    Arguments
    ---------
    prog_curve : numpy array
        normalized estimated progress curve for a macroreplication
    frac_inter_budgets : numpy array
        fractions of budget at which the progress curve is defined

    Returns
    -------
    area : float
        area under the estimated progress curve
    """
    area = np.dot(prog_curve[:-1], np.diff(frac_inter_budgets))
    return area


def solve_time_of_prog_curve(prog_curve, frac_inter_budgets, solve_tol):
    """
    Compute the solve time of a normalized estimated progress curve.

    Arguments
    ---------
    prog_curve : numpy array
        normalized estimated progress curves for a macroreplication
    frac_inter_budgets : numpy array
        fractions of budget at which the progress curve is defined
    solve_tol : float in (0,1]
        relative optimality gap definining when a problem is solved

    Returns
    -------
    solve_time : float
        time at which the normalized progress curve first drops below
        solve_tol, i.e., the "alpha" solve time
    """
    # Alpha solve time defined as infinity if the problem is not solved
    # to within solve_tol.
    solve_time = np.inf
    # Pass over progress curve to find first solve_tol crossing time.
    for i in range(len(prog_curve)):
        if prog_curve[i] < solve_tol:
            solve_time = frac_inter_budgets[i]
            break
    return solve_time


class MetaExperiment(object):
    """
    Base class for running one or more solver on one or more problem.

    Attributes
    ----------
    solver_names : list of strings
        list of solver names
    n_solvers : int > 0
        number of solvers
    problem_names : list of strings
        list of problem names
    n_problems : int > 0
        number of problems
    all_solver_fixed_factors : dict of dict
        fixed solver factors for each solver
            outer key is solver name
            inner key is factor name
    all_problem_fixed_factors : dict of dict
        fixed problem factors for each problem
            outer key is problem name
            inner key is factor name
    all_oracle_fixed_factors : dict of dict
        fixed oracle factors for each problem
            outer key is problem name
            inner key is factor name
    experiments : list of list of Experiment objects
        all problem-solver pairs

    Arguments
    ---------
    solver_names : list of strings
        list of solver names
    problem_names : list of strings
        list of problem names
    fixed_factors_filename : string
        name of .py file containing dictionaries of fixed factors
        for solvers/problems/oracles.
    """
    def __init__(self, solver_names, problem_names, fixed_factors_filename=None):
        self.solver_names = solver_names
        self.n_solvers = len(solver_names)
        self.problem_names = problem_names
        self.n_problems = len(problem_names)
        # Read in fixed solver/problem/oracle factors from .py file in the Experiments folder.
        # File should contain three dictionaries of dictionaries called
        #   - all_solver_fixed_factors
        #   - all_problem_fixed_factors
        #   - all_oracle_fixed_factors
        fixed_factors_filename = "experiments.inputs." + fixed_factors_filename
        all_factors = importlib.import_module(fixed_factors_filename)
        self.all_solver_fixed_factors = getattr(all_factors, "all_solver_fixed_factors")
        self.all_problem_fixed_factors = getattr(all_factors, "all_problem_fixed_factors")
        self.all_oracle_fixed_factors = getattr(all_factors, "all_oracle_fixed_factors")
        # Create all problem-solver pairs (i.e., instances of Experiment class)
        self.experiments = []
        for solver_name in solver_names:
            solver_experiments = []
            for problem_name in problem_names:
                try:
                    # If a file exists, read in Experiment object.
                    with open("experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle", "rb") as file:
                        next_experiment = pickle.load(file)
                    # TO DO: Check if the solver/problem/oracle factors in the file match
                    # those for the MetaExperiment.
                except Exception:
                    # If no file exists, create new Experiment object.
                    print("No experiment file exists for " + solver_name + " on " + problem_name + ". Creating new experiment.")
                    next_experiment = Experiment(solver_name=solver_name,
                                                 problem_name=problem_name,
                                                 solver_fixed_factors=self.all_solver_fixed_factors[solver_name],
                                                 problem_fixed_factors=self.all_problem_fixed_factors[problem_name],
                                                 oracle_fixed_factors=self.all_oracle_fixed_factors[problem_name])
                    # next_experiment.record_experiment_results()
                solver_experiments.append(next_experiment)
            self.experiments.append(solver_experiments)

    def run(self, n_macroreps=10):
        """
        Run n_macroreps of each solver on each problem.

        Arguments
        ---------
        n_macroreps : int
            number of macroreplications of the solver to run on the problem
        """
        for solver_index in range(self.n_solvers):
            for problem_index in range(self.n_problems):
                experiment = self.experiments[solver_index][problem_index]
                # If the problem-solver pair has not been run in this way before,
                # run it now and save result to .pickle file.
                if (getattr(experiment, "n_macroreps", None) != n_macroreps):
                    print("Running " + experiment.solver.name + " on " + experiment.problem.name + ".")
                    experiment.clear_runs()
                    experiment.run(n_macroreps)

    def post_replicate(self, n_postreps, n_postreps_init_opt, crn_across_budget=True, crn_across_macroreps=False):
        """
        For each problem-solver pair, run postreplications at solutions
        recommended by the solver on each macroreplication.

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
        for solver_index in range(self.n_solvers):
            for problem_index in range(self.n_problems):
                experiment = self.experiments[solver_index][problem_index]
                # If the problem-solver pair has not been post-processed in this way before,
                # post-process it now.
                if (getattr(experiment, "n_postreps", None) != n_postreps
                        or getattr(experiment, "n_postreps_init_opt", None) != n_postreps_init_opt
                        or getattr(experiment, "crn_across_budget", None) != crn_across_budget
                        or getattr(experiment, "crn_across_macroreps", None) != crn_across_macroreps):
                    print("Post-processing " + experiment.solver.name + " on " + experiment.problem.name + ".")
                    experiment.clear_postreps()
                    experiment.post_replicate(n_postreps, n_postreps_init_opt, crn_across_budget, crn_across_macroreps)

    def plot_progress_curves(self, plot_type, beta=0.50, normalize=True):
        """
        Produce plots of the solvers' aggregated performances on each problem.

        Arguments
        ---------
        plot_type : string
            indicates which type of plot to produce
                "mean" : estimated mean progress curve
                "quantile" : estimated beta quantile progress curve
        beta : float in (0,1)
            quantile to plot, e.g., beta quantile
        normalize : Boolean
            normalize progress curves w.r.t. optimality gaps?
        """
        for problem_index in range(self.n_problems):
            stylize_plot(plot_type=plot_type, solver_name="SOLVERSET", problem_name=self.problem_names[problem_index], normalize=normalize, budget=self.experiments[0][problem_index].problem.factors["budget"], beta=beta)
            for solver_index in range(self.n_solvers):
                experiment = self.experiments[solver_index][problem_index]
                if plot_type == "mean":
                    # Plot estimated mean progress curve.
                    if normalize:
                        estimator = np.mean(experiment.all_prog_curves, axis=0)
                        plt.step(experiment.unique_frac_budgets, estimator, where='post')
                    else:
                        estimator = np.mean(experiment.all_est_objective, axis=0)
                        plt.step(experiment.unique_budgets, estimator, where='post')
                elif plot_type == "quantile":
                    # Plot estimated beta-quantile progress curve.
                    if normalize:
                        estimator = np.quantile(experiment.all_prog_curves, q=beta, axis=0)
                        plt.step(experiment.unique_frac_budgets, estimator, where='post')
                    else:
                        estimator = np.quantile(experiment.all_est_objective, q=beta, axis=0)
                        plt.step(experiment.unique_budgets, estimator, where='post')
                else:
                    print("Not a valid plot type.")
            plt.legend(labels=self.solver_names, loc="upper right")
            save_plot(solver_name="SOLVERSET", problem_name=self.problem_names[problem_index], plot_type=plot_type, normalize=normalize)

    def plot_solvability_curves(self, solve_tols=[0.10]):
        """
        Produce the solvability curve (cdf of the solve times) for solvers
        on each problem.

        Arguments
        ---------
        solve_tols : list of floats in (0,1]
            relative optimality gap(s) definining when a problem is solved
        """
        for problem_index in range(self.n_problems):
            # Compute solve times for each solver at each tolerance
            for solver_index in range(self.n_solvers):
                experiment = self.experiments[solver_index][problem_index]
                experiment.compute_solvability(solve_tols=solve_tols)
            # For each tolerance, plot solvability curves for each solver
            for tol_index in range(len(solve_tols)):
                solve_tol = solve_tols[tol_index]
                stylize_solvability_plot(solver_name="SOLVERSET", problem_name=self.problem_names[problem_index], solve_tol=solve_tol, plot_type="single")
                for solver_index in range(self.n_solvers):
                    experiment = self.experiments[solver_index][problem_index]
                    # Construct matrix showing when macroreplications are solved.
                    solve_matrix = np.zeros((experiment.n_macroreps, len(experiment.unique_frac_budgets)))
                    # Pass over progress curves to find first solve_tol crossing time.
                    for mrep in range(experiment.n_macroreps):
                        for budget_index in range(len(experiment.unique_frac_budgets)):
                            if experiment.solve_times[tol_index][mrep] <= experiment.unique_frac_budgets[budget_index]:
                                solve_matrix[mrep][budget_index] = 1
                    # Compute proportion of macroreplications "solved" by intermediate budget.
                    estimator = np.mean(solve_matrix, axis=0)
                    # Plot solvability curve.
                    plt.step(experiment.unique_frac_budgets, estimator, where='post')
                plt.legend(labels=self.solver_names, loc="lower right")
                save_plot(solver_name="SOLVERSET", problem_name=self.problem_names[problem_index], plot_type="cdf solve times", normalize=True, extra=solve_tol)

    def plot_area_scatterplot(self, plot_CIs=True, all_in_one=True):
        """
        Plot a scatter plot of mean and standard deviation of area under progress curves.
        Either one plot for each solver or one plot for all solvers.
        """
        # Compute areas under progress curves (and summary statistics) for each
        # problem-solver pair.
        for solver_index in range(self.n_solvers):
            for problem_index in range(self.n_problems):
                experiment = self.experiments[solver_index][problem_index]
                experiment.compute_area_stats(compute_CIs=plot_CIs)
                experiment.record_experiment_results()
        # Produce plot(s).
        if all_in_one:
            stylize_area_plot(solver_name="SOLVERSET")
        for solver_index in range(self.n_solvers):
            if not all_in_one:
                stylize_area_plot(solver_name=self.solver_names[solver_index])
            # Aggregate statistics.
            area_means = [self.experiments[solver_index][problem_index].area_mean for problem_index in range(self.n_problems)]
            area_std_devs = [self.experiments[solver_index][problem_index].area_std_dev for problem_index in range(self.n_problems)]
            if plot_CIs:
                area_means_CIs = [self.experiments[solver_index][problem_index].area_mean_CI for problem_index in range(self.n_problems)]
                area_std_devs_CIs = [self.experiments[solver_index][problem_index].area_std_dev_CI for problem_index in range(self.n_problems)]
            # Plot scatter plot.
            if plot_CIs:
                xerr = [np.array(area_means) - np.array(area_means_CIs)[:, 0], np.array(area_means_CIs)[:, 1] - np.array(area_means)]
                yerr = [np.array(area_std_devs) - np.array(area_std_devs_CIs)[:, 0], np.array(area_std_devs_CIs)[:, 1] - np.array(area_std_devs)]
                plt.errorbar(x=area_means,
                             y=area_std_devs,
                             xerr=xerr,
                             yerr=yerr
                             )
            else:
                plt.scatter(x=area_means, y=area_std_devs)
            if not all_in_one:
                save_plot(solver_name=self.solver_names[solver_index], problem_name="PROBLEMSET", plot_type="area", normalize=True)
        if all_in_one:
            plt.legend(labels=self.solver_names, loc="upper right")
            save_plot(solver_name="SOLVERSET", problem_name="PROBLEMSET", plot_type="area", normalize=True)

    def plot_solvability_profiles(self, solve_tol=0.1, beta=0.5, ref_solver=None):
        """
        Plot the solvability profiles for each solver on a set of problems.
        Three types of plots:
            1) cdf solvability profile
            2) quantile solvability profile
            3) difference solvability profile

        Arguments
        ---------
        solve_tol : float in (0,1]
            relative optimality gap definining when a problem is solved
        beta : float in (0,1)
            quantile to compute, e.g., beta quantile
        ref_solver : str
            name of solver used as benchmark for difference profiles
        """
        all_solver_unique_frac_budgets = []
        all_solvability_profiles = []
        stylize_solvability_plot(solver_name="SOLVERSET", problem_name="PROBLEMSET", solve_tol=solve_tol, beta=None, plot_type="cdf")
        for solver_index in range(self.n_solvers):
            solvability_curves = []
            all_budgets = []
            for problem_index in range(self.n_problems):
                experiment = self.experiments[solver_index][problem_index]
                # Compute solve times.
                experiment.compute_solvability(solve_tols=[solve_tol])
                experiment.compute_solvability_quantiles(beta=beta, compute_CIs=False)
                # Construct matrix showing when macroreplications are solved.
                solve_matrix = np.zeros((experiment.n_macroreps, len(experiment.unique_frac_budgets)))
                # Pass over progress curves to find first solve_tol crossing time.
                for mrep in range(experiment.n_macroreps):
                    for budget_index in range(len(experiment.unique_frac_budgets)):
                        # TO DO: HARD-CODED for tol_index=0
                        if experiment.solve_times[0][mrep] <= experiment.unique_frac_budgets[budget_index]:
                            solve_matrix[mrep][budget_index] = 1
                solvability_curves.append(list(np.mean(solve_matrix, axis=0)))
                all_budgets.append(list(experiment.unique_frac_budgets))
            # Compute the solver's solvability profile.
            solver_unique_frac_budgets = np.unique([budget for budgets in all_budgets for budget in budgets])
            all_solve_matrix = np.zeros((self.n_problems, len(solver_unique_frac_budgets)))
            for problem_index in range(self.n_problems):
                for budget_index in range(len(solver_unique_frac_budgets)):
                    problem_budget_index = np.max(np.where(np.array(all_budgets[problem_index]) <= solver_unique_frac_budgets[budget_index]))
                    all_solve_matrix[problem_index][budget_index] = solvability_curves[problem_index][problem_budget_index]
            solvability_profile = np.mean(all_solve_matrix, axis=0)
            # Plot the solver's solvability profile.
            plt.step(solver_unique_frac_budgets, solvability_profile, where='post')
            # Append results.
            all_solver_unique_frac_budgets.append(solver_unique_frac_budgets)
            all_solvability_profiles.append(solvability_profile)
        plt.legend(labels=self.solver_names, loc="lower right")
        # TO DO: Change the y-axis label produced by this helper function.
        save_plot(solver_name="SOLVERSET", problem_name="PROBLEMSET", plot_type="cdf solvability", normalize=True, extra=solve_tol)
        # Plot solvability profiles for each solver.
        stylize_solvability_plot(solver_name="SOLVERSET", problem_name="PROBLEMSET", solve_tol=solve_tol, beta=beta, plot_type="quantile")
        for solver_index in range(self.n_solvers):
            solvability_quantiles = []
            for problem_index in range(self.n_problems):
                experiment = self.experiments[solver_index][problem_index]
                # TO DO: Hard-coded for first tol_index
                solvability_quantiles.append(experiment.solve_time_quantiles[0])
            plt.step(np.sort(solvability_quantiles + [0, 1]), np.append(np.linspace(start=0, stop=1, num=self.n_problems + 1), [1]), where='post')
        save_plot(solver_name="SOLVERSET", problem_name="PROBLEMSET", plot_type="quantile solvability", normalize=True, extra=[solve_tol, beta])
        # Plot difference solvability profiles. (Optional)
        if ref_solver is not None:
            stylize_difference_plot(solve_tol=solve_tol)
            non_ref_solvers = [solver_name for solver_name in self.solver_names if solver_name != ref_solver]
            ref_solver_index = self.solver_names.index(ref_solver)
            for solver_index in range(self.n_solvers):
                solver_name = self.solver_names[solver_index]
                if solver_name is not ref_solver:
                    diff_budgets, diff_solvability_profile = compute_difference_solvability_profile(budgets_1=all_solver_unique_frac_budgets[solver_index],
                                                                                                    solv_profile_1=all_solvability_profiles[solver_index],
                                                                                                    budgets_2=all_solver_unique_frac_budgets[ref_solver_index],
                                                                                                    solv_profile_2=all_solvability_profiles[ref_solver_index]
                                                                                                    )
                    plt.step(diff_budgets, diff_solvability_profile, where='post')
            plt.plot([0, 1], [0, 0], color='black', linestyle='dashed')
            plt.legend(labels=[non_ref_solver + " - " + ref_solver for non_ref_solver in non_ref_solvers], loc="upper right")
            save_plot(solver_name="SOLVERSET", problem_name="PROBLEMSET", plot_type="difference", normalize=True)


def compute_difference_solvability_profile(budgets_1, solv_profile_1, budgets_2, solv_profile_2):
    """
    Calculate the difference of two solvability profiles (Solver 1 - Solver 2).

    Parameters
    ----------
    budgets_1 : list of floats
        list of intermediate budgets for Solver 1
    solv_profile_1 : list of floats
        solvability profile of Solver 1
    budgets_2 : list of floats
        list of intermediate budgets for Solver 2
    solv_profile_2 : list of floats
        solvability profile of Solver 2
    """
    diff_budgets = np.unique(list(budgets_1) + list(budgets_2))
    diff_solvability_profile = []
    for budget in diff_budgets:
        solv_profile_1_index = np.max(np.where(budgets_1 <= budget))
        solv_profile_2_index = np.max(np.where(budgets_2 <= budget))
        diff_solvability_profile.append(solv_profile_1[solv_profile_1_index] - solv_profile_2[solv_profile_2_index])
    return(diff_budgets, diff_solvability_profile)
