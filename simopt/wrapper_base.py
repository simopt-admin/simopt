#!/usr/bin/env python
"""
Summary
-------
Provide base classes for experiments and meta experiments.
Plus helper functions for reading/writing data and plotting.
Listing
-------
Curve : class
mean_of_curves : function
quantile_of_curves : function
cdf_of_curves_crossing_times : function
quantile_cross_jump : function
difference_of_curves : function
max_difference_of_curves : function
Experiment : class
trim_solver_results : function
read_experiment_results : function
post_normalize : function
bootstrap_sample_all : function
bootstrap_procedure : function
functional_of_curves : function
compute_bootstrap_CI : function
plot_bootstrap_CIs : function
report_max_halfwidth : function
plot_progress_curves : function
plot_solvability_cdfs : function
plot_area_scatterplots : function
plot_solvability_profiles : function
setup_plot : function
save_plot : function
MetaExperiment : class
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


class Curve(object):
    """
    Base class for all curves.
    Attributes
    ----------
    x_vals : list of floats
        values of horizontal components
    y_vals : list of floats
        values of vertical components
    n_points : int
        number of values in x- and y- vectors
    Parameters
    ----------
    x_vals : list of floats
        values of horizontal components
    y_vals : list of floats
        values of vertical components
    """
    def __init__(self, x_vals, y_vals):
        if len(x_vals) != len(y_vals):
            print("Vectors of x- and y- values must be of same length.")
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.n_points = len(x_vals)

    def lookup(self, x):
        """
        Lookup the y-value of the curve at an intermediate x-value.
        Parameters
        ----------
        x : float
            x-value at which to lookup the y-value
        Returns
        -------
        y : float
            y-value corresponding to x
        """
        if x < self.x_vals[0]:
            y = np.nan
        else:
            idx = np.max(np.where(np.array(self.x_vals) <= x))
            y = self.y_vals[idx]
        return y

    def compute_crossing_time(self, threshold):
        """
        Compute the first time at which a curve drops below a given threshold.
        Parameters
        ----------
        threshold : float
            value for which to find first crossing time
        Returns
        -------
        crossing_time : float
            first time at which a curve drops below threshold
        """
        # Crossing time is defined as infinity if the curve does not drop
        # below threshold.
        crossing_time = np.inf
        # Pass over curve to find first crossing time.
        for i in range(self.n_points):
            if self.y_vals[i] < threshold:
                crossing_time = self.x_vals[i]
                break
        return crossing_time

    def compute_area_under_curve(self):
        """
        Compute the area under a curve.
        Returns
        -------
        area : float
            area under the curve
        """
        area = np.dot(self.y_vals[:-1], np.diff(self.x_vals))
        return area

    def curve_to_mesh(self, mesh):
        """
        Create a curve defined at equally spaced x values.
        Parameters
        ----------
        mesh : list of floats
            list of uniformly spaced x values
        Returns
        -------
        mesh_curve : wrapper_base.Curve object
            curve with equally spaced x values
        """
        mesh_curve = Curve(x_vals=mesh, y_vals=[self.lookup(x) for x in mesh])
        return mesh_curve

    def curve_to_full_curve(self):
        """
        Create a curve with duplicate x- and y-values to indicate steps.
        Returns
        -------
        full_curve : wrapper_base.Curve object
            curve with duplicate x- and y-values
        """
        duplicate_x_vals = [x for x in self.x_vals for _ in (0, 1)]
        duplicate_y_vals = [y for y in self.y_vals for _ in (0, 1)]
        full_curve = Curve(x_vals=duplicate_x_vals[1:], y_vals=duplicate_y_vals[:-1])
        return full_curve

    def plot(self, color_str="C0", curve_type="regular"):
        """
        Plot a curve.
        Parameters
        ----------
        color_str : str
            string indicating line color, e.g., "C0", "C1", etc.
        Returns
        -------
        handle : list of matplotlib.lines.Line2D objects
            curve handle, to use when creating legends
        """
        if curve_type == "regular":
            linestyle = "-"
            linewidth = 2
        elif curve_type == "conf_bound":
            linestyle = "--"
            linewidth = 1
        handle, = plt.step(self.x_vals,
                           self.y_vals,
                           color=color_str,
                           linestyle=linestyle,
                           linewidth=linewidth,
                           where="post"
                           )
        return handle


def mean_of_curves(curves):
    """
    Compute pointwise (w.r.t. x values) mean of curves.
    Starting and ending x values must coincide for all curves.
    Parameters
    ----------
    curves : list of wrapper_base.Curve objects
        collection of curves to aggregate
    Returns
    -------
    mean_curve : wrapper_base.Curve object
        mean curve
    """
    unique_x_vals = np.unique([x_val for curve in curves for x_val in curve.x_vals])
    mean_y_vals = [np.mean([curve.lookup(x_val) for curve in curves]) for x_val in unique_x_vals]
    mean_curve = Curve(x_vals=unique_x_vals.tolist(), y_vals=mean_y_vals)
    return mean_curve


def quantile_of_curves(curves, beta):
    """
    Compute pointwise (w.r.t. x values) quantile of curves.
    Starting and ending x values must coincide for all curves.
    Parameters
    ----------
    curves : list of wrapper_base.Curve objects
        collection of curves to aggregate
    beta : float
        quantile level
    Returns
    -------
    quantile_curve : wrapper_base.Curve object
        quantile curve
    """
    unique_x_vals = np.unique([x_val for curve in curves for x_val in curve.x_vals])
    quantile_y_vals = [np.quantile([curve.lookup(x_val) for curve in curves], q=beta) for x_val in unique_x_vals]
    quantile_curve = Curve(x_vals=unique_x_vals.tolist(), y_vals=quantile_y_vals)
    return quantile_curve


def cdf_of_curves_crossing_times(curves, threshold):
    """
    Compute the cdf of crossing times of curves.
    Parameters
    ----------
    curves : list of wrapper_base.Curve objects
        collection of curves to aggregate
    threshold : float
        value for which to find first crossing time
    Returns
    -------
    cdf_curve : wrapper_base.Curve object
        cdf of crossing times
    """
    n_curves = len(curves)
    crossing_times = [curve.compute_crossing_time(threshold) for curve in curves]
    unique_x_vals = [0] + list(np.unique([crossing_time for crossing_time in crossing_times if crossing_time < np.inf])) + [1]
    cdf_y_vals = [sum(crossing_time <= x_val for crossing_time in crossing_times) / n_curves for x_val in unique_x_vals]
    cdf_curve = Curve(x_vals=unique_x_vals, y_vals=cdf_y_vals)
    return cdf_curve


def quantile_cross_jump(curves, threshold, beta):
    """
    Compute a simple curve with a jump at the quantile of the crossing times.
    Parameters
    ----------
    curves : list of wrapper_base.Curve objects
        collection of curves to aggregate
    threshold : float
        value for which to find first crossing time
    beta : float
        quantile level
    Returns
    -------
    jump_curve : wrapper_base.Curve object
        piecewise-constant curve with a jump at the quantile crossing time (if finite)
    """
    solve_time_quantile = np.quantile([curve.compute_crossing_time(threshold=threshold) for curve in curves], q=beta)
    # Note: np.quantile will evaluate to np.nan if forced to interpolate
    # between a finite and infinite value. These are rare cases. Since
    # crossing times must be non-negative, the quantile should be mapped
    # to positive infinity.
    if solve_time_quantile == np.inf or np.isnan(solve_time_quantile):
        jump_curve = Curve(x_vals=[0, 1], y_vals=[0, 0])
    else:
        jump_curve = Curve(x_vals=[0, solve_time_quantile, 1], y_vals=[0, 1, 1])
    return jump_curve


def difference_of_curves(curve1, curve2):
    """
    Compute the difference of two curves (Curve 1 - Curve 2).
    Parameters
    ----------
    curve1, curve2 : wrapper_base.Curve objects
        curves to take the difference of
    Returns
    -------
    difference_curve : wrapper_base.Curve object
        difference of curves
    """
    unique_x_vals = np.unique(curve1.x_vals + curve2.x_vals)
    difference_y_vals = [(curve1.lookup(x_val) - curve2.lookup(x_val)) for x_val in unique_x_vals]
    difference_curve = Curve(x_vals=unique_x_vals.tolist(), y_vals=difference_y_vals)
    return difference_curve


def max_difference_of_curves(curve1, curve2):
    """
    Compute the maximum difference of two curves (Curve 1 - Curve 2)
    Parameters
    ----------
    curve1, curve2 : wrapper_base.Curve objects
        curves to take the difference of
    Returns
    -------
    max_diff : float
        maximum difference of curves
    """
    difference_curve = difference_of_curves(curve1, curve2)
    max_diff = max(difference_curve.y_vals)
    return max_diff


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
    crn_across_budget : bool
        use CRN for post-replications at solutions recommended at different times?
    crn_across_macroreps : bool
        use CRN for post-replications at solutions recommended on different macroreplications?
    all_post_replicates : list of lists of lists
        all post-replicates from all solutions from all macroreplications
    all_est_objectives : numpy array of arrays
        estimated objective values of all solutions from all macroreplications
    n_postreps_init_opt : int
        number of postreplications to take at initial solution (x0) and
        optimal solution (x*)
    crn_across_init_opt : bool
        use CRN for post-replications at solutions x0 and x*?
    x0 : tuple
        initial solution (x0)
    x0_postreps : list
        post-replicates at x0
    xstar : tuple
        proxy for optimal solution (x*)
    xstar_postreps : list
        post-replicates at x*
    objective_curves : list of wrapper_base.Curve objects
        curves of estimated objective function values,
        one for each macroreplication
    progress_curves : list of wrapper_base.Curve objects
        progress curves, one for each macroreplication
    Arguments
    ---------
    solver_name : str
        name of solver
    problem_name : str
        name of problem
    solver_rename : str
        user-specified name for solver
    problem_rename : str
        user-specified name for problem
    solver_fixed_factors : dict
        dictionary of user-specified solver factors
    problem_fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed_factors : dict
        dictionary of user-specified model factors
    file_name_path : str
        path of .pickle file for saving wrapper_base.Experiment object
    """
    def __init__(self, solver_name, problem_name, solver_rename=None, problem_rename=None, solver_fixed_factors={}, problem_fixed_factors={}, model_fixed_factors={}, file_name_path=None):
        if solver_rename is None:
            self.solver = solver_directory[solver_name](fixed_factors=solver_fixed_factors)
        else:
            self.solver = solver_directory[solver_name](name=solver_rename, fixed_factors=solver_fixed_factors)
        if problem_rename is None:
            self.problem = problem_directory[problem_name](fixed_factors=problem_fixed_factors, model_fixed_factors=model_fixed_factors)
        else:
            self.problem = problem_directory[problem_name](name=problem_rename, fixed_factors=problem_fixed_factors, model_fixed_factors=model_fixed_factors)
        if file_name_path is None:
            self.file_name_path = f"./experiments/outputs/{self.solver.name}_on_{self.problem.name}.pickle"
        else:
            self.file_name_path = file_name_path

    def check_compatibility(self):
        """
        Check whether the experiment's solver and problem are compatible.
        Returns
        -------
        error_str : str
            error message in the event problem and solver are incompatible
        """
        error_str = ""
        # Check number of objectives.
        if self.solver.objective_type == "single" and self.problem.n_objectives > 1:
            error_str += "Solver cannot solve a multi-objective problem.\n"
        elif self.solver.objective_type == "multi" and self.problem.n_objectives == 1:
            error_str += "Multi-objective solver being run on a single-objective problem.\n"
        # Check constraint types.
        constraint_types = ["unconstrained", "box", "deterministic", "stochastic"]
        if constraint_types.index(self.solver.constraint_type) < constraint_types.index(self.problem.constraint_type):
            error_str += "Solver can handle upto " + self.solver.constraint_type + " constraints, but problem has " + self.problem.constraint_type + " constraints.\n"
        # Check variable types.
        if self.solver.variable_type == "discrete" and self.problem.variable_type != "discrete":
            error_str += "Solver is for discrete variables but problem variables are " + self.problem.variable_type + ".\n"
        elif self.solver.variable_type == "continuous" and self.problem.variable_type != "continuous":
            error_str += "Solver is for continuous variables but problem variables are " + self.problem.variable_type + ".\n"
        # Check for existence of gradient estimates.
        if self.solver.gradient_needed and not self.problem.gradient_available:
            error_str += "Gradient-based solver does not have access to gradient for this problem.\n"
        return error_str

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
        rng3 = MRG32k3a(s_ss_sss_index=[2, 3, 0])  # unused
        self.solver.attach_rngs([rng1, rng2, rng3])
        # Run n_macroreps of the solver on the problem.
        # Report recommended solutions and corresponding intermediate budgets.
        for mrep in range(self.n_macroreps):
            print(f"Running macroreplication {mrep + 1} of {self.n_macroreps} of Solver {self.solver.name} on Problem {self.problem.name}.")
            # Create, initialize, and attach RNGs used for simulating solutions.
            progenitor_rngs = [MRG32k3a(s_ss_sss_index=[mrep + 2, ss, 0]) for ss in range(self.problem.model.n_rngs)]
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

    def check_run(self):
        """
        Check if the experiment has been run.

        Returns
        -------
        ran : bool
            has the experiment been run?
        """
        if getattr(self, "all_recommended_xs", None) is None:
            ran = False
        else:
            ran = True
        return ran

    def post_replicate(self, n_postreps, crn_across_budget=True, crn_across_macroreps=False):
        """
        Run postreplications at solutions recommended by the solver.
        Arguments
        ---------
        n_postreps : int
            number of postreplications to take at each recommended solution
        crn_across_budget : bool
            use CRN for post-replications at solutions recommended at different times?
        crn_across_macroreps : bool
            use CRN for post-replications at solutions recommended on different macroreplications?
        """
        self.n_postreps = n_postreps
        self.crn_across_budget = crn_across_budget
        self.crn_across_macroreps = crn_across_macroreps
        # Create, initialize, and attach RNGs for model.
        # Stream 0: reserved for post-replications.
        # Skip over first set of substreams dedicated for sampling x0 and x*.
        baseline_rngs = [MRG32k3a(s_ss_sss_index=[0, self.problem.model.n_rngs + rng_index, 0]) for rng_index in range(self.problem.model.n_rngs)]
        # Initialize matrix containing
        #     all postreplicates of objective,
        #     for each macroreplication,
        #     for each budget.
        self.all_post_replicates = [[[] for _ in range(len(self.all_intermediate_budgets[mrep]))] for mrep in range(self.n_macroreps)]
        # Simulate intermediate recommended solutions.
        for mrep in range(self.n_macroreps):
            for budget_index in range(len(self.all_intermediate_budgets[mrep])):
                x = self.all_recommended_xs[mrep][budget_index]
                fresh_soln = Solution(x, self.problem)
                fresh_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
                self.problem.simulate(solution=fresh_soln, m=self.n_postreps)
                # Store results
                self.all_post_replicates[mrep][budget_index] = list(fresh_soln.objectives[:fresh_soln.n_reps][:, 0])  # 0 <- assuming only one objective
                if crn_across_budget:
                    # Reset each rng to start of its current substream.
                    for rng in baseline_rngs:
                        rng.reset_substream()
            if crn_across_macroreps:
                # Reset each rng to start of its current substream.
                for rng in baseline_rngs:
                    rng.reset_substream()
            else:
                # Advance each rng to start of
                #     substream = current substream + # of model RNGs.
                for rng in baseline_rngs:
                    for _ in range(self.problem.model.n_rngs):
                        rng.advance_substream()
        # Store estimated objective for each macrorep for each budget.
        self.all_est_objectives = [[np.mean(self.all_post_replicates[mrep][budget_index]) for budget_index in range(len(self.all_intermediate_budgets[mrep]))] for mrep in range(self.n_macroreps)]
        # Save Experiment object to .pickle file.
        self.record_experiment_results()

    def check_postreplicate(self):
        """
        Check if the experiment has been postreplicated.

        Returns
        -------
        postreplicated : bool
            has the experiment been postreplicated?
        """
        if getattr(self, "all_est_objectives", None) is None:
            postreplicated = False
        else:
            postreplicated = True
        return postreplicated

    def bootstrap_sample(self, bootstrap_rng, normalize=True):
        """
        Generate a bootstrap sample of estimated objective curves or estimated
        progress curves.
        Parameters
        ----------
        bootstrap_rng : MRG32k3a object
            random number generator to use for bootstrapping
        normalize : Boolean
            normalize progress curves w.r.t. optimality gaps?
        Returns
        -------
        bootstrap_curves : list of wrapper_base.Curve objects
            bootstrapped estimated objective curves or estimated progress
            curves of all solutions from all bootstrapped macroreplications
        """
        bootstrap_curves = []
        # Uniformly resample M macroreplications (with replacement) from 0, 1, ..., M-1.
        # Subsubstream 0: reserved for this outer-level bootstrapping.
        bs_mrep_idxs = bootstrap_rng.choices(range(self.n_macroreps), k=self.n_macroreps)
        # Advance RNG subsubstream to prepare for inner-level bootstrapping.
        bootstrap_rng.advance_subsubstream()
        # Subsubstream 1: reserved for bootstrapping at x0 and x*.
        # Bootstrap sample post-replicates at common x0.
        # Uniformly resample L postreps (with replacement) from 0, 1, ..., L-1.
        bs_postrep_idxs = bootstrap_rng.choices(range(self.n_postreps_init_opt), k=self.n_postreps_init_opt)
        # Compute the mean of the resampled postreplications.
        bs_initial_obj_val = np.mean([self.x0_postreps[postrep] for postrep in bs_postrep_idxs])
        # Reset subsubstream if using CRN across budgets.
        # This means the same postreplication indices will be used for resampling at x0 and x*.
        if self.crn_across_init_opt:
            bootstrap_rng.reset_subsubstream()
        # Bootstrap sample postreplicates at reference optimal solution x*.
        # Uniformly resample L postreps (with replacement) from 0, 1, ..., L.
        bs_postrep_idxs = bootstrap_rng.choices(range(self.n_postreps_init_opt), k=self.n_postreps_init_opt)
        # Compute the mean of the resampled postreplications.
        bs_optimal_obj_val = np.mean([self.xstar_postreps[postrep] for postrep in bs_postrep_idxs])
        # Compute initial optimality gap.
        bs_initial_opt_gap = bs_initial_obj_val - bs_optimal_obj_val
        # Advance RNG subsubstream to prepare for inner-level bootstrapping.
        # Will now be at start of subsubstream 2.
        bootstrap_rng.advance_subsubstream()
        # Bootstrap within each bootstrapped macroreplication.
        # Option 1: Simpler (default) CRN scheme, which makes for faster code.
        if self.crn_across_budget and not self.crn_across_macroreps:
            for idx in range(self.n_macroreps):
                mrep = bs_mrep_idxs[idx]
                # Inner-level bootstrapping over intermediate recommended solutions.
                est_objectives = []
                # Same postreplication indices for all intermediate budgets on
                # a given macroreplciation.
                bs_postrep_idxs = bootstrap_rng.choices(range(self.n_postreps), k=self.n_postreps)
                for budget in range(len(self.all_intermediate_budgets[mrep])):
                    # If solution is x0...
                    if self.all_recommended_xs[mrep][budget] == self.x0:
                        est_objectives.append(bs_initial_obj_val)
                    # ...else if solution is x*...
                    elif self.all_recommended_xs[mrep][budget] == self.xstar:
                        est_objectives.append(bs_optimal_obj_val)
                    # ... else solution other than x0 or x*.
                    else:
                        # Compute the mean of the resampled postreplications.
                        est_objectives.append(np.mean([self.all_post_replicates[mrep][budget][postrep] for postrep in bs_postrep_idxs]))
                # Record objective or progress curve.
                if normalize:
                    frac_intermediate_budgets = [budget / self.problem.factors["budget"] for budget in self.all_intermediate_budgets[mrep]]
                    norm_est_objectives = [(est_objective - bs_optimal_obj_val) / bs_initial_opt_gap for est_objective in est_objectives]
                    new_progress_curve = Curve(x_vals=frac_intermediate_budgets, y_vals=norm_est_objectives)
                    bootstrap_curves.append(new_progress_curve)
                else:
                    new_objective_curve = Curve(x_vals=self.all_intermediate_budgets[mrep], y_vals=est_objectives)
                    bootstrap_curves.append(new_objective_curve)
        # Option 2: Non-default CRN behavior.
        else:
            for idx in range(self.n_macroreps):
                mrep = bs_mrep_idxs[idx]
                # Inner-level bootstrapping over intermediate recommended solutions.
                est_objectives = []
                for budget in range(len(self.all_intermediate_budgets[mrep])):
                    # If solution is x0...
                    if self.all_recommended_xs[mrep][budget] == self.x0:
                        est_objectives.append(bs_initial_obj_val)
                    # ...else if solution is x*...
                    elif self.all_recommended_xs[mrep][budget] == self.xstar:
                        est_objectives.append(bs_optimal_obj_val)
                    # ... else solution other than x0 or x*.
                    else:
                        # Uniformly resample N postreps (with replacement) from 0, 1, ..., N-1.
                        bs_postrep_idxs = bootstrap_rng.choices(range(self.n_postreps), k=self.n_postreps)
                        # Compute the mean of the resampled postreplications.
                        est_objectives.append(np.mean([self.all_post_replicates[mrep][budget][postrep] for postrep in bs_postrep_idxs]))
                        # Reset subsubstream if using CRN across budgets.
                        if self.crn_across_budget:
                            bootstrap_rng.reset_subsubstream()
                # If using CRN across macroreplications...
                if self.crn_across_macroreps:
                    # ...reset subsubstreams...
                    bootstrap_rng.reset_subsubstream()
                # ...else if not using CRN across macrorep...
                else:
                    # ...advance subsubstream.
                    bootstrap_rng.advance_subsubstream()
                # Record objective or progress curve.
                if normalize:
                    frac_intermediate_budgets = [budget / self.problem.factors["budget"] for budget in self.all_intermediate_budgets[mrep]]
                    norm_est_objectives = [(est_objective - bs_optimal_obj_val) / bs_initial_opt_gap for est_objective in est_objectives]
                    new_progress_curve = Curve(x_vals=frac_intermediate_budgets, y_vals=norm_est_objectives)
                    bootstrap_curves.append(new_progress_curve)
                else:
                    new_objective_curve = Curve(x_vals=self.all_intermediate_budgets[mrep], y_vals=est_objectives)
                    bootstrap_curves.append(new_objective_curve)
        return bootstrap_curves

    def clear_run(self):
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
        self.clear_postreplicate()

    def clear_postreplicate(self):
        """
        Delete results from post_replicate() method and any downstream results.
        """
        attributes = ["n_postreps",
                      "crn_across_budget",
                      "crn_across_macroreps",
                      "all_post_replicates",
                      "all_est_objectives"]
        for attribute in attributes:
            try:
                delattr(self, attribute)
            except Exception:
                pass
        self.clear_postnorm()

    def clear_postnorm(self):
        """
        Delete results from post_normalize() associated with experiment.
        """
        attributes = ["n_postreps_init_opt",
                      "crn_across_init_opt",
                      "x0",
                      "x0_postreps",
                      "xstar",
                      "xstar_postreps",
                      "objective_curves",
                      "progress_curves"
                      ]
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
    # re-recommend the latest recommended solution.
    # (Necessary for clean plotting of progress curves.)
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


def post_normalize(experiments, n_postreps_init_opt, crn_across_init_opt=True, proxy_init_val=None, proxy_opt_val=None, proxy_opt_x=None):
    """
    Construct objective curves and (normalized) progress curves
    for a collection of experiments on a given problem.
    Parameters
    ----------
    experiments : list of wrapper_base.Experiment objects
        experiments of different solvers on a common problem
    n_postreps_init_opt : int
        number of postreplications to take at initial x0 and optimal x*
    crn_across_init_opt : bool
        use CRN for post-replications at solutions x0 and x*?
    proxy_init_val : float
        known objective function value of initial solution
    proxy_opt_val : float
        proxy for or bound on optimal objective function value
    proxy_opt_x : tuple
        proxy for optimal solution
    """
    # Check that all experiments have the same problem and same
    # post-experimental setup.
    ref_experiment = experiments[0]
    for experiment in experiments:
        # Check if problems are the same.
        if experiment.problem != ref_experiment.problem:
            print("At least two experiments have different problem instances.")
        # Check if experiments have common number of macroreps.
        if experiment.n_macroreps != ref_experiment.n_macroreps:
            print("At least two experiments have different numbers of macro-replications.")
        # Check if experiment has been post-replicated and with common number of postreps.
        if getattr(experiment, "n_postreps", None) is None:
            print(f"The experiment of {experiment.solver_name} on {experiment.problem_name} has not been post-replicated.")
        elif getattr(experiment, "n_postreps", None) != getattr(ref_experiment, "n_postreps", None):
            print("At least two experiments have different numbers of post-replications.")
            print("Estimation of optimal solution x* may be based on different numbers of post-replications.")
    # Take post-replications at common x0.
    # Create, initialize, and attach RNGs for model.
        # Stream 0: reserved for post-replications.
    baseline_rngs = [MRG32k3a(s_ss_sss_index=[0, rng_index, 0]) for rng_index in range(experiment.problem.model.n_rngs)]
    x0 = ref_experiment.problem.factors["initial_solution"]
    if proxy_init_val is not None:
        x0_postreps = [proxy_init_val] * n_postreps_init_opt
    else:
        initial_soln = Solution(x0, ref_experiment.problem)
        initial_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
        ref_experiment.problem.simulate(solution=initial_soln, m=n_postreps_init_opt)
        x0_postreps = list(initial_soln.objectives[:n_postreps_init_opt][:, 0])  # 0 <- assuming only one objective
    if crn_across_init_opt:
        # Reset each rng to start of its current substream.
        for rng in baseline_rngs:
            rng.reset_substream()
    # Determine (proxy for) optimal solution and/or (proxy for) its
    # objective function value. If deterministic (proxy for) f(x*),
    # create duplicate post-replicates to facilitate later bootstrapping.
    # If proxy for f(x*) is specified...
    if proxy_opt_val is not None:
        xstar = None
        xstar_postreps = [proxy_opt_val] * n_postreps_init_opt
    # ...else if proxy for x* is specified...
    elif proxy_opt_x is not None:
        xstar = proxy_opt_x
        # Take post-replications at xstar.
        opt_soln = Solution(xstar, ref_experiment.problem)
        opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
        ref_experiment.problem.simulate(solution=opt_soln, m=n_postreps_init_opt)
        xstar_postreps = list(opt_soln.objectives[:n_postreps_init_opt][:, 0])  # 0 <- assuming only one objective
    # ...else if f(x*) is known...
    elif ref_experiment.problem.optimal_value is not None:
        xstar = None
        xstar_postreps = [ref_experiment.problem.optimal_value] * n_postreps_init_opt
    # ...else if x* is known...
    elif ref_experiment.problem.optimal_solution is not None:
        xstar = ref_experiment.problem.optimal_solution
        # Take post-replications at xstar.
        opt_soln = Solution(xstar, ref_experiment.problem)
        opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
        ref_experiment.problem.simulate(solution=opt_soln, m=n_postreps_init_opt)
        xstar_postreps = list(opt_soln.objectives[:n_postreps_init_opt][:, 0])  # 0 <- assuming only one objective
    # ...else determine x* empirically as estimated best solution
    # found by any solver on any macroreplication.
    else:
        # TO DO: Simplify this block of code.
        best_est_objectives = np.zeros(len(experiments))
        for experiment_idx in range(len(experiments)):
            experiment = experiments[experiment_idx]
            exp_best_est_objectives = np.zeros(experiment.n_macroreps)
            for mrep in range(experiment.n_macroreps):
                exp_best_est_objectives[mrep] = np.max(experiment.problem.minmax[0] * np.array(experiment.all_est_objectives[mrep]))
            best_est_objectives[experiment_idx] = np.max(exp_best_est_objectives)
        best_experiment_idx = np.argmax(best_est_objectives)
        best_experiment = experiments[best_experiment_idx]
        best_exp_best_est_objectives = np.zeros(experiment.n_macroreps)
        for mrep in range(best_experiment.n_macroreps):
            best_exp_best_est_objectives[mrep] = np.max(best_experiment.problem.minmax[0] * np.array(best_experiment.all_est_objectives[mrep]))
        best_mrep = np.argmax(best_exp_best_est_objectives)
        best_budget_idx = np.argmax(experiment.problem.minmax[0] * np.array(best_experiment.all_est_objectives[best_mrep]))
        xstar = best_experiment.all_recommended_xs[best_mrep][best_budget_idx]
        # Take post-replications at x*.
        opt_soln = Solution(xstar, ref_experiment.problem)
        opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
        ref_experiment.problem.simulate(solution=opt_soln, m=n_postreps_init_opt)
        xstar_postreps = list(opt_soln.objectives[:n_postreps_init_opt][:, 0])  # 0 <- assuming only one objective
    # Compute signed initial optimality gap = f(x0) - f(x*).
    initial_obj_val = np.mean(x0_postreps)
    opt_obj_val = np.mean(xstar_postreps)
    initial_opt_gap = initial_obj_val - opt_obj_val
    # Store x0 and x* info and compute progress curves for each Experiment.
    for experiment in experiments:
        # DOUBLE-CHECK FOR SHALLOW COPY ISSUES.
        experiment.n_postreps_init_opt = n_postreps_init_opt
        experiment.crn_across_init_opt = crn_across_init_opt
        experiment.x0 = x0
        experiment.x0_postreps = x0_postreps
        experiment.xstar = xstar
        experiment.xstar_postreps = xstar_postreps
        # Construct objective and progress curves.
        experiment.objective_curves = []
        experiment.progress_curves = []
        for mrep in range(experiment.n_macroreps):
            est_objectives = []
            # Substitute estimates at x0 and x* (based on N postreplicates)
            # with new estimates (based on L postreplicates).
            for budget in range(len(experiment.all_intermediate_budgets[mrep])):
                if experiment.all_recommended_xs[mrep][budget] == x0:
                    est_objectives.append(np.mean(x0_postreps))
                elif experiment.all_recommended_xs[mrep][budget] == xstar:
                    est_objectives.append(np.mean(xstar_postreps))
                else:
                    est_objectives.append(experiment.all_est_objectives[mrep][budget])
            experiment.objective_curves.append(Curve(x_vals=experiment.all_intermediate_budgets[mrep], y_vals=est_objectives))
            # Normalize by initial optimality gap.
            norm_est_objectives = [(est_objective - opt_obj_val) / initial_opt_gap for est_objective in est_objectives]
            frac_intermediate_budgets = [budget / experiment.problem.factors["budget"] for budget in experiment.all_intermediate_budgets[mrep]]
            experiment.progress_curves.append(Curve(x_vals=frac_intermediate_budgets, y_vals=norm_est_objectives))
        # Save Experiment object to .pickle file.
        experiment.record_experiment_results()


def bootstrap_sample_all(experiments, bootstrap_rng, normalize=True):
    """
    Generate bootstrap samples of estimated progress curves (normalized
    and unnormalized) from a set of experiments.
    Arguments
    ---------
    experiments : list of list of wrapper_base.Experiment objects
        experiments of different solvers and/or problems
    bootstrap_rng : MRG32k3a object
        random number generator to use for bootstrapping
    normalize : bool
        normalize progress curves w.r.t. optimality gaps?
    Returns
    -------
    bootstrap_curves : list of list of list of wrapper_base.Curve objects
        bootstrapped estimated objective curves or estimated progress curves
        of all solutions from all macroreplications
    """
    n_solvers = len(experiments)
    n_problems = len(experiments[0])
    bootstrap_curves = [[[] for _ in range(n_problems)] for _ in range(n_solvers)]
    # Obtain a bootstrap sample from each experiment.
    for solver_idx in range(n_solvers):
        for problem_idx in range(n_problems):
            experiment = experiments[solver_idx][problem_idx]
            bootstrap_curves[solver_idx][problem_idx] = experiment.bootstrap_sample(bootstrap_rng, normalize)
            # Reset substream for next solver-problem pair.
            bootstrap_rng.reset_substream()
    # Advance substream of random number generator to prepare for next bootstrap sample.
    bootstrap_rng.advance_substream()
    return bootstrap_curves


def bootstrap_procedure(experiments, n_bootstraps, plot_type, beta=None, solve_tol=None, estimator=None, normalize=True):
    """
    Parameters
    ----------
    experiments : list of list of wrapper_base.Experiment objects
        experiments of different solvers and/or problems
    n_bootstraps : int > 0
        number of times to generate a bootstrap sample of estimated progress curves
    plot_type : string
        indicates which type of plot to produce
            "mean" : estimated mean progress curve
            "quantile" : estimated beta quantile progress curve
            "area_mean" : mean of area under progress curve
            "area_std_dev" : standard deviation of area under progress curve
            "solve_time_quantile" : beta quantile of solve time
            "solve_time_cdf" : cdf of solve time
            "cdf_solvability" : cdf solvability profile
            "quantile_solvability" : quantile solvability profile
            "diff_cdf_solvability" : difference of cdf solvability profiles
            "diff_quantile_solvability" : difference of quantile solvability profiles
    beta : float in (0,1)
        quantile to plot, e.g., beta quantile
    solve_tol : float in (0,1]
        relative optimality gap definining when a problem is solved
    estimator : float or wrapper_base.Curve object
        main estimator, e.g., mean convergence curve from an experiment
    normalize : bool
        normalize progress curves w.r.t. optimality gaps?
    Returns
    -------
    bs_CI_lower_bounds, bs_CI_upper_bounds = floats or wrapper_base.Curve objects
        lower and upper bound(s) of bootstrap CI(s), as floats or curves
    """
    # Create random number generator for bootstrap sampling.
    # Stream 1 dedicated for bootstrapping.
    bootstrap_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
    # Obtain n_bootstrap replications.
    bootstrap_replications = []
    for bs_index in range(n_bootstraps):
        # Generate bootstrap sample of estimated objective/progress curves.
        bootstrap_curves = bootstrap_sample_all(experiments, bootstrap_rng=bootstrap_rng, normalize=normalize)
        # Apply the functional of the bootstrap sample.
        bootstrap_replications.append(functional_of_curves(bootstrap_curves, plot_type, beta=beta, solve_tol=solve_tol))
    # Distinguish cases where functional returns a scalar vs a curve.
    if plot_type in {"area_mean", "area_std_dev", "solve_time_quantile"}:
        # Functional returns a scalar.
        bs_CI_lower_bounds, bs_CI_upper_bounds = compute_bootstrap_CI(bootstrap_replications, conf_level=0.95, bias_correction=True, overall_estimator=estimator)
    elif plot_type in {"mean", "quantile", "solve_time_cdf", "cdf_solvability", "quantile_solvability", "diff_cdf_solvability", "diff_quantile_solvability"}:
        # Functional returns a curve.
        unique_budgets = list(np.unique([budget for curve in bootstrap_replications for budget in curve.x_vals]))
        bs_CI_lbs = []
        bs_CI_ubs = []
        for budget in unique_budgets:
            bootstrap_subreplications = [curve.lookup(x=budget) for curve in bootstrap_replications]
            sub_estimator = estimator.lookup(x=budget)
            bs_CI_lower_bound, bs_CI_upper_bound = compute_bootstrap_CI(bootstrap_subreplications,
                                                                        conf_level=0.95,
                                                                        bias_correction=True,
                                                                        overall_estimator=sub_estimator
                                                                        )
            bs_CI_lbs.append(bs_CI_lower_bound)
            bs_CI_ubs.append(bs_CI_upper_bound)
        bs_CI_lower_bounds = Curve(x_vals=unique_budgets, y_vals=bs_CI_lbs)
        bs_CI_upper_bounds = Curve(x_vals=unique_budgets, y_vals=bs_CI_ubs)
    return bs_CI_lower_bounds, bs_CI_upper_bounds


def functional_of_curves(bootstrap_curves, plot_type, beta=0.5, solve_tol=0.1):
    """
    Compute a functional of the bootstrapped objective/progress curves.
    Parameters
    ----------
    bootstrap_curves : list of list of list of wrapper_base.Curve objects
        bootstrapped estimated objective curves or estimated progress curves
        of all solutions from all macroreplications
    plot_type : string
        indicates which type of plot to produce
            "mean" : estimated mean progress curve
            "quantile" : estimated beta quantile progress curve
            "area_mean" : mean of area under progress curve
            "area_std_dev" : standard deviation of area under progress curve
            "solve_time_quantile" : beta quantile of solve time
            "solve_time_cdf" : cdf of solve time
            "cdf_solvability" : cdf solvability profile
            "quantile_solvability" : quantile solvability profile
            "diff_cdf_solvability" : difference of cdf solvability profiles
            "diff_quantile_solvability" : difference of quantile solvability profiles
    beta : float in (0,1)
        quantile to plot, e.g., beta quantile
    solve_tol : float in (0,1]
        relative optimality gap definining when a problem is solved
    Returns
    -------
    functional : list
        functional of bootstrapped curves, e.g, mean progress curves,
        mean area under progress curve, quantile of crossing time, etc.
    """
    if plot_type == "mean":
        # Single experiment --> returns a curve.
        functional = mean_of_curves(bootstrap_curves[0][0])
    elif plot_type == "quantile":
        # Single experiment --> returns a curve.
        functional = quantile_of_curves(bootstrap_curves[0][0], beta=beta)
    elif plot_type == "area_mean":
        # Single experiment --> returns a scalar.
        functional = np.mean([curve.compute_area_under_curve() for curve in bootstrap_curves[0][0]])
    elif plot_type == "area_std_dev":
        # Single experiment --> returns a scalar.
        functional = np.std([curve.compute_area_under_curve() for curve in bootstrap_curves[0][0]], ddof=1)
    elif plot_type == "solve_time_quantile":
        # Single experiment --> returns a scalar
        functional = np.quantile([curve.compute_crossing_time(threshold=solve_tol) for curve in bootstrap_curves[0][0]], q=beta)
    elif plot_type == "solve_time_cdf":
        # Single experiment --> returns a curve.
        functional = cdf_of_curves_crossing_times(bootstrap_curves[0][0], threshold=solve_tol)
    elif plot_type == "cdf_solvability":
        # One solver, multiple problems --> returns a curve.
        functional = mean_of_curves([cdf_of_curves_crossing_times(curves=progress_curves, threshold=solve_tol) for progress_curves in bootstrap_curves[0]])
    elif plot_type == "quantile_solvability":
        # One solver, multiple problems --> returns a curve.
        functional = mean_of_curves([quantile_cross_jump(curves=progress_curves, threshold=solve_tol, beta=beta) for progress_curves in bootstrap_curves[0]])
    elif plot_type == "diff_cdf_solvability":
        # Two solvers, multiple problems --> returns a curve.
        solvability_profile_1 = mean_of_curves([cdf_of_curves_crossing_times(curves=progress_curves, threshold=solve_tol) for progress_curves in bootstrap_curves[0]])
        solvability_profile_2 = mean_of_curves([cdf_of_curves_crossing_times(curves=progress_curves, threshold=solve_tol) for progress_curves in bootstrap_curves[1]])
        functional = difference_of_curves(solvability_profile_1, solvability_profile_2)
    elif plot_type == "diff_quantile_solvability":
        # Two solvers, multiple problems --> returns a curve.
        solvability_profile_1 = mean_of_curves([quantile_cross_jump(curves=progress_curves, threshold=solve_tol, beta=beta) for progress_curves in bootstrap_curves[0]])
        solvability_profile_2 = mean_of_curves([quantile_cross_jump(curves=progress_curves, threshold=solve_tol, beta=beta) for progress_curves in bootstrap_curves[1]])
        functional = difference_of_curves(solvability_profile_1, solvability_profile_2)
    else:
        print("Not a valid plot type.")
    return functional


def compute_bootstrap_CI(observations, conf_level=0.95, bias_correction=True, overall_estimator=None):
    """
    Construct a bootstrap confidence interval for an estimator.
    Parameters
    ----------
    observations : list
        estimators from all bootstrap instances
    conf_level : float in (0,1)
        confidence level for confidence intervals, i.e., 1-gamma
    bias_correction : bool
        use bias-corrected bootstrap CIs (via percentile method)?
    overall estimator : float
        estimator to compute bootstrap confidence interval of
        (required for bias corrected CI)
    Returns
    -------
    bs_CI_lower_bound : float
        lower bound of bootstrap CI
    bs_CI_upper_bound : float
        upper bound of bootstrap CI
    """
    # Compute bootstrapping confidence interval via percentile method.
    # See Efron (1981) "Nonparameteric Standard Errors and Confidence Intervals."
    if bias_correction:
        if overall_estimator is None:
            print("Estimator required to compute bias-corrected CIs.")
        # For biased-corrected CIs, see equation (4.4) on page 146.
        z0 = norm.ppf(np.mean([obs < overall_estimator for obs in observations]))
        zconflvl = norm.ppf(conf_level)
        q_lower = norm.cdf(2 * z0 - zconflvl)
        q_upper = norm.cdf(2 * z0 + zconflvl)
    else:
        # For uncorrected CIs, see equation (4.3) on page 146.
        q_lower = (1 - conf_level) / 2
        q_upper = 1 - (1 - conf_level) / 2
    bs_CI_lower_bound = np.quantile(observations, q=q_lower)
    bs_CI_upper_bound = np.quantile(observations, q=q_upper)
    return bs_CI_lower_bound, bs_CI_upper_bound


def plot_bootstrap_CIs(bs_CI_lower_bounds, bs_CI_upper_bounds, color_str="C0"):
    """
    Plot bootstrap confidence intervals.
    Parameters
    ----------
    bs_CI_lower_bounds, bs_CI_upper_bounds : wrapper_base.Curve objects
        lower and upper bounds of bootstrap CIs, as curves
    color_str : str
        string indicating line color, e.g., "C0", "C1", etc.
    """
    bs_CI_lower_bounds.plot(color_str=color_str, curve_type="conf_bound")
    bs_CI_upper_bounds.plot(color_str=color_str, curve_type="conf_bound")
    # Shade space between curves.
    # Convert to full curves to get piecewise-constant shaded areas.
    plt.fill_between(x=bs_CI_lower_bounds.curve_to_full_curve().x_vals,
                     y1=bs_CI_lower_bounds.curve_to_full_curve().y_vals,
                     y2=bs_CI_upper_bounds.curve_to_full_curve().y_vals,
                     color=color_str,
                     alpha=0.2
                     )


def report_max_halfwidth(curve_pairs, normalize):
    """
    Compute and print caption for max halfwidth of one or more bootstrap CI curves
    Parameters
    ----------
    curve_pairs : list of list of wrapper_base.Curve objects
        list of paired bootstrap CI curves
    normalize : bool
        normalize progress curves w.r.t. optimality gaps?
    """
    # Compute max halfwidth of bootstrap confidence intervals.
    min_lower_bound = np.inf
    max_upper_bound = -np.inf
    max_halfwidths = []
    for curve_pair in curve_pairs:
        min_lower_bound = min(min_lower_bound, min(curve_pair[0].y_vals))
        max_upper_bound = max(max_upper_bound, max(curve_pair[1].y_vals))
        max_halfwidths.append(0.5 * max_difference_of_curves(curve_pair[1], curve_pair[0]))
    max_halfwidth = max(max_halfwidths)
    # Print caption about max halfwidth.
    if normalize:
        xloc = 0.05
        yloc = -0.35
    else:
        # xloc = 0.05 * budget of the problem
        xloc = 0.05 * curve_pairs[0][0].x_vals[-1]
        yloc = min_lower_bound - 0.25 * (max_upper_bound - min_lower_bound)
    txt = f"The max halfwidth of the bootstrap CIs is {round(max_halfwidth, 2)}."
    plt.text(x=xloc, y=yloc, s=txt)


def check_common_problem_and_reference(experiments):
    """
    Check if a collection of experiments have the same problem, x0, and x*.
    Parameters
    ----------
    experiments : list of wrapper_base.Experiment objects
        experiments of different solvers on a common problem
    """
    ref_experiment = experiments[0]
    for experiment in experiments:
        if experiment.problem != ref_experiment.problem:
            print("At least two experiments have different problem instances.")
        if experiment.x0 != ref_experiment.x0:
            print("At least two experiments have different starting solutions.")
        if experiment.xstar != ref_experiment.xstar:
            print("At least two experiments have different optimal solutions.")


def plot_progress_curves(experiments, plot_type, beta=0.50, normalize=True, all_in_one=True, plot_CIs=True, print_max_hw=True):
    """
    Plot individual or aggregate progress curves for one or more solvers
    on a single problem.
    Parameters
    ----------
    experiments : list of wrapper_base.Experiment objects
        experiments of different solvers on a common problem
    plot_type : string
        indicates which type of plot to produce
            "all" : all estimated progress curves
            "mean" : estimated mean progress curve
            "quantile" : estimated beta quantile progress curve
    beta : float in (0,1)
        quantile to plot, e.g., beta quantile
    normalize : bool
        normalize progress curves w.r.t. optimality gaps?
    all_in_one : bool
        plot curves together or separately
    plot_CIs : bool
        plot bootstrapping confidence intervals?
    print_max_hw : bool
        print caption with max half-width
    """
    # Check if problems are the same with the same x0 and x*.
    check_common_problem_and_reference(experiments)
    file_list = []
    # Set up plot.
    n_experiments = len(experiments)
    if all_in_one:
        ref_experiment = experiments[0]
        setup_plot(plot_type=plot_type,
                   solver_name="SOLVER SET",
                   problem_name=ref_experiment.problem.name,
                   normalize=normalize,
                   budget=ref_experiment.problem.factors["budget"],
                   beta=beta
                   )
        solver_curve_handles = []
        if print_max_hw:
            curve_pairs = []
        for exp_idx in range(n_experiments):
            experiment = experiments[exp_idx]
            color_str = "C" + str(exp_idx)
            if plot_type == "all":
                # Plot all estimated progress curves.
                if normalize:
                    handle = experiment.progress_curves[0].plot(color_str=color_str)
                    for curve in experiment.progress_curves[1:]:
                        curve.plot(color_str=color_str)
                else:
                    handle = experiment.objective_curves[0].plot(color_str=color_str)
                    for curve in experiment.objective_curves[1:]:
                        curve.plot(color_str=color_str)
            elif plot_type == "mean":
                # Plot estimated mean progress curve.
                if normalize:
                    estimator = mean_of_curves(experiment.progress_curves)
                else:
                    estimator = mean_of_curves(experiment.objective_curves)
                handle = estimator.plot(color_str=color_str)
            elif plot_type == "quantile":
                # Plot estimated beta-quantile progress curve.
                if normalize:
                    estimator = quantile_of_curves(experiment.progress_curves, beta)
                else:
                    estimator = quantile_of_curves(experiment.objective_curves, beta)
                handle = estimator.plot(color_str=color_str)
            else:
                print("Not a valid plot type.")
            solver_curve_handles.append(handle)
            if plot_CIs and plot_type != "all":
                # Note: "experiments" needs to be a list of list of Experiments.
                bs_CI_lb_curve, bs_CI_ub_curve = bootstrap_procedure(experiments=[[experiment]],
                                                                     n_bootstraps=100,
                                                                     plot_type=plot_type,
                                                                     beta=beta,
                                                                     estimator=estimator,
                                                                     normalize=normalize
                                                                     )
                plot_bootstrap_CIs(bs_CI_lb_curve, bs_CI_ub_curve, color_str=color_str)
                if print_max_hw:
                    curve_pairs.append([bs_CI_lb_curve, bs_CI_ub_curve])
        plt.legend(handles=solver_curve_handles, labels=[experiment.solver.name for experiment in experiments], loc="upper right")
        if print_max_hw and plot_type != "all":
            report_max_halfwidth(curve_pairs=curve_pairs, normalize=normalize)
        return [save_plot(solver_name="SOLVER SET",
                  problem_name=ref_experiment.problem.name,
                  plot_type=plot_type,
                  normalize=normalize
                  )]
    else:  # Plot separately.
        
        for experiment in experiments:
            setup_plot(plot_type=plot_type,
                       solver_name=experiment.solver.name,
                       problem_name=experiment.problem.name,
                       normalize=normalize,
                       budget=experiment.problem.factors["budget"],
                       beta=beta
                       )
            if plot_type == "all":
                # Plot all estimated progress curves.
                if normalize:
                    for curve in experiment.progress_curves:
                        curve.plot()
                else:
                    for curve in experiment.objective_curves:
                        curve.plot()
            elif plot_type == "mean":
                # Plot estimated mean progress curve.
                if normalize:
                    estimator = mean_of_curves(experiment.progress_curves)
                else:
                    estimator = mean_of_curves(experiment.objective_curves)
                estimator.plot()
            elif plot_type == "quantile":
                # Plot estimated beta-quantile progress curve.
                if normalize:
                    estimator = quantile_of_curves(experiment.progress_curves, beta)
                else:
                    estimator = quantile_of_curves(experiment.objective_curves, beta)
                estimator.plot()
            else:
                print("Not a valid plot type.")
            if plot_CIs and plot_type != "all":
                # Note: "experiments" needs to be a list of list of Experiments.
                bs_CI_lb_curve, bs_CI_ub_curve = bootstrap_procedure(experiments=[[experiment]],
                                                                     n_bootstraps=100,
                                                                     plot_type=plot_type,
                                                                     beta=beta,
                                                                     estimator=estimator,
                                                                     normalize=normalize
                                                                     )
                plot_bootstrap_CIs(bs_CI_lb_curve, bs_CI_ub_curve)
                if print_max_hw:
                    report_max_halfwidth(curve_pairs=[[bs_CI_lb_curve, bs_CI_ub_curve]], normalize=normalize)
            file_list.append(save_plot(solver_name=experiment.solver.name,
                      problem_name=experiment.problem.name,
                      plot_type=plot_type,
                      normalize=normalize
                      ))
    return file_list


def plot_solvability_cdfs(experiments, solve_tol=0.1, all_in_one=True, plot_CIs=True, print_max_hw=True):
    """
    Plot the solvability cdf for one or more solvers on a single problem.
    Arguments
    ---------
    experiments : list of wrapper_base.Experiment objects
        experiments of different solvers on a common problem
    solve_tol : float in (0,1]
        relative optimality gap definining when a problem is solved
    all_in_one : bool
        plot curves together or separately
    plot_CIs : bool
        plot bootstrapping confidence intervals?
    print_max_hw : bool
        print caption with max half-width
    """
    # Check if problems are the same with the same x0 and x*.
    file_list = []
    check_common_problem_and_reference(experiments)
    # Set up plot.
    n_experiments = len(experiments)
    if all_in_one:
        ref_experiment = experiments[0]
        setup_plot(plot_type="solve_time_cdf",
                   solver_name="SOLVER SET",
                   problem_name=ref_experiment.problem.name,
                   solve_tol=solve_tol
                   )
        solver_curve_handles = []
        if print_max_hw:
            curve_pairs = []
        for exp_idx in range(n_experiments):
            experiment = experiments[exp_idx]
            color_str = "C" + str(exp_idx)
            # Plot cdf of solve times.
            estimator = cdf_of_curves_crossing_times(experiment.progress_curves, threshold=solve_tol)
            handle = estimator.plot(color_str=color_str)
            solver_curve_handles.append(handle)
            if plot_CIs:
                # Note: "experiments" needs to be a list of list of Experiments.
                bs_CI_lb_curve, bs_CI_ub_curve = bootstrap_procedure(experiments=[[experiment]],
                                                                     n_bootstraps=100,
                                                                     plot_type="solve_time_cdf",
                                                                     solve_tol=solve_tol,
                                                                     estimator=estimator,
                                                                     normalize=True
                                                                     )
                plot_bootstrap_CIs(bs_CI_lb_curve, bs_CI_ub_curve, color_str=color_str)
                if print_max_hw:
                    curve_pairs.append([bs_CI_lb_curve, bs_CI_ub_curve])
        plt.legend(handles=solver_curve_handles, labels=[experiment.solver.name for experiment in experiments], loc="lower right")
        if print_max_hw:
            report_max_halfwidth(curve_pairs=curve_pairs, normalize=True)
        return [save_plot(solver_name="SOLVER SET",
                  problem_name=ref_experiment.problem.name,
                  plot_type="solve_time_cdf",
                  normalize=True,
                  extra=solve_tol
                  )]
    else:  # Plot separately.
        for experiment in experiments:
            setup_plot(plot_type="solve_time_cdf",
                       solver_name=experiment.solver.name,
                       problem_name=experiment.problem.name,
                       solve_tol=solve_tol
                       )
            estimator = cdf_of_curves_crossing_times(experiment.progress_curves, threshold=solve_tol)
            estimator.plot()
            if plot_CIs:
                # Note: "experiments" needs to be a list of list of Experiments.
                bs_CI_lb_curve, bs_CI_ub_curve = bootstrap_procedure(experiments=[[experiment]],
                                                                     n_bootstraps=100,
                                                                     plot_type="solve_time_cdf",
                                                                     solve_tol=solve_tol,
                                                                     estimator=estimator,
                                                                     normalize=True
                                                                     )
                plot_bootstrap_CIs(bs_CI_lb_curve, bs_CI_ub_curve)
                if print_max_hw:
                    report_max_halfwidth(curve_pairs=[[bs_CI_lb_curve, bs_CI_ub_curve]], normalize=True)
            file_list.append(save_plot(solver_name=experiment.solver.name,
                      problem_name=experiment.problem.name,
                      plot_type="solve_time_cdf",
                      normalize=True,
                      extra=solve_tol
                      ))
    return file_list          


def plot_area_scatterplots(experiments, all_in_one=True, plot_CIs=True, print_max_hw=True):
    """
    Plot a scatter plot of mean and standard deviation of area under progress curves.
    Either one plot for each solver or one plot for all solvers.
    Parameters
    ----------
    experiments : list of list of wrapper_base.Experiment objects
        experiments used to produce plots
    all_in_one : bool
        plot curves together or separately
    plot_CIs : bool
        plot bootstrapping confidence intervals?
    print_max_hw : bool
        print caption with max half-width
    """
    # Set up plot.
    n_solvers = len(experiments)
    n_problems = len(experiments[0])
    if all_in_one:
        marker_list = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]
        setup_plot(plot_type="area",
                   solver_name="SOLVER SET",
                   problem_name="PROBLEM SET"
                   )
        solver_names = [solver_experiments[0].solver.name for solver_experiments in experiments]
        solver_curve_handles = []
        # TO DO: Build up capability to print max half-width.
        if print_max_hw:
            curve_pairs = []
        for solver_idx in range(n_solvers):
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                color_str = "C" + str(solver_idx)
                marker_str = marker_list[solver_idx % len(marker_list)]  # Cycle through list of marker types.
                # Plot mean and standard deviation of area under progress curve.
                areas = [curve.compute_area_under_curve() for curve in experiment.progress_curves]
                mean_estimator = np.mean(areas)
                std_dev_estimator = np.std(areas, ddof=1)
                if plot_CIs:
                    # Note: "experiments" needs to be a list of list of Experiments.
                    mean_bs_CI_lb, mean_bs_CI_ub = bootstrap_procedure(experiments=[[experiment]],
                                                                       n_bootstraps=100,
                                                                       plot_type="area_mean",
                                                                       estimator=mean_estimator,
                                                                       normalize=True
                                                                       )
                    std_dev_bs_CI_lb, std_dev_bs_CI_ub = bootstrap_procedure(experiments=[[experiment]],
                                                                             n_bootstraps=100,
                                                                             plot_type="area_std_dev",
                                                                             estimator=std_dev_estimator,
                                                                             normalize=True
                                                                             )
                    # if print_max_hw:
                    #     curve_pairs.append([bs_CI_lb_curve, bs_CI_ub_curve])
                    x_err = [[mean_estimator - mean_bs_CI_lb], [mean_bs_CI_ub - mean_estimator]]
                    y_err = [[std_dev_estimator - std_dev_bs_CI_lb], [std_dev_bs_CI_ub - std_dev_estimator]]
                    handle = plt.errorbar(x=mean_estimator,
                                          y=std_dev_estimator,
                                          xerr=x_err,
                                          yerr=y_err,
                                          color=color_str,
                                          marker=marker_str,
                                          elinewidth=1
                                          )
                else:
                    handle = plt.scatter(x=mean_estimator, y=std_dev_estimator, color=color_str, marker=marker_str)
            solver_curve_handles.append(handle)
        plt.legend(handles=solver_curve_handles, labels=solver_names, loc="upper right")
        return [save_plot(solver_name="SOLVER SET",
                  problem_name="PROBLEM SET",
                  plot_type="area_scatterplot",
                  normalize=True
                  )]
    else:
        file_list = []
        for solver_idx in range(n_solvers):
            ref_experiment = experiments[solver_idx][0]
            setup_plot(plot_type="area",
                       solver_name=ref_experiment.solver.name,
                       problem_name="PROBLEM SET"
                       )
            if print_max_hw:
                curve_pairs = []
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                # Plot mean and standard deviation of area under progress curve.
                areas = [curve.compute_area_under_curve() for curve in experiment.progress_curves]
                mean_estimator = np.mean(areas)
                std_dev_estimator = np.std(areas, ddof=1)
                if plot_CIs:
                    # Note: "experiments" needs to be a list of list of Experiments.
                    mean_bs_CI_lb, mean_bs_CI_ub = bootstrap_procedure(experiments=[[experiment]],
                                                                       n_bootstraps=100,
                                                                       plot_type="area_mean",
                                                                       estimator=mean_estimator,
                                                                       normalize=True
                                                                       )
                    std_dev_bs_CI_lb, std_dev_bs_CI_ub = bootstrap_procedure(experiments=[[experiment]],
                                                                             n_bootstraps=100,
                                                                             plot_type="area_std_dev",
                                                                             estimator=std_dev_estimator,
                                                                             normalize=True
                                                                             )
                    # if print_max_hw:
                    #     curve_pairs.append([bs_CI_lb_curve, bs_CI_ub_curve])
                    x_err = [[mean_estimator - mean_bs_CI_lb], [mean_bs_CI_ub - mean_estimator]]
                    y_err = [[std_dev_estimator - std_dev_bs_CI_lb], [std_dev_bs_CI_ub - std_dev_estimator]]
                    handle = plt.errorbar(x=mean_estimator,
                                          y=std_dev_estimator,
                                          xerr=x_err,
                                          yerr=y_err,
                                          marker="o",
                                          color="C0",
                                          elinewidth=1
                                          )
                else:
                    handle = plt.scatter(x=mean_estimator, y=std_dev_estimator, color="C0", marker="o")
            file_list.append(save_plot(solver_name=experiment.solver.name,
                      problem_name="PROBLEM SET",
                      plot_type="area_scatterplot",
                      normalize=True
                      ))
    return file_list          


def plot_solvability_profiles(experiments, plot_type, all_in_one=True, plot_CIs=True, print_max_hw=True, solve_tol=0.1, beta=0.5, ref_solver=None):
    """
    Plot the (difference of) solvability profiles for each solver on a set of problems.
    Parameters
    ----------
    experiments : list of list of wrapper_base.Experiment objects
        experiments used to produce plots
    plot_type : string
        indicates which type of plot to produce
            "cdf_solvability" : cdf-solvability profile
            "quantile_solvability" : quantile-solvability profile
            "diff_cdf_solvability" : difference of cdf-solvability profiles
            "diff_quantile_solvability" : difference of quantile-solvability profiles
    all_in_one : bool
        plot curves together or separately
    plot_CIs : bool
        plot bootstrapping confidence intervals?
    print_max_hw : bool
        print caption with max half-width
    solve_tol : float in (0,1]
        relative optimality gap definining when a problem is solved
    beta : float in (0,1)
        quantile to compute, e.g., beta quantile
    ref_solver : str
        name of solver used as benchmark for difference profiles
    """
    # Set up plot.
    n_solvers = len(experiments)
    n_problems = len(experiments[0])
    if all_in_one:
        if plot_type == "cdf_solvability":
            setup_plot(plot_type=plot_type,
                       solver_name="SOLVER SET",
                       problem_name="PROBLEM SET",
                       solve_tol=solve_tol
                       )
        elif plot_type == "quantile_solvability":
            setup_plot(plot_type=plot_type,
                       solver_name="SOLVER SET",
                       problem_name="PROBLEM SET",
                       beta=beta,
                       solve_tol=solve_tol
                       )
        elif plot_type == "diff_cdf_solvability":
            setup_plot(plot_type=plot_type,
                       solver_name="SOLVER SET",
                       problem_name="PROBLEM SET",
                       solve_tol=solve_tol
                       )
        elif plot_type == "diff_quantile_solvability":
            setup_plot(plot_type=plot_type,
                       solver_name="SOLVER SET",
                       problem_name="PROBLEM SET",
                       beta=beta,
                       solve_tol=solve_tol
                       )
        solver_names = [solver_experiments[0].solver.name for solver_experiments in experiments]
        solver_curves = []
        solver_curve_handles = []
        for solver_idx in range(n_solvers):
            solver_sub_curves = []
            color_str = "C" + str(solver_idx)
            # For each problem compute the cdf or quantile of solve times.
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                if plot_type in {"cdf_solvability", "diff_cdf_solvability"}:
                    sub_curve = cdf_of_curves_crossing_times(curves=experiment.progress_curves, threshold=solve_tol)
                if plot_type in {"quantile_solvability", "diff_quantile_solvability"}:
                    sub_curve = quantile_cross_jump(curves=experiment.progress_curves, threshold=solve_tol, beta=beta)
                solver_sub_curves.append(sub_curve)
            # Plot solvability profile for the solver.
            # Exploit the fact that each solvability profile is an average of more basic curves.
            solver_curve = mean_of_curves(solver_sub_curves)
            # CAUTION: Using mean above requires an equal number of macro-replications per problem.
            solver_curves.append(solver_curve)
            if plot_type in {"cdf_solvability", "quantile_solvability"}:
                handle = solver_curve.plot(color_str=color_str)
                solver_curve_handles.append(handle)
                if plot_CIs:
                    # Note: "experiments" needs to be a list of list of Experiments.
                    bs_CI_lb_curve, bs_CI_ub_curve = bootstrap_procedure(experiments=[experiments[solver_idx]],
                                                                         n_bootstraps=100,
                                                                         plot_type=plot_type,
                                                                         solve_tol=solve_tol,
                                                                         beta=beta,
                                                                         estimator=solver_curve,
                                                                         normalize=True
                                                                         )
                    plot_bootstrap_CIs(bs_CI_lb_curve, bs_CI_ub_curve, color_str=color_str)

        if plot_type == "cdf_solvability":
            plt.legend(handles=solver_curve_handles, labels=solver_names, loc="lower right")
            return [save_plot(solver_name="SOLVER SET",
                      problem_name="PROBLEM SET",
                      plot_type=plot_type,
                      normalize=True,
                      extra=solve_tol
                      )]
        elif plot_type == "quantile_solvability":
            plt.legend(handles=solver_curve_handles, labels=solver_names, loc="lower right")
            return [save_plot(solver_name="SOLVER SET",
                      problem_name="PROBLEM SET",
                      plot_type=plot_type,
                      normalize=True,
                      extra=[solve_tol, beta]
                      )]
        elif plot_type in {"diff_cdf_solvability", "diff_quantile_solvability"}:
            print(solver_names)
            non_ref_solvers = [solver_name for solver_name in solver_names if solver_name != ref_solver]
            ref_solver_idx = solver_names.index(ref_solver)
            for solver_idx in range(n_solvers):
                if solver_idx is not ref_solver_idx:
                    diff_solver_curve = difference_of_curves(solver_curves[solver_idx], solver_curves[ref_solver_idx])
                    color_str = "C" + str(solver_idx)
                    handle = diff_solver_curve.plot(color_str=color_str)
                    solver_curve_handles.append(handle)
                    if plot_CIs:
                        # Note: "experiments" needs to be a list of list of Experiments.
                        bs_CI_lb_curve, bs_CI_ub_curve = bootstrap_procedure(experiments=[experiments[solver_idx], experiments[ref_solver_idx]],
                                                                             n_bootstraps=100,
                                                                             plot_type=plot_type,
                                                                             solve_tol=solve_tol,
                                                                             beta=beta,
                                                                             estimator=diff_solver_curve,
                                                                             normalize=True
                                                                             )
                        plot_bootstrap_CIs(bs_CI_lb_curve, bs_CI_ub_curve, color_str=color_str)

            offset_labels = [f"{non_ref_solver} - {ref_solver}" for non_ref_solver in non_ref_solvers]
            plt.legend(handles=solver_curve_handles, labels=offset_labels, loc="lower right")
            if plot_type == "diff_cdf_solvability":
                return [save_plot(solver_name="SOLVER SET",
                          problem_name="PROBLEM SET",
                          plot_type=plot_type,
                          normalize=True,
                          extra=solve_tol
                          )]
            elif plot_type == "diff_quantile_solvability":
                return [save_plot(solver_name="SOLVER SET",
                          problem_name="PROBLEM SET",
                          plot_type=plot_type,
                          normalize=True,
                          extra=[solve_tol, beta]
                          )]
    else:
        file_list = []
        solver_names = [solver_experiments[0].solver.name for solver_experiments in experiments]
        solver_curves = []
        for solver_idx in range(n_solvers):
            solver_sub_curves = []
            # For each problem compute the cdf or quantile of solve times.
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                if plot_type in {"cdf_solvability", "diff_cdf_solvability"}:
                    sub_curve = cdf_of_curves_crossing_times(curves=experiment.progress_curves, threshold=solve_tol)
                if plot_type in {"quantile_solvability", "diff_quantile_solvability"}:
                    sub_curve = quantile_cross_jump(curves=experiment.progress_curves, threshold=solve_tol, beta=beta)
                solver_sub_curves.append(sub_curve)
            # Plot solvability profile for the solver.
            # Exploit the fact that each solvability profile is an average of more basic curves.
            solver_curve = mean_of_curves(solver_sub_curves)
            solver_curves.append(solver_curve)
            if plot_type in {"cdf_solvability", "quantile_solvability"}:
                # Set up plot.
                if plot_type == "cdf_solvability":
                    file_list.append(setup_plot(plot_type=plot_type,
                               solver_name=experiments[solver_idx][0].solver.name,
                               problem_name="PROBLEM SET",
                               solve_tol=solve_tol
                               ))
                elif plot_type == "quantile_solvability":
                    file_list.append(setup_plot(plot_type=plot_type,
                               solver_name=experiments[solver_idx][0].solver.name,
                               problem_name="PROBLEM SET",
                               beta=beta,
                               solve_tol=solve_tol
                               ))
                handle = solver_curve.plot()
                if plot_CIs:
                    # Note: "experiments" needs to be a list of list of Experiments.
                    bs_CI_lb_curve, bs_CI_ub_curve = bootstrap_procedure(experiments=[experiments[solver_idx]],
                                                                         n_bootstraps=100,
                                                                         plot_type=plot_type,
                                                                         solve_tol=solve_tol,
                                                                         beta=beta,
                                                                         estimator=solver_curve,
                                                                         normalize=True
                                                                         )
                    plot_bootstrap_CIs(bs_CI_lb_curve, bs_CI_ub_curve)
                if plot_type == "cdf_solvability":
                    file_list.append(save_plot(solver_name=experiments[solver_idx][0].solver.name,
                              problem_name="PROBLEM SET",
                              plot_type=plot_type,
                              normalize=True,
                              extra=solve_tol
                              ))
                elif plot_type == "quantile_solvability":
                    file_list.append(save_plot(solver_name=experiments[solver_idx][0].solver.name,
                              problem_name="PROBLEM SET",
                              plot_type=plot_type,
                              normalize=True,
                              extra=[solve_tol, beta]
                              ))
        if plot_type in {"diff_cdf_solvability", "diff_quantile_solvability"}:
            non_ref_solvers = [solver_name for solver_name in solver_names if solver_name != ref_solver]
            ref_solver_idx = solver_names.index(ref_solver)
            for solver_idx in range(n_solvers):
                if solver_idx is not ref_solver_idx:
                    if plot_type == "diff_cdf_solvability":
                        file_list.append(setup_plot(plot_type=plot_type,
                                   solver_name=experiments[solver_idx][0].solver.name,
                                   problem_name="PROBLEM SET",
                                   solve_tol=solve_tol
                                   ))
                    elif plot_type == "diff_quantile_solvability":
                        file_list.append(setup_plot(plot_type=plot_type,
                                   solver_name=experiments[solver_idx][0].solver.name,
                                   problem_name="PROBLEM SET",
                                   beta=beta,
                                   solve_tol=solve_tol
                                   ))
                    diff_solver_curve = difference_of_curves(solver_curves[solver_idx], solver_curves[ref_solver_idx])
                    handle = diff_solver_curve.plot()
                    if plot_CIs:
                        # Note: "experiments" needs to be a list of list of Experiments.
                        bs_CI_lb_curve, bs_CI_ub_curve = bootstrap_procedure(experiments=[experiments[solver_idx], experiments[ref_solver_idx]],
                                                                             n_bootstraps=100,
                                                                             plot_type=plot_type,
                                                                             solve_tol=solve_tol,
                                                                             beta=beta,
                                                                             estimator=diff_solver_curve,
                                                                             normalize=True
                                                                             )
                        plot_bootstrap_CIs(bs_CI_lb_curve, bs_CI_ub_curve)
                    if plot_type == "diff_cdf_solvability":
                        file_list.append(save_plot(solver_name=experiments[solver_idx][0].solver.name,
                                  problem_name="PROBLEM SET",
                                  plot_type=plot_type,
                                  normalize=True,
                                  extra=solve_tol
                                  ))
                    elif plot_type == "diff_quantile_solvability":
                        file_list.append(save_plot(solver_name=experiments[solver_idx][0].solver.name,
                                  problem_name="PROBLEM SET",
                                  plot_type=plot_type,
                                  normalize=True,
                                  extra=[solve_tol, beta]
                                  ))
    return file_list          


def setup_plot(plot_type, solver_name="SOLVER SET", problem_name="PROBLEM SET", normalize=True, budget=None, beta=None, solve_tol=None):
    """
    Create new figure. Add labels to plot and reformat axes.
    Parameters
    ----------
    plot_type : string
        indicates which type of plot to produce
            "all" : all estimated progress curves
            "mean" : estimated mean progress curve
            "quantile" : estimated beta quantile progress curve
            "solve_time_cdf" : cdf of solve time
            "cdf_solvability" : cdf solvability profile
            "quantile_solvability" : quantile solvability profile
            "diff_cdf_solvability" : difference of cdf solvability profiles
            "diff_quantile_solvability" : difference of quantile solvability profiles
            "area" : area scatterplot
    solver_name : string
        name of solver
    problem_name : string
        name of problem
    normalize : Boolean
        normalize progress curves w.r.t. optimality gaps?
    budget : int
        budget of problem, measured in function evaluations
    beta : float in (0,1)
        quantile to compute, e.g., beta quantile
    solve_tol : float in (0,1]
        relative optimality gap definining when a problem is solved
    """
    plt.figure()
    # Set up axes and axis labels.
    if normalize:
        plt.xlabel("Fraction of Budget", size=14)
        plt.ylabel("Fraction of Initial Optimality Gap", size=14)
        plt.xlim((0, 1))
        plt.ylim((-0.1, 1.1))
    else:
        plt.xlabel("Budget", size=14)
        plt.ylabel("Objective Function Value", size=14)
        plt.xlim((0, budget))
    plt.tick_params(axis="both", which="major", labelsize=12)
    # Specify title (plus alternative y-axis label and alternative axes).
    if plot_type == "all":
        if normalize:
            title = f"{solver_name} on {problem_name}\nProgress Curves"
        else:
            title = f"{solver_name} on {problem_name}\nObjective Curves"
    elif plot_type == "mean":
        if normalize:
            title = f"{solver_name} on {problem_name}\nMean Progress Curve"
        else:
            title = f"{solver_name} on {problem_name}\nMean Objective Curve"
    elif plot_type == "quantile":
        if normalize:
            title = f"{solver_name} on {problem_name}\n{round(beta, 2)}-Quantile Progress Curve"
        else:
            title = f"{solver_name} on {problem_name}\n{round(beta, 2)}-Quantile Objective Curve"
    elif plot_type == "solve_time_cdf":
        plt.ylabel("Fraction of Macroreplications Solved", size=14)
        title = f"{solver_name} on {problem_name}\nCDF of {round(solve_tol, 2)}-Solve Times"
    elif plot_type == "cdf_solvability":
        plt.ylabel("Problem Averaged Solve Fraction", size=14)
        title = f"CDF-Solvability Profile for {solver_name}\nProfile of CDFs of {round(solve_tol, 2)}-Solve Times"
    elif plot_type == "quantile_solvability":
        plt.ylabel("Fraction of Problems Solved", size=14)
        title = f"Quantile Solvability Profile for {solver_name}\nProfile of {round(beta, 2)}-Quantiles of {round(solve_tol, 2)}-Solve Times"
    elif plot_type == "diff_cdf_solvability":
        plt.ylabel("Difference in Problem Averaged Solve Fraction", size=14)
        title = f"Difference of CDF-Solvability Profile for {solver_name}\nDifference of Profiles of CDFs of {round(solve_tol, 2)}-Solve Times"
        plt.plot([0, 1], [0, 0], color="black", linestyle="--")
        plt.ylim((-1, 1))
    elif plot_type == "diff_quantile_solvability":
        plt.ylabel("Difference in Fraction of Problems Solved", size=14)
        title = f"Difference of Quantile Solvability Profile for {solver_name}\nDifference of Profiles of {round(beta, 2)}-Quantiles of {round(solve_tol, 2)}-Solve Times"
        plt.plot([0, 1], [0, 0], color="black", linestyle="--")
        plt.ylim((-1, 1))
    elif plot_type == "area":
        plt.xlabel("Mean Area", size=14)
        plt.ylabel("Std Dev of Area")
        plt.xlim((0, 1))
        plt.ylim((0, 0.5))
        title = f"{solver_name}\nAreas Under Progress Curves"
    plt.title(title, size=14)


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
            "solve_time_cdf" : cdf of solve time
            "cdf_solvability" : cdf solvability profile
            "quantile_solvability" : quantile solvability profile
            "diff_cdf_solvability" : difference of cdf solvability profiles
            "diff_quantile_solvability" : difference of quantile solvability profiles
            "area" : area scatterplot
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
    elif plot_type == "solve_time_cdf":
        plot_name = f"cdf_{extra}_solve_times"
    elif plot_type == "cdf_solvability":
        plot_name = f"profile_cdf_{extra}_solve_times"
    elif plot_type == "quantile_solvability":
        plot_name = f"profile_{extra[1]}_quantile_{extra[0]}_solve_times"
    elif plot_type == "diff_cdf_solvability":
        plot_name = "diff_cdf_solvability_profile"
    elif plot_type == "diff_quantile_solvability":
        plot_name = "diff_quantile_solvability_profile"
    elif plot_type == "area":
        plot_name = "area_scatterplot"
    if not normalize:
        plot_name = plot_name + "_unnorm"
    path_name = f"experiments/plots/{solver_name}_on_{problem_name}_{plot_type}.png"
    # Reformat path_name to be suitable as a string literal.
    path_name = path_name.replace("\\", "")
    path_name = path_name.replace("$", "")
    path_name = path_name.replace(" ", "_")
    plt.savefig(path_name, bbox_inches="tight")
    return path_name


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
    all_model_fixed_factors : dict of dict
        fixed model factors for each problem
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
    solver_renames : list of strings
        user-specified names for solvers
    problem_renames : list of strings
        user-specified names for problems
    fixed_factors_filename : string
        name of .py file containing dictionaries of fixed factors
        for solvers/problems/models.
    """
    def __init__(self, solver_names, problem_names, solver_renames=None, problem_renames=None, fixed_factors_filename=None):
        self.n_solvers = len(solver_names)
        self.n_problems = len(problem_names)
        if solver_renames is None:
            self.solver_names = solver_names
        else:
            self.solver_names = solver_renames
        if problem_renames is None:
            self.problem_names = problem_names
        else:
            self.problem_names = problem_renames
        # Read in fixed solver/problem/model factors from .py file in the Experiments folder.
        # File should contain three dictionaries of dictionaries called
        #   - all_solver_fixed_factors
        #   - all_problem_fixed_factors
        #   - all_model_fixed_factors
        if fixed_factors_filename is None:
            self.all_solver_fixed_factors = {solver_name: {} for solver_name in self.solver_names}
            self.all_problem_fixed_factors = {problem_name: {} for problem_name in self.problem_names}
            self.all_model_fixed_factors = {problem_name: {} for problem_name in self.problem_names}
        else:
            fixed_factors_filename = "experiments.inputs." + fixed_factors_filename
            all_factors = importlib.import_module(fixed_factors_filename)
            self.all_solver_fixed_factors = getattr(all_factors, "all_solver_fixed_factors")
            self.all_problem_fixed_factors = getattr(all_factors, "all_problem_fixed_factors")
            self.all_model_fixed_factors = getattr(all_factors, "all_model_fixed_factors")
        # Create all problem-solver pairs (i.e., instances of Experiment class)
        self.experiments = []
        for solver_idx in range(self.n_solvers):
            solver_experiments = []
            for problem_idx in range(self.n_problems):
                try:
                    # If a file exists, read in Experiment object.
                    with open(f"experiments/outputs/{self.solver_names[solver_idx]}_on_{self.problem_names[problem_idx]}.pickle", "rb") as file:
                        next_experiment = pickle.load(file)
                    # TO DO: Check if the solver/problem/model factors in the file match
                    # those for the MetaExperiment.
                except Exception:
                    # If no file exists, create new Experiment object.
                    print(f"No experiment file exists for {self.solver_names[solver_idx]} on {self.problem_names[problem_idx]}. Creating new experiment.")
                    next_experiment = Experiment(solver_name=solver_names[solver_idx],
                                                 problem_name=problem_names[problem_idx],
                                                 solver_rename=self.solver_names[solver_idx],
                                                 problem_rename=self.problem_names[problem_idx],
                                                 solver_fixed_factors=self.all_solver_fixed_factors[self.solver_names[solver_idx]],
                                                 problem_fixed_factors=self.all_problem_fixed_factors[self.problem_names[problem_idx]],
                                                 model_fixed_factors=self.all_model_fixed_factors[self.problem_names[problem_idx]])
                solver_experiments.append(next_experiment)
            self.experiments.append(solver_experiments)

    def check_compatibility(self):
        """
        Check whether all experiments' solvers and problems are compatible.

        Returns
        -------
        error_str : str
            error message in the event any problem and solver are incompatible
        """
        error_str = ""
        for solver_idx in range(self.n_solvers):
            for problem_idx in range(self.n_problems):
                new_error_str = self.experiments[solver_idx][problem_idx].check_compatibility()
                if new_error_str != "":
                    error_str += f"For solver {self.solver_names[solver_idx]} and problem {self.problem_names[problem_idx]}... {new_error_str}"
        return error_str

    def run(self, n_macroreps):
        """
        Run n_macroreps of each solver on each problem.
        Arguments
        ---------
        n_macroreps : int
            number of macroreplications of the solver to run on the problem
        """
        for solver_idx in range(self.n_solvers):
            for problem_idx in range(self.n_problems):
                experiment = self.experiments[solver_idx][problem_idx]
                # If the problem-solver pair has not been run in this way before,
                # run it now and save result to .pickle file.
                if (getattr(experiment, "n_macroreps", None) != n_macroreps):
                    print(f"Running {n_macroreps} macro-replications of {experiment.solver.name} on {experiment.problem.name}.")
                    experiment.clear_run()
                    experiment.run(n_macroreps)

    def post_replicate(self, n_postreps, crn_across_budget=True, crn_across_macroreps=False):
        """
        For each problem-solver pair, run postreplications at solutions
        recommended by the solver on each macroreplication.
        Arguments
        ---------
        n_postreps : int
            number of postreplications to take at each recommended solution
        crn_across_budget : bool
            use CRN for post-replications at solutions recommended at different times?
        crn_across_macroreps : bool
            use CRN for post-replications at solutions recommended on different macroreplications?
        """
        for solver_index in range(self.n_solvers):
            for problem_index in range(self.n_problems):
                experiment = self.experiments[solver_index][problem_index]
                # If the problem-solver pair has not been post-replicated in this way before,
                # post-process it now.
                if (getattr(experiment, "n_postreps", None) != n_postreps
                        or getattr(experiment, "crn_across_budget", None) != crn_across_budget
                        or getattr(experiment, "crn_across_macroreps", None) != crn_across_macroreps):
                    print(f"Post-processing {experiment.solver.name} on {experiment.problem.name}.")
                    experiment.clear_postreplicate()
                    experiment.post_replicate(n_postreps, crn_across_budget, crn_across_macroreps)

    def post_normalize(self, n_postreps_init_opt, crn_across_init_opt=True):
        """
        Construct objective curves and (normalized) progress curves
        for all collections of experiments on all given problem.

        Parameters
        ----------
        experiments : list of wrapper_base.Experiment objects
            experiments of different solvers on a common problem
        n_postreps_init_opt : int
            number of postreplications to take at initial x0 and optimal x*
        crn_across_init_opt : bool
            use CRN for post-replications at solutions x0 and x*?
        """
        for problem_idx in range(self.n_problems):
            experiments_same_problem = [self.experiments[solver_idx][problem_idx] for solver_idx in range(self.n_solvers)]
            post_normalize(experiments=experiments_same_problem,
                           n_postreps_init_opt=n_postreps_init_opt,
                           crn_across_init_opt=crn_across_init_opt)
