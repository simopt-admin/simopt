"""Provide base classes for problem-solver pairs and helper functions for reading/writing data and plotting."""

from __future__ import annotations

import ast
import csv
import importlib
import itertools
import os
import pickle
import re
import shutil
import subprocess
import time
from multiprocessing import Pool
from typing import Literal

import matplotlib.lines as mpl_lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mrg32k3a.mrg32k3a import MRG32k3a
from scipy.stats import norm

from simopt.base import ObjectiveType, Problem, Solution, Solver, VariableType
from simopt.directory import (
    model_directory,
    problem_directory,
    solver_directory,
)

"""Set the default directory for saving experiment results."""
EXPERIMENT_DIR = os.path.join(
    os.getcwd(), "experiments", time.strftime("%Y-%m-%d_%H-%M-%S")
)


class Curve:
    """Base class for all curves.

    Attributes
    ----------
    x_vals : list [float]
        Values of horizontal components.
    y_vals : list [float]
        Values of vertical components.
    n_points : int
        Number of values in x- and y- vectors.

    Parameters
    ----------
    x_vals : list [float]
        Values of horizontal components.
    y_vals : list [float]
        Values of vertical components.

    """

    @property
    def x_vals(self) -> list[float]:
        """Values of horizontal components."""
        return self.__x_vals

    @property
    def y_vals(self) -> list[float]:
        """Values of vertical components."""
        return self.__y_vals

    @property
    def n_points(self) -> int:
        """Number of values in x- and y- vectors."""
        return len(self.x_vals)

    def __init__(self, x_vals: list[float], y_vals: list[float]) -> None:
        """Initialize a curve with x- and y-values.

        Parameters
        ----------
        x_vals : list[float]
            Values of horizontal components.
        y_vals : list[float]
            Values of vertical components.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(x_vals, list) or not all(
            [isinstance(x, (int, float)) for x in x_vals]
        ):
            error_msg = "x_vals must be a list of floats."
            raise TypeError(error_msg)
        if not isinstance(y_vals, list) or not all(
            [isinstance(y, (int, float)) for y in y_vals]
        ):
            error_msg = "y_vals must be a list of floats."
            raise TypeError(error_msg)
        # Value checking
        if len(x_vals) != len(y_vals):
            error_msg = f"Length of x- {len(x_vals)} and y-values {len(y_vals)} must be equal."
            raise ValueError(error_msg)

        # Each attribute is read-only
        self.__x_vals = x_vals
        self.__y_vals = y_vals

    def lookup(self, x_val: float) -> float:
        """Lookup the y-value of the curve at an intermediate x-value.

        Parameters
        ----------
        x_val : float
            X-value at which to lookup the y-value.

        Returns
        -------
        float
            Y-value corresponding to x.

        Raises
        ------
        TypeError

        """
        # Type checking
        if not isinstance(x_val, (int, float)):
            error_msg = "x_val must be a float."
            raise TypeError(error_msg)

        if x_val < self.x_vals[0]:
            return np.nan
        else:
            idx = np.max(np.where(np.array(self.x_vals) <= x_val))
            return self.y_vals[idx]

    def compute_crossing_time(self, threshold: float) -> float:
        """Compute the first time at which a curve drops below a given threshold.

        Parameters
        ----------
        threshold : float
            Value for which to find first crossing time.

        Returns
        -------
        float
            First time at which a curve drops below threshold.

        Raises
        ------
        TypeError

        """
        # Type checking
        if not isinstance(threshold, (int, float)):
            error_msg = "Threshold must be a float."
            raise TypeError(error_msg)

        # Use binary search to find the first x-value below threshold.
        # TODO: Test this
        # index = bisect.bisect_left(self.y_vals, threshold)
        # if index == self.n_points:
        #     return np.inf
        # else:
        #     return self.x_vals[index]

        for i in range(self.n_points):
            if self.y_vals[i] < threshold:
                return self.x_vals[i]
        # If threshold is never crossed, return infinity.
        return np.inf

    def compute_area_under_curve(self) -> float:
        """Compute the area under a curve.

        Returns
        -------
        float
            Area under the curve.

        """
        area = np.dot(self.y_vals[:-1], np.diff(self.x_vals))
        return area

    def curve_to_mesh(self, mesh: list[float]) -> Curve:
        """Create a curve defined at equally spaced x values.

        Parameters
        ----------
        mesh : list[float]
            List of uniformly spaced x-values.

        Returns
        -------
        ``experiment_base.Curve``
            Curve with equally spaced x-values.

        Raises
        ------
        TypeError

        """
        # Type checking
        if not isinstance(mesh, list) or not all(
            [isinstance(x, (int, float)) for x in mesh]
        ):
            error_msg = "Mesh must be a list of floats."
            raise TypeError(error_msg)

        mesh_curve = Curve(x_vals=mesh, y_vals=[self.lookup(x) for x in mesh])
        return mesh_curve

    def curve_to_full_curve(self) -> Curve:
        """Create a curve with duplicate x- and y-values to indicate steps.

        Returns
        -------
        ``experiment_base.Curve``
            Curve with duplicate x- and y-values.

        """
        duplicate_x_vals = [x for x in self.x_vals for _ in (0, 1)]
        duplicate_y_vals = [y for y in self.y_vals for _ in (0, 1)]
        full_curve = Curve(
            x_vals=duplicate_x_vals[1:], y_vals=duplicate_y_vals[:-1]
        )
        return full_curve

    def plot(
        self,
        color_str: str = "C0",
        curve_type: Literal["regular", "conf_bound"] = "regular",
    ) -> mpl_lines.Line2D:
        """Plot a curve.

        Parameters
        ----------
        color_str : str, default="C0"
            String indicating line color, e.g., "C0", "C1", etc.
        curve_type : str, default="regular"
            String indicating type of line: "regular" or "conf_bound".

        Returns
        -------
        ``matplotlib.lines.Line2D``
            Curve handle, to use when creating legends.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(color_str, str):
            error_msg = "Color must be a string."
            raise TypeError(error_msg)
        if not isinstance(curve_type, str):
            error_msg = "Curve type must be a string."
            raise TypeError(error_msg)
        # Value checking
        if curve_type not in ["regular", "conf_bound"]:
            error_msg = "Invalid curve type."
            raise ValueError(error_msg)

        if curve_type == "regular":
            linestyle = "-"
            linewidth = 2
        elif curve_type == "conf_bound":
            linestyle = "--"
            linewidth = 1
        (handle,) = plt.step(
            self.x_vals,
            self.y_vals,
            color=color_str,
            linestyle=linestyle,
            linewidth=linewidth,
            where="post",
        )
        return handle


def mean_of_curves(curves: list[Curve]) -> Curve:
    """Compute pointwise (w.r.t. x-values) mean of curves.

    Starting and ending x-values must coincide for all curves.

    Parameters
    ----------
    curves : list [``experiment_base.Curve``]
        Collection of curves to aggregate.

    Returns
    -------
    ``experiment_base.Curve object``
        Mean curve.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(curves, list) or not all(
        [isinstance(curve, Curve) for curve in curves]
    ):
        error_msg = "Curves must be a list of Curve objects."
        raise TypeError(error_msg)

    unique_x_vals = np.unique(
        [x_val for curve in curves for x_val in curve.x_vals]
    )
    mean_y_vals = [
        float(np.mean([curve.lookup(float(x_val)) for curve in curves]))
        for x_val in unique_x_vals
    ]
    mean_curve = Curve(x_vals=unique_x_vals.tolist(), y_vals=mean_y_vals)
    return mean_curve


def quantile_of_curves(curves: list[Curve], beta: float) -> Curve:
    """Compute pointwise (w.r.t. x values) quantile of curves.

    Starting and ending x values must coincide for all curves.

    Parameters
    ----------
    curves : list [``experiment_base.Curve``]
        Collection of curves to aggregate.
    beta : float
        Quantile level.

    Returns
    -------
    ``experiment_base.Curve``
        Quantile curve.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(curves, list) or not all(
        [isinstance(curve, Curve) for curve in curves]
    ):
        error_msg = "Curves must be a list of Curve objects."
        raise TypeError(error_msg)
    if not isinstance(beta, (int, float)):
        error_msg = "Beta must be a float."
        raise TypeError(error_msg)

    unique_x_vals = np.unique(
        [x_val for curve in curves for x_val in curve.x_vals]
    )
    quantile_y_vals = [
        float(np.quantile([curve.lookup(x_val) for curve in curves], q=beta))
        for x_val in unique_x_vals
    ]
    quantile_curve = Curve(
        x_vals=unique_x_vals.tolist(), y_vals=quantile_y_vals
    )
    return quantile_curve


def cdf_of_curves_crossing_times(
    curves: list[Curve], threshold: float
) -> Curve:
    """Compute the cdf of crossing times of curves.

    Parameters
    ----------
    curves : list [``experiment_base.Curve``]
        Collection of curves to aggregate.
    threshold : float
        Value for which to find first crossing time.

    Returns
    -------
    ``experiment_base.Curve``
        CDF of crossing times.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(curves, list) or not all(
        [isinstance(curve, Curve) for curve in curves]
    ):
        error_msg = "Curves must be a list of Curve objects."
        raise TypeError(error_msg)
    if not isinstance(threshold, (int, float)):
        error_msg = "Threshold must be a float."
        raise TypeError(error_msg)

    n_curves = len(curves)
    crossing_times = [
        curve.compute_crossing_time(threshold) for curve in curves
    ]
    unique_x_vals = [
        0,
        *list(
            np.unique(
                [
                    crossing_time
                    for crossing_time in crossing_times
                    if crossing_time < np.inf
                ]
            )
        ),
        1,
    ]
    cdf_y_vals = [
        sum(crossing_time <= x_val for crossing_time in crossing_times)
        / n_curves
        for x_val in unique_x_vals
    ]
    cdf_curve = Curve(x_vals=unique_x_vals, y_vals=cdf_y_vals)
    return cdf_curve


def quantile_cross_jump(
    curves: list[Curve], threshold: float, beta: float
) -> Curve:
    """Compute a simple curve with a jump at the quantile of the crossing times.

    Parameters
    ----------
    curves : list [``experiment_base.Curve``]
        Collection of curves to aggregate.
    threshold : float
        Value for which to find first crossing time.
    beta : float
        Quantile level.

    Returns
    -------
    ``experiment_base.Curve``
        Piecewise-constant curve with a jump at the quantile crossing time (if finite).

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(curves, list) or not all(
        [isinstance(curve, Curve) for curve in curves]
    ):
        error_msg = "Curves must be a list of Curve objects."
        raise TypeError(error_msg)
    if not isinstance(threshold, (int, float)):
        error_msg = "Threshold must be a float."
        raise TypeError(error_msg)
    if not isinstance(beta, (int, float)):
        error_msg = "Beta must be a float."
        raise TypeError(error_msg)

    solve_time_quantile = float(
        np.quantile(
            [
                curve.compute_crossing_time(threshold=threshold)
                for curve in curves
            ],
            q=beta,
        )
    )
    # Note: np.quantile will evaluate to np.nan if forced to interpolate
    # between a finite and infinite value. These are rare cases. Since
    # crossing times must be non-negative, the quantile should be mapped
    # to positive infinity.
    if solve_time_quantile == np.inf or np.isnan(solve_time_quantile):
        return Curve(x_vals=[0, 1], y_vals=[0, 0])
    else:
        return Curve(x_vals=[0, solve_time_quantile, 1], y_vals=[0, 1, 1])


def difference_of_curves(curve_1: Curve, curve_2: Curve) -> Curve:
    """Compute the difference of two curves (Curve 1 - Curve 2).

    Parameters
    ----------
    curve_1: ``experiment_base.Curve``
        First curve to take the difference of.
    curve_2 : ``experiment_base.Curve``
        Curves to take the difference of.

    Returns
    -------
    ``experiment_base.Curve``
        Difference of curves.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(curve_1, Curve):
        error_msg = "curve_1 must be a Curve object."
        raise TypeError(error_msg)
    if not isinstance(curve_2, Curve):
        error_msg = "curve_2 must be a Curve object."
        raise TypeError(error_msg)

    unique_x_vals_np = np.unique(curve_1.x_vals + curve_2.x_vals)
    unique_x_vals = [float(x_val) for x_val in unique_x_vals_np]
    difference_y_vals = [
        (curve_1.lookup(x_val) - curve_2.lookup(x_val))
        for x_val in unique_x_vals
    ]
    difference_curve = Curve(x_vals=unique_x_vals, y_vals=difference_y_vals)
    return difference_curve


def max_difference_of_curves(curve_1: Curve, curve_2: Curve) -> float:
    """Compute the maximum difference of two curves (Curve 1 - Curve 2).

    Parameters
    ----------
    curve_1: ``experiment_base.Curve``
        First curve to take the difference of.
    curve_2 : ``experiment_base.Curve``
        Curves to take the difference of.

    Returns
    -------
    float
        Maximum difference of curves.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(curve_1, Curve):
        error_msg = "curve_1 must be a Curve object."
        raise TypeError(error_msg)
    if not isinstance(curve_2, Curve):
        error_msg = "curve_2 must be a Curve object."
        raise TypeError(error_msg)

    difference_curve = difference_of_curves(curve_1, curve_2)
    max_diff = max(difference_curve.y_vals)
    return max_diff


class ProblemSolver:
    """Base class for running one solver on one problem.

    Attributes
    ----------
    solver : ``base.Solver``
        Simulation-optimization solver.
    problem : ``base.Problem``
        Simulation-optimization problem.
    n_macroreps : int
        Number of macroreplications run.
    file_name_path : str
        Path of .pickle file for saving ``experiment_base.ProblemSolver`` object.
    all_recommended_xs : list [list [tuple]]
        Sequences of recommended solutions from each macroreplication.
    all_intermediate_budgets : list [list]
        Sequences of intermediate budgets from each macroreplication.
    timings : list [float]
        Runtimes (in seconds) for each macroreplication.
    n_postreps : int
        Number of postreplications to take at each recommended solution.
    crn_across_budget : bool
        True if CRN used for post-replications at solutions recommended at
        different times, otherwise False.
    crn_across_macroreps : bool
        True if CRN used for post-replications at solutions recommended on
        different macroreplications, otherwise False.
    all_post_replicates : list [list [list]]
        All post-replicates from all solutions from all macroreplications.
    all_est_objectives : numpy array [numpy array]
        Estimated objective values of all solutions from all macroreplications.
    n_postreps_init_opt : int
        Number of postreplications to take at initial solution (x0) and
        optimal solution (x*).
    crn_across_init_opt : bool
        True if CRN used for post-replications at solutions x0 and x*, otherwise False.
    x0 : tuple
        Initial solution (x0).
    x0_postreps : list
        Post-replicates at x0.
    xstar : tuple
        Proxy for optimal solution (x*).
    xstar_postreps : list
        Post-replicates at x*.
    objective_curves : list [``experiment_base.Curve``]
        Curves of estimated objective function values,
        one for each macroreplication.
    progress_curves : list [``experiment_base.Curve``]
        Progress curves, one for each macroreplication.

    Parameters
    ----------
    solver_name : str, optional
        Name of solver.
    problem_name : str, optional
        Name of problem.
    solver_rename : str, optional
        User-specified name for solver.
    problem_rename : str, optional
        User-specified name for problem.
    solver : ``base.Solver``, optional
        Simulation-optimization solver.
    problem : ``base.Problem``, optional
        Simulation-optimization problem.
    solver_fixed_factors : dict, optional
        Dictionary of user-specified solver factors.
    problem_fixed_factors : dict, optional
        Dictionary of user-specified problem factors.
    model_fixed_factors : dict, optional
        Dictionary of user-specified model factors.
    file_name_path : str | os.PathLike, optional
        Path of .pickle file for saving ``experiment_base.ProblemSolver`` objects.
    create_pickle : bool, optional
        True if creating pickle file to store ProblemSolver object, False otherwise.

    """

    @property
    def solver(self) -> Solver:
        """Simulation-optimization solver."""
        return self.__solver

    @solver.setter
    def solver(self, solver: Solver) -> None:
        self.__solver = solver

    @property
    def problem(self) -> Problem:
        """Simulation-optimization problem."""
        return self.__problem

    @problem.setter
    def problem(self, problem: Problem) -> None:
        self.__problem = problem

    @property
    def n_macroreps(self) -> int:
        """Number of macroreplications run."""
        return self.__n_macroreps

    @n_macroreps.setter
    def n_macroreps(self, n_macroreps: int) -> None:
        self.__n_macroreps = n_macroreps

    @property
    def file_name_path(self) -> str | os.PathLike:
        """Path of .pickle file for saving ``experiment_base.ProblemSolver`` object."""
        return self.__file_name_path

    @file_name_path.setter
    def file_name_path(self, file_name_path: str | os.PathLike) -> None:
        self.__file_name_path = file_name_path

    @property
    def all_recommended_xs(self) -> list[list[tuple]]:
        """Sequences of recommended solutions from each macroreplication."""
        return self.__all_recommended_xs

    @all_recommended_xs.setter
    def all_recommended_xs(self, all_recommended_xs: list[list[tuple]]) -> None:
        self.__all_recommended_xs = all_recommended_xs

    @property
    def all_intermediate_budgets(self) -> list[list]:
        """Sequences of intermediate budgets from each macroreplication."""
        return self.__all_intermediate_budgets

    @all_intermediate_budgets.setter
    def all_intermediate_budgets(
        self, all_intermediate_budgets: list[list]
    ) -> None:
        self.__all_intermediate_budgets = all_intermediate_budgets

    @property
    def timings(self) -> list[float]:
        """Runtimes (in seconds) for each macroreplication."""
        return self.__timings

    @timings.setter
    def timings(self, timings: list[float]) -> None:
        self.__timings = timings

    @property
    def n_postreps(self) -> int:
        """Number of postreps to take at each recommended solution."""
        return self.__n_postreps

    @n_postreps.setter
    def n_postreps(self, n_postreps: int) -> None:
        self.__n_postreps = n_postreps

    @property
    def crn_across_budget(self) -> bool:
        """True if CRN used for post-replications at solutions recommended at different times, otherwise False."""
        return self.__crn_across_budget

    @crn_across_budget.setter
    def crn_across_budget(self, crn_across_budget: bool) -> None:
        self.__crn_across_budget = crn_across_budget

    @property
    def crn_across_macroreps(self) -> bool:
        """True if CRN used for post-replications at solutions recommended on different macroreplications, otherwise False."""
        return self.__crn_across_macroreps

    @crn_across_macroreps.setter
    def crn_across_macroreps(self, crn_across_macroreps: bool) -> None:
        self.__crn_across_macroreps = crn_across_macroreps

    @property
    def all_post_replicates(self) -> list[list[list]]:
        """All post-replicates from all solutions from all macroreplications."""
        return self.__all_post_replicates

    @all_post_replicates.setter
    def all_post_replicates(
        self, all_post_replicates: list[list[list]]
    ) -> None:
        self.__all_post_replicates = all_post_replicates

    @property
    def all_est_objectives(self) -> list[list[float]]:
        """Estimated objective values of all solutions from all macroreplications."""
        return self.__all_est_objectives

    @all_est_objectives.setter
    def all_est_objectives(self, all_est_objectives: list[list[float]]) -> None:
        self.__all_est_objectives = all_est_objectives

    @property
    def n_postreps_init_opt(self) -> int:
        """Number of postreps to take at initial solution (x0) and optimal solution (x*)."""
        return self.__n_postreps_init_opt

    @n_postreps_init_opt.setter
    def n_postreps_init_opt(self, n_postreps_init_opt: int) -> None:
        self.__n_postreps_init_opt = n_postreps_init_opt

    @property
    def crn_across_init_opt(self) -> bool:
        """True if CRN used for post-replications at solutions x0 and x*, otherwise False."""
        return self.__crn_across_init_opt

    @crn_across_init_opt.setter
    def crn_across_init_opt(self, crn_across_init_opt: bool) -> None:
        self.__crn_across_init_opt = crn_across_init_opt

    @property
    def x0(self) -> tuple:
        """Initial solution (x0)."""
        return self.__x0

    @x0.setter
    def x0(self, x0: tuple) -> None:
        self.__x0 = x0

    @property
    def x0_postreps(self) -> list:
        """Post-replicates at x0."""
        return self.__x0_postreps

    @x0_postreps.setter
    def x0_postreps(self, x0_postreps: list) -> None:
        self.__x0_postreps = x0_postreps

    @property
    def xstar(self) -> tuple:
        """Proxy for optimal solution (x*)."""
        return self.__xstar

    @xstar.setter
    def xstar(self, xstar: tuple) -> None:
        self.__xstar = xstar

    @property
    def xstar_postreps(self) -> list:
        """Post-replicates at x*."""
        return self.__xstar_postreps

    @xstar_postreps.setter
    def xstar_postreps(self, xstar_postreps: list) -> None:
        self.__xstar_postreps = xstar_postreps

    @property
    def objective_curves(self) -> list[Curve]:
        """Curves of estimated objective function values, one for each macroreplication."""
        return self.__objective_curves

    @objective_curves.setter
    def objective_curves(self, objective_curves: list[Curve]) -> None:
        self.__objective_curves = objective_curves

    @property
    def progress_curves(self) -> list[Curve]:
        """Progress curves, one for each macroreplication."""
        return self.__progress_curves

    @progress_curves.setter
    def progress_curves(self, progress_curves: list[Curve]) -> None:
        self.__progress_curves = progress_curves

    @property
    def has_run(self) -> bool:
        """True if the solver has been run on the problem, otherwise False."""
        return self.__has_run

    @has_run.setter
    def has_run(self, has_run: bool) -> None:
        self.__has_run = has_run

    @property
    def has_postreplicated(self) -> bool:
        """True if the solver has been postreplicated, otherwise False."""
        return self.__has_postreplicated

    @has_postreplicated.setter
    def has_postreplicated(self, has_postreplicated: bool) -> None:
        self.__has_postreplicated = has_postreplicated

    @property
    def has_postnormalized(self) -> bool:
        """True if the solver has been postprocessed, otherwise False."""
        return self.__has_postnormalized

    @has_postnormalized.setter
    def has_postnormalized(self, has_postnormalized: bool) -> None:
        self.__has_postnormalized = has_postnormalized

    def __init__(
        self,
        solver_name: str | None = None,
        problem_name: str | None = None,
        solver_rename: str | None = None,
        problem_rename: str | None = None,
        solver: Solver | None = None,
        problem: Problem | None = None,
        solver_fixed_factors: dict | None = None,
        problem_fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
        file_name_path: str | os.PathLike | None = None,
        create_pickle: bool = True,
    ) -> None:
        """Create a ProblemSolver object.

        There are two ways to create a ProblemSolver object:
        1. Provide the names of the solver and problem to look up in ``directory.py``.
        2. Provide the solver and problem objects to pair.

        Parameters
        ----------
        solver_name : str, optional
            Name of solver.
        problem_name : str, optional
            Name of problem.
        solver_rename : str, optional
            User-specified name for solver.
        problem_rename : str, optional
            User-specified name for problem.
        solver : ``base.Solver``, optional
            Simulation-optimization solver.
        problem : ``base.Problem``, optional
            Simulation-optimization problem.
        solver_fixed_factors : dict, optional
            Dictionary of user-specified solver factors.
        problem_fixed_factors : dict, optional
            Dictionary of user-specified problem factors.
        model_fixed_factors : dict, optional
            Dictionary of user-specified model factors.
        file_name_path : str | os.PathLike, optional
            Path of .pickle file for saving ``experiment_base.ProblemSolver`` objects.
        create_pickle : bool, optional
            True if creating pickle file to store ProblemSolver object, False otherwise.

        Raises
        ------
        TypeError
        ValueError

        """
        # Default arguments
        if solver_fixed_factors is None:
            solver_fixed_factors = {}
        if problem_fixed_factors is None:
            problem_fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        # Type checking
        if not isinstance(solver_name, (str, type(None))):
            error_msg = "Solver name must be a string or None."
            raise TypeError(error_msg)
        if not isinstance(problem_name, (str, type(None))):
            error_msg = "Problem name must be a string or None."
            raise TypeError(error_msg)
        if not isinstance(solver_rename, (str, type(None))):
            error_msg = "Solver rename must be a string or None."
            raise TypeError(error_msg)
        if not isinstance(problem_rename, (str, type(None))):
            error_msg = "Problem rename must be a string or None."
            raise TypeError(error_msg)
        if not isinstance(solver, (Solver, type(None))):
            error_msg = "Solver must be a Solver object or None."
            raise TypeError(error_msg)
        if not isinstance(problem, (Problem, type(None))):
            error_msg = "Problem must be a Problem object or None."
            raise TypeError(error_msg)
        if not isinstance(solver_fixed_factors, dict):
            error_msg = "Solver fixed factors must be a dictionary or None."
            raise TypeError(error_msg)
        if not isinstance(problem_fixed_factors, dict):
            error_msg = "Problem fixed factors must be a dictionary or None."
            raise TypeError(error_msg)
        if not isinstance(model_fixed_factors, dict):
            error_msg = "Model fixed factors must be a dictionary or None."
            raise TypeError(error_msg)
        if not isinstance(file_name_path, (str, os.PathLike, type(None))):
            error_msg = "File name path must be a string or None."
            raise TypeError(error_msg)
        if not isinstance(create_pickle, bool):
            error_msg = "Create pickle must be a boolean."
            raise TypeError(error_msg)

        # Initialize values
        self.create_pickle = create_pickle
        self.has_run = False
        self.has_postreplicated = False
        self.has_postnormalized = False
        self.xstar = ()

        # Initialize solver.
        if isinstance(solver, Solver):  # Method 2
            self.solver = solver
        else:  # Method 1
            if solver_name is None:
                error_msg = "Solver name must be provided if solver object is not provided."
                raise ValueError(error_msg)
            if solver_name not in solver_directory:
                error_msg = "Solver name not found in solver directory."
                raise ValueError(error_msg)
            self.solver = solver_directory[solver_name](
                fixed_factors=solver_fixed_factors
            )
        # Rename solver if necessary.
        if solver_rename is not None:
            if solver_rename == "":
                error_msg = "Solver rename cannot be an empty string."
                raise ValueError(error_msg)
            self.solver.name = solver_rename

        # Initialize problem.
        if isinstance(problem, Problem):  # Method #2
            self.problem = problem
        else:  # Method #1
            if problem_name is None:
                error_msg = "Problem name must be provided if problem object is not provided."
                raise ValueError(error_msg)
            if problem_name not in problem_directory:
                error_msg = "Problem name not found in problem directory."
                raise ValueError(error_msg)
            self.problem = problem_directory[problem_name](
                fixed_factors=problem_fixed_factors,
                model_fixed_factors=model_fixed_factors,
            )
        # Rename problem if necessary.
        if problem_rename is not None:
            if problem_rename == "":
                error_msg = "Problem rename cannot be an empty string."
                raise ValueError(error_msg)
            self.problem.name = problem_rename

        # Initialize file path.
        if file_name_path is None:
            self.file_name_path = os.path.join(
                EXPERIMENT_DIR,
                f"{self.solver.name}_on_{self.problem.name}.pickle",
            )
        else:
            self.file_name_path = file_name_path

    # TODO: Convert this to throwing exceptions?
    # TODO: Convert this functionality to run automatically
    def check_compatibility(self) -> str:
        """Check whether the experiment's solver and problem are compatible.

        Returns
        -------
        str
            Error message in the event problem and solver are incompatible.

        """
        # make a string builder
        error_messages = []
        # Check number of objectives.
        if (
            self.solver.objective_type == ObjectiveType.SINGLE
            and self.problem.n_objectives > 1
        ):
            error_message = "Solver cannot solve a multi-objective problem"
            error_messages.append(error_message)
        elif (
            self.solver.objective_type == ObjectiveType.MULTI
            and self.problem.n_objectives == 1
        ):
            error_message = "Solver cannot solve a single-objective problem"
            error_messages.append(error_message)
        # Check constraint types.
        if (
            self.solver.constraint_type.value
            < self.problem.constraint_type.value
        ):
            solver_str = self.solver.constraint_type.name.lower()
            problem_str = self.problem.constraint_type.name.lower()
            error_message = f"Solver can handle {solver_str} constraints, but problem has {problem_str} constraints."
            error_messages.append(error_message)
        # Check variable types.
        if (
            self.solver.variable_type == VariableType.DISCRETE
            and self.problem.variable_type != VariableType.DISCRETE
        ):
            problem_type = self.problem.variable_type.name.lower()
            error_message = f"Solver is for discrete variables but problem variables are {problem_type}."
            error_messages.append(error_message)
        elif (
            self.solver.variable_type == VariableType.CONTINUOUS
            and self.problem.variable_type != VariableType.CONTINUOUS
        ):
            problem_type = self.problem.variable_type.name.lower()
            error_message = f"Solver is for continuous variables but problem variables are {problem_type}."
            error_messages.append(error_message)
        # Check for existence of gradient estimates.
        if self.solver.gradient_needed and not self.problem.gradient_available:
            error_message = "Solver requires gradient estimates but problem does not have them."
            error_messages.append(error_message)
        # Strip trailing newline character.
        error_str = "\n".join(error_messages)
        return error_str

    def run(self, n_macroreps: int) -> None:
        """Run n_macroreps of the solver on the problem.

        Notes
        -----
        RNGs dedicated for random problem instances and temporarily unused.
        Under development.

        Parameters
        ----------
        n_macroreps : int
            Number of macroreplications of the solver to run on the problem.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(n_macroreps, int):
            error_msg = "Number of macroreplications must be an integer."
            raise TypeError(error_msg)
        # Value checking
        if n_macroreps <= 0:
            error_msg = "Number of macroreplications must be positive."
            raise ValueError(error_msg)

        print(
            "Running Solver",
            self.solver.name,
            "on Problem",
            self.problem.name + ".",
        )

        # Initialize variables
        self.n_macroreps = n_macroreps
        self.all_recommended_xs = [[] for _ in range(n_macroreps)]
        self.all_intermediate_budgets = [[] for _ in range(n_macroreps)]
        self.timings = [0.0 for _ in range(n_macroreps)]

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
        # rng0 = MRG32k3a(s_ss_sss_index=[2, 0, 0])  # Currently unused.
        rng_list = [MRG32k3a(s_ss_sss_index=[2, i + 1, 0]) for i in range(3)]
        self.solver.attach_rngs(rng_list)

        # Start a timer
        self.function_start = time.time()

        print("Starting macroreplications in parallel")
        with Pool() as process_pool:
            # Start the macroreplications in parallel (async)
            result = process_pool.map_async(
                self.run_multithread, range(n_macroreps)
            )
            # Wait for the results to be returned (or 1 second)
            while not result.ready():
                # Update status bar here
                result.wait(1)

            # Grab all the data out of the result
            for mrep in range(n_macroreps):
                (
                    self.all_recommended_xs[mrep],
                    self.all_intermediate_budgets[mrep],
                    self.timings[mrep],
                ) = result.get()[mrep]
        print(
            f"Finished running {n_macroreps} macroreplications in {round(time.time() - self.function_start, 3)} seconds."
        )

        # Delete stuff we don't need to save
        del self.function_start

        self.has_run = True
        self.has_postreplicated = False
        self.has_postnormalized = False

        # Save ProblemSolver object to .pickle file if specified.
        if self.create_pickle:
            file_name = os.path.basename(self.file_name_path)
            self.record_experiment_results(file_name=file_name)

    def run_multithread(self, mrep: int) -> tuple:
        """Run a single macroreplication of the solver on the problem.

        Parameters
        ----------
        mrep : int
            Index of the macroreplication.

        Returns
        -------
        tuple
            Tuple of recommended solutions, intermediate budgets, and runtime.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(mrep, int):
            error_msg = "Macroreplication index must be an integer."
            raise TypeError(error_msg)
        # Value checking
        if mrep < 0:
            error_msg = "Macroreplication index must be non-negative."
            raise ValueError(error_msg)

        print(
            f"Macroreplication {mrep + 1}: Starting Solver {self.solver.name} on Problem {self.problem.name}."
        )
        # Create, initialize, and attach RNGs used for simulating solutions.
        progenitor_rngs = [
            MRG32k3a(s_ss_sss_index=[mrep + 3, ss, 0])
            for ss in range(self.problem.model.n_rngs)
        ]
        # Create a new set of RNGs for the solver based on the current macroreplication.
        # Tried re-using the progentior RNGs, but we need to match the number needed by the solver, not the problem
        solver_rngs = [
            MRG32k3a(
                s_ss_sss_index=[
                    mrep + 3,
                    self.problem.model.n_rngs + rng_index,
                    0,
                ]
            )
            for rng_index in range(len(self.solver.rng_list))
        ]

        # Set progenitor_rngs and rng_list for solver.
        self.solver.solution_progenitor_rngs = progenitor_rngs
        self.solver.rng_list = solver_rngs

        # print([rng.s_ss_sss_index for rng in progenitor_rngs])
        # Run the solver on the problem.
        tic = time.perf_counter()
        recommended_solns, intermediate_budgets = self.solver.solve(
            problem=self.problem
        )
        toc = time.perf_counter()
        runtime = toc - tic
        print(
            f"Macroreplication {mrep + 1}: Finished Solver {self.solver.name} on Problem {self.problem.name} in {runtime:0.4f} seconds."
        )

        # Trim the recommended solutions and intermediate budgets
        recommended_solns, intermediate_budgets = trim_solver_results(
            problem=self.problem,
            recommended_solutions=recommended_solns,
            intermediate_budgets=intermediate_budgets,
        )
        # Return tuple (rec_solns, int_budgets, runtime)
        return (
            [solution.x for solution in recommended_solns],
            intermediate_budgets,
            runtime,
        )

    def post_replicate(
        self,
        n_postreps: int,
        crn_across_budget: bool = True,
        crn_across_macroreps: bool = False,
    ) -> None:
        """Run postreplications at solutions recommended by the solver.

        Parameters
        ----------
        n_postreps : int
            Number of postreplications to take at each recommended solution.
        crn_across_budget : bool, default=True
            True if CRN used for post-replications at solutions recommended at different times,
            otherwise False.
        crn_across_macroreps : bool, default=False
            True if CRN used for post-replications at solutions recommended on different
            macroreplications, otherwise False.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(n_postreps, int):
            error_msg = "Number of postreplications must be an integer."
            raise TypeError(error_msg)
        if not isinstance(crn_across_budget, bool):
            error_msg = "CRN across budget must be a boolean."
            raise TypeError(error_msg)
        if not isinstance(crn_across_macroreps, bool):
            error_msg = "CRN across macroreplications must be a boolean."
            raise TypeError(error_msg)
        # Value checking
        if n_postreps <= 0:
            error_msg = "Number of postreplications must be positive."
            raise ValueError(error_msg)

        print(
            f"Setting up {n_postreps} postreplications for {self.n_macroreps} macroreplications of {self.solver.name} on {self.problem.name}."
        )

        self.n_postreps = n_postreps
        self.crn_across_budget = crn_across_budget
        self.crn_across_macroreps = crn_across_macroreps
        # Initialize variables
        self.all_post_replicates = [[] for _ in range(self.n_macroreps)]
        for mrep in range(self.n_macroreps):
            self.all_post_replicates[mrep] = [] * len(
                self.all_intermediate_budgets[mrep]
            )
        self.timings = [0.0 for _ in range(self.n_macroreps)]

        self.function_start = time.time()

        print("Starting postreplications in parallel")
        with Pool() as process_pool:
            # Start the macroreplications in parallel (async)
            result = process_pool.map_async(
                self.post_replicate_multithread, range(self.n_macroreps)
            )
            # Wait for the results to be returned (or 1 second)
            while not result.ready():
                # Update status bar here
                result.wait(1)

            # Grab all the data out of the result
            for mrep in range(self.n_macroreps):
                self.all_post_replicates[mrep], self.timings[mrep] = (
                    result.get()[mrep]
                )

            # # The all post replicates is tricky because it is a dictionary of lists of lists
            # # We need to convert it to a list of lists of lists
            # self.all_post_replicates = [self.all_post_replicates[i] for i in range(len(self.all_post_replicates.keys()))]
            # Store estimated objective for each macrorep for each budget.
            self.all_est_objectives = [
                [
                    float(np.mean(self.all_post_replicates[mrep][budget_index]))
                    for budget_index in range(
                        len(self.all_intermediate_budgets[mrep])
                    )
                ]
                for mrep in range(self.n_macroreps)
            ]
        print(
            f"Finished running {self.n_macroreps} postreplications in {round(time.time() - self.function_start, 3)} seconds."
        )

        # Delete stuff we don't need to save
        del self.function_start

        self.has_postreplicated = True
        self.has_postnormalized = False

        # Save ProblemSolver object to .pickle file if specified.
        if self.create_pickle:
            file_name = os.path.basename(self.file_name_path)
            self.record_experiment_results(file_name=file_name)

    def post_replicate_multithread(self, mrep: int) -> tuple:
        """Run postreplications at solutions recommended by the solver.

        Parameters
        ----------
        mrep : int
            Index of the macroreplication.

        Returns
        -------
        tuple
            Tuple of post-replicates and runtime.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(mrep, int):
            error_msg = "Macroreplication index must be an integer."
            raise TypeError(error_msg)
        # Value checking
        if mrep < 0:
            error_msg = "Macroreplication index must be non-negative."
            raise ValueError(error_msg)

        print(
            f"Macroreplication {mrep + 1}: Starting postreplications for {self.solver.name} on {self.problem.name}."
        )
        # Create RNG list for the macroreplication.
        if self.crn_across_macroreps:
            # Use the same RNGs for all macroreps.
            baseline_rngs = [
                MRG32k3a(
                    s_ss_sss_index=[0, self.problem.model.n_rngs + rng_index, 0]
                )
                for rng_index in range(self.problem.model.n_rngs)
            ]
        else:
            baseline_rngs = [
                MRG32k3a(
                    s_ss_sss_index=[
                        0,
                        self.problem.model.n_rngs * (mrep + 1) + rng_index,
                        0,
                    ]
                )
                for rng_index in range(self.problem.model.n_rngs)
            ]

        tic = time.perf_counter()

        # Create an empty list for each budget
        post_replicates = []
        # Loop over all recommended solutions.
        for budget_index in range(len(self.all_intermediate_budgets[mrep])):
            x = self.all_recommended_xs[mrep][budget_index]
            fresh_soln = Solution(x, self.problem)
            # Attach RNGs for postreplications.
            # If CRN is used across budgets, then we should use a copy rather
            # than passing in the original RNGs.
            if self.crn_across_budget:
                fresh_soln.attach_rngs(rng_list=baseline_rngs, copy=True)
            else:
                fresh_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
            self.problem.simulate(
                solution=fresh_soln, num_macroreps=self.n_postreps
            )
            # Store results
            post_replicates.append(
                list(fresh_soln.objectives[: fresh_soln.n_reps][:, 0])
            )  # 0 <- assuming only one objective
        toc = time.perf_counter()
        runtime = toc - tic
        print(f"\t{mrep + 1}: Finished in {round(runtime, 3)} seconds")

        # Return tuple (post_replicates, runtime)
        return (post_replicates, runtime)

    def bootstrap_sample(
        self, bootstrap_rng: MRG32k3a, normalize: bool = True
    ) -> list[Curve]:
        """Generate a bootstrap sample of estimated objective curves or estimated progress curves.

        Parameters
        ----------
        bootstrap_rng : ``mrg32k3a.mrg32k3a.MRG32k3a``
            Random number generator to use for bootstrapping.
        normalize : bool, default=True
            True if progress curves are to be normalized w.r.t.
            optimality gaps, otherwise False.

        Returns
        -------
        bootstrap_curves : list [``experiment_base.Curve``]
            Bootstrapped estimated objective curves or estimated progress curves of all solutions from all bootstrapped macroreplications.

        Raises
        ------
        TypeError

        """
        # Type checking
        if not isinstance(bootstrap_rng, MRG32k3a):
            error_msg = "Bootstrap RNG must be an MRG32k3a object."
            raise TypeError(error_msg)
        if not isinstance(normalize, bool):
            error_msg = "Normalize must be a boolean."
            raise TypeError(error_msg)

        bootstrap_curves = []
        # Uniformly resample M macroreplications (with replacement) from 0, 1, ..., M-1.
        # Subsubstream 0: reserved for this outer-level bootstrapping.
        bs_mrep_idxs = bootstrap_rng.choices(
            range(self.n_macroreps), k=self.n_macroreps
        )
        # Advance RNG subsubstream to prepare for inner-level bootstrapping.
        bootstrap_rng.advance_subsubstream()
        # Subsubstream 1: reserved for bootstrapping at x0 and x*.
        # Bootstrap sample post-replicates at common x0.
        # Uniformly resample L postreps (with replacement) from 0, 1, ..., L-1.
        bs_postrep_idxs = bootstrap_rng.choices(
            range(self.n_postreps_init_opt), k=self.n_postreps_init_opt
        )
        # Compute the mean of the resampled postreplications.
        bs_initial_obj_val = np.mean(
            [self.x0_postreps[postrep] for postrep in bs_postrep_idxs]
        )
        # Reset subsubstream if using CRN across budgets.
        # This means the same postreplication indices will be used for resampling at x0 and x*.
        if self.crn_across_init_opt:
            bootstrap_rng.reset_subsubstream()
        # Bootstrap sample postreplicates at reference optimal solution x*.
        # Uniformly resample L postreps (with replacement) from 0, 1, ..., L.
        bs_postrep_idxs = bootstrap_rng.choices(
            range(self.n_postreps_init_opt), k=self.n_postreps_init_opt
        )
        # Compute the mean of the resampled postreplications.
        bs_optimal_obj_val = np.mean(
            [self.xstar_postreps[postrep] for postrep in bs_postrep_idxs]
        )
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
                bs_postrep_idxs = bootstrap_rng.choices(
                    range(self.n_postreps), k=self.n_postreps
                )
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
                        est_objectives.append(
                            np.mean(
                                [
                                    self.all_post_replicates[mrep][budget][
                                        postrep
                                    ]
                                    for postrep in bs_postrep_idxs
                                ]
                            )
                        )
                # Record objective or progress curve.
                if normalize:
                    frac_intermediate_budgets = [
                        budget / self.problem.factors["budget"]
                        for budget in self.all_intermediate_budgets[mrep]
                    ]
                    norm_est_objectives = [
                        (est_objective - bs_optimal_obj_val)
                        / bs_initial_opt_gap
                        for est_objective in est_objectives
                    ]
                    new_progress_curve = Curve(
                        x_vals=frac_intermediate_budgets,
                        y_vals=norm_est_objectives,
                    )
                    bootstrap_curves.append(new_progress_curve)
                else:
                    new_objective_curve = Curve(
                        x_vals=self.all_intermediate_budgets[mrep],
                        y_vals=est_objectives,
                    )
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
                        bs_postrep_idxs = bootstrap_rng.choices(
                            range(self.n_postreps), k=self.n_postreps
                        )
                        # Compute the mean of the resampled postreplications.
                        est_objectives.append(
                            np.mean(
                                [
                                    self.all_post_replicates[mrep][budget][
                                        postrep
                                    ]
                                    for postrep in bs_postrep_idxs
                                ]
                            )
                        )
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
                    frac_intermediate_budgets = [
                        budget / self.problem.factors["budget"]
                        for budget in self.all_intermediate_budgets[mrep]
                    ]
                    norm_est_objectives = [
                        (est_objective - bs_optimal_obj_val)
                        / bs_initial_opt_gap
                        for est_objective in est_objectives
                    ]
                    new_progress_curve = Curve(
                        x_vals=frac_intermediate_budgets,
                        y_vals=norm_est_objectives,
                    )
                    bootstrap_curves.append(new_progress_curve)
                else:
                    new_objective_curve = Curve(
                        x_vals=self.all_intermediate_budgets[mrep],
                        y_vals=est_objectives,
                    )
                    bootstrap_curves.append(new_objective_curve)
        return bootstrap_curves

    def record_experiment_results(self, file_name: os.PathLike | str) -> None:
        """Save ``experiment_base.ProblemSolver`` object to .pickle file.

        Parameters
        ----------
        file_name : str
            Name of .pickle file for saving ``experiment_base.ProblemSolver`` objects.
            File name is appended to the ``EXPERIMENT_DIR`` directory path.

        Raises
        ------
        FileNotFoundError

        """
        # Type checking
        if not isinstance(file_name, (str, os.PathLike)):
            error_msg = "File name must be a string or os.PathLike object."
            raise TypeError(error_msg)

        file_path = os.path.join(EXPERIMENT_DIR, file_name)
        folder_name = os.path.dirname(file_path)

        print(f"File Name: {file_name}")
        print(f"Folder Name: {folder_name}")
        print(f"File Path: {file_path}")

        # Create the directory if it does not exist.
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # Delete the file if it already exists.
        elif os.path.exists(file_path):
            os.remove(file_path)
        # Create and dump the object to the file
        with open(file_path, "xb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def log_experiment_results(self, print_solutions: bool = True) -> None:
        """Create readable .txt file from a problem-solver pair's .pickle file.

        Parameters
        ----------
        print_solutions : bool, optional
            True if recommended solutions are to be printed in the .txt file, otherwise False.

        Raises
        ------
        TypeError

        """
        # Type checking
        if not isinstance(print_solutions, bool):
            error_msg = "Print solutions must be a boolean."
            raise TypeError(error_msg)

        if not os.path.exists(EXPERIMENT_DIR):
            os.makedirs(EXPERIMENT_DIR)
        results_filepath = os.path.join(
            EXPERIMENT_DIR, "experiment_results.txt"
        )

        with open(results_filepath, "w") as file:
            # Title txt file with experiment information.
            file.write(str(self.file_name_path))
            file.write("\n")
            file.write(f"Problem: {self.problem.name}\n")
            file.write(f"Solver: {self.solver.name}\n\n")

            # Display model factors.
            file.write("Model Factors:\n")
            for key, value in self.problem.model.factors.items():
                # Excluding model factors corresponding to decision variables.
                if key not in self.problem.model_decision_factors:
                    file.write(f"\t{key}: {value}\n")
            file.write("\n")
            # Display problem factors.
            file.write("Problem Factors:\n")
            for key, value in self.problem.factors.items():
                file.write(f"\t{key}: {value}\n")
            file.write("\n")
            # Display solver factors.
            file.write("Solver Factors:\n")
            for key, value in self.solver.factors.items():
                file.write(f"\t{key}: {value}\n")
            file.write("\n")

            # Display macroreplication information.
            file.write(f"{self.n_macroreps} macroreplications were run.\n")
            # If results have been postreplicated, list the number of post-replications.
            if self.has_postreplicated:
                file.write(
                    f"{self.n_postreps} postreplications were run at each recommended solution.\n\n"
                )
            # If post-normalized, state initial solution (x0) and proxy optimal solution (x_star)
            # and how many replications were taken of them (n_postreps_init_opt).
            if self.has_postnormalized:
                file.write(
                    f"The initial solution is {tuple([round(x, 4) for x in self.x0])}. Its estimated objective is {round(np.mean(self.x0_postreps), 4)}.\n"
                )
                if self.xstar is None:
                    file.write(
                        f"No proxy optimal solution was used. A proxy optimal objective function value of {self.problem.optimal_value} was provided.\n"
                    )
                else:
                    file.write(
                        f"The proxy optimal solution is {tuple([round(x, 4) for x in self.xstar])}. Its estimated objective is {round(np.mean(self.xstar_postreps), 4)}.\n"
                    )
                file.write(
                    f"{self.n_postreps_init_opt} postreplications were taken at x0 and x_star.\n\n"
                )
            # Display recommended solution at each budget value for each macroreplication.
            file.write("Macroreplication Results:\n")
            for mrep in range(self.n_macroreps):
                file.write(f"\nMacroreplication {mrep + 1}:\n")
                for budget in range(len(self.all_intermediate_budgets[mrep])):
                    file.write(
                        f"\tBudget: {round(self.all_intermediate_budgets[mrep][budget], 4)}"
                    )
                    # Optionally print solutions.
                    if print_solutions:
                        file.write(
                            f"\tRecommended Solution: {tuple([round(x, 4) for x in self.all_recommended_xs[mrep][budget]])}"
                        )
                    # If postreplicated, add estimated objective function values.
                    if self.has_postreplicated:
                        file.write(
                            f"\tEstimated Objective: {round(self.all_est_objectives[mrep][budget], 4)}\n"
                        )
                # file.write(f"\tThe time taken to complete this macroreplication was {round(self.timings[mrep], 2)} s.\n")
        file.close()


def trim_solver_results(
    problem: Problem,
    recommended_solutions: list[Solution],
    intermediate_budgets: list[int],
) -> tuple[list[Solution], list[int]]:
    """Trim solutions recommended by solver after problem's max budget.

    Parameters
    ----------
    problem : ``base.Problem``
        Problem object on which the solver was run.
    recommended_solutions : list [``base.Solution``]
        Solutions recommended by the solver.
    intermediate_budgets : list [int]
        Intermediate budgets at which solver recommended different solutions.

    Returns
    -------
    list [``base.Solution``]
        Solutions recommended by the solver after trimming.
    list [int]
        Intermediate budgets at which solver recommended different solutions after trimming.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(problem, Problem):
        error_msg = "Problem must be a Problem object."
        raise TypeError(error_msg)
    if not isinstance(recommended_solutions, list) or not all(
        isinstance(solution, Solution) for solution in recommended_solutions
    ):
        error_msg = "Recommended solutions must be a list of Solution objects."
        raise TypeError(error_msg)
    if not isinstance(intermediate_budgets, list) or not all(
        isinstance(budget, int) for budget in intermediate_budgets
    ):
        error_msg = "Intermediate budgets must be a list of integers."
        if not isinstance(intermediate_budgets, list):
            error_msg += f" Found {type(intermediate_budgets)}."
        else:
            error_msg += (
                f" Found {[type(budget) for budget in intermediate_budgets]}."
            )
        raise TypeError(error_msg)

    # Remove solutions corresponding to intermediate budgets exceeding max budget.
    invalid_idxs = [
        idx
        for idx, element in enumerate(intermediate_budgets)
        if element > problem.factors["budget"]
    ]
    for invalid_idx in sorted(invalid_idxs, reverse=True):
        del recommended_solutions[invalid_idx]
        del intermediate_budgets[invalid_idx]
    # If no solution is recommended at the final budget,
    # re-recommend the latest recommended solution.
    # (Necessary for clean plotting of progress curves.)
    if intermediate_budgets[-1] < problem.factors["budget"]:
        recommended_solutions.append(recommended_solutions[-1])
        intermediate_budgets.append(problem.factors["budget"])
    return recommended_solutions, intermediate_budgets


def read_experiment_results(file_name_path: str | os.PathLike) -> ProblemSolver:
    """Read in ``experiment_base.ProblemSolver`` object from .pickle file.

    Parameters
    ----------
    file_name_path : str
        Path of .pickle file for reading ``experiment_base.ProblemSolver`` object.

    Returns
    -------
    experiment : ``experiment_base.ProblemSolver``
        Problem-solver pair that has been run or has been post-processed.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(file_name_path, (str, os.PathLike)):
        error_msg = "File name path must be a string or os.PathLike object."
        raise TypeError(error_msg)
    # Value checking
    if not os.path.exists(file_name_path):
        error_msg = f"File {file_name_path} does not exist."
        raise ValueError(error_msg)  # Change to FileNotFoundError?

    with open(file_name_path, "rb") as file:
        experiment = pickle.load(file)
    return experiment


def post_normalize(
    experiments: list[ProblemSolver],
    n_postreps_init_opt: int,
    crn_across_init_opt: bool = True,
    proxy_init_val: float | None = None,
    proxy_opt_val: float | None = None,
    proxy_opt_x: tuple | None = None,
    create_pair_pickles: bool = False,
) -> None:
    """Construct objective curves and (normalized) progress curves for a collection of experiments on a given problem.

    Parameters
    ----------
    experiments : list [``experiment_base.ProblemSolver``]
        Problem-solver pairs of different solvers on a common problem.
    n_postreps_init_opt : int
        Number of postreplications to take at initial x0 and optimal x*.
    crn_across_init_opt : bool, default=True
        True if CRN used for post-replications at solutions x0 and x*, otherwise False.
    proxy_init_val : float, optional
        Known objective function value of initial solution.
    proxy_opt_val : float, optional
        Proxy for or bound on optimal objective function value.
    proxy_opt_x : tuple, optional
        Proxy for optimal solution.
    create_pair_pickles : bool, default = False
        True if creating pickle files for each problem-solver pair, False otherwise.

    Raises
    ------
    TypeError
    Exception

    """
    # Type checking
    if not isinstance(experiments, list) or not all(
        isinstance(experiment, ProblemSolver) for experiment in experiments
    ):
        error_msg = "Experiments must be a list of ProblemSolver objects."
        raise TypeError(error_msg)
    if not isinstance(n_postreps_init_opt, int):
        error_msg = (
            "Number of postreplications at x0 and x* must be an integer."
        )
        raise TypeError(error_msg)
    if not isinstance(crn_across_init_opt, bool):
        error_msg = "CRN across x0 and x* must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(proxy_init_val, (int, float, type(None))):
        error_msg = "Proxy initial value must be a float or None."
        raise TypeError(error_msg)
    if not isinstance(proxy_opt_val, (int, float, type(None))):
        error_msg = "Proxy optimal value must be a float or None."
        raise TypeError(error_msg)
    if not isinstance(proxy_opt_x, (tuple, type(None))):
        error_msg = "Proxy optimal solution must be a tuple or None."
        raise TypeError(error_msg)
    if not isinstance(create_pair_pickles, bool):
        error_msg = "Create pair pickles must be a boolean."
        raise TypeError(error_msg)

    # Check that all experiments have the same problem and same
    # post-experimental setup.
    ref_experiment = experiments[0]
    for experiment in experiments:
        # Check if problems are the same.
        if experiment.problem != ref_experiment.problem:
            error_msg = "At least two experiments have different problems."
            raise Exception(error_msg)
        # Check if experiments have common number of macroreps.
        if experiment.n_macroreps != ref_experiment.n_macroreps:
            error_msg = "At least two experiments have different numbers of macro-replications."
            raise Exception(error_msg)
        # Check if experiment has been post-replicated and with common number of postreps.
        if not experiment.has_run:
            error_msg = f"The experiment of {experiment.solver.name} on {experiment.problem.name} has not been run."
            raise Exception(error_msg)
        elif not experiment.has_postreplicated:
            error_msg = f"The experiment of {experiment.solver.name} on {experiment.problem.name} has not been post-replicated."
            raise Exception(error_msg)
        elif getattr(experiment, "n_postreps", None) != getattr(
            ref_experiment, "n_postreps", None
        ):
            error_msg = "At least two experiments have different numbers of post-replications.\nEstimation of optimal solution x* may be based on different numbers of post-replications."
            raise Exception(error_msg)
    print(f"Postnormalizing on Problem {ref_experiment.problem.name}.")
    # Take post-replications at common x0.
    # Create, initialize, and attach RNGs for model.
    #     Stream 0: reserved for post-replications.
    baseline_rngs = [
        MRG32k3a(s_ss_sss_index=[0, rng_index, 0])
        for rng_index in range(ref_experiment.problem.model.n_rngs)
    ]
    x0 = ref_experiment.problem.factors["initial_solution"]
    if proxy_init_val is not None:
        x0_postreps = [proxy_init_val] * n_postreps_init_opt
    else:
        initial_soln = Solution(x0, ref_experiment.problem)
        initial_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
        ref_experiment.problem.simulate(
            solution=initial_soln, num_macroreps=n_postreps_init_opt
        )
        x0_postreps = list(
            initial_soln.objectives[:n_postreps_init_opt][:, 0]
        )  # 0 <- assuming only one objective
    if crn_across_init_opt:
        # Reset each rng to start of its current substream.
        for rng in baseline_rngs:
            rng.reset_substream()
    # Determine (proxy for) optimal solution and/or (proxy for) its
    # objective function value. If deterministic (proxy for) f(x*),
    # create duplicate post-replicates to facilitate later bootstrapping.
    # If proxy for f(x*) is specified...
    print("Finding f(x*)...")
    if proxy_opt_val is not None:
        if proxy_opt_x is None:
            xstar = None
        else:
            xstar = (
                proxy_opt_x  # Assuming the provided x is optimal in this case.
            )
        print("\t...using provided proxy f(x*).")
        xstar_postreps = [proxy_opt_val] * n_postreps_init_opt
    # ...else if proxy for x* is specified...
    elif proxy_opt_x is not None:
        print("\t...using provided proxy x*.")
        xstar = proxy_opt_x
        # Take post-replications at xstar.
        opt_soln = Solution(xstar, ref_experiment.problem)
        opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
        ref_experiment.problem.simulate(
            solution=opt_soln, num_macroreps=n_postreps_init_opt
        )
        xstar_postreps = list(
            opt_soln.objectives[:n_postreps_init_opt][:, 0]
        )  # 0 <- assuming only one objective
    # ...else if f(x*) is known...
    elif ref_experiment.problem.optimal_value is not None:
        print("\t...using coded f(x*).")
        xstar = None
        # NOTE: optimal_value is a tuple.
        # Currently hard-coded for single objective case, i.e., optimal_value[0].
        xstar_postreps = [
            ref_experiment.problem.optimal_value
        ] * n_postreps_init_opt
    # ...else if x* is known...
    elif ref_experiment.problem.optimal_solution is not None:
        print("\t...using coded x*.")
        xstar = ref_experiment.problem.optimal_solution
        # Take post-replications at xstar.
        opt_soln = Solution(xstar, ref_experiment.problem)
        opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
        ref_experiment.problem.simulate(
            solution=opt_soln, num_macroreps=n_postreps_init_opt
        )
        xstar_postreps = list(
            opt_soln.objectives[:n_postreps_init_opt][:, 0]
        )  # 0 <- assuming only one objective
    # ...else determine x* empirically as estimated best solution
    # found by any solver on any macroreplication.
    else:
        print("\t...using best postreplicated solution as proxy for x*.")
        # TODO: Simplify this block of code.
        best_est_objectives = np.zeros(len(experiments))
        for experiment_idx in range(len(experiments)):
            experiment = experiments[experiment_idx]
            exp_best_est_objectives = np.zeros(experiment.n_macroreps)
            for mrep in range(experiment.n_macroreps):
                exp_best_est_objectives[mrep] = np.max(
                    experiment.problem.minmax[0]
                    * np.array(experiment.all_est_objectives[mrep])
                )
            best_est_objectives[experiment_idx] = np.max(
                exp_best_est_objectives
            )
        best_experiment_idx = np.argmax(best_est_objectives)
        best_experiment = experiments[best_experiment_idx]
        best_exp_best_est_objectives = np.zeros(ref_experiment.n_macroreps)
        for mrep in range(best_experiment.n_macroreps):
            best_exp_best_est_objectives[mrep] = np.max(
                best_experiment.problem.minmax[0]
                * np.array(best_experiment.all_est_objectives[mrep])
            )
        best_mrep = np.argmax(best_exp_best_est_objectives)
        best_budget_idx = np.argmax(
            ref_experiment.problem.minmax[0]
            * np.array(best_experiment.all_est_objectives[best_mrep])
        )
        xstar = best_experiment.all_recommended_xs[best_mrep][best_budget_idx]
        # Take post-replications at x*.
        opt_soln = Solution(xstar, ref_experiment.problem)
        opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
        ref_experiment.problem.simulate(
            solution=opt_soln, num_macroreps=n_postreps_init_opt
        )
        xstar_postreps = list(
            opt_soln.objectives[:n_postreps_init_opt][:, 0]
        )  # 0 <- assuming only one objective
    # Compute signed initial optimality gap = f(x0) - f(x*).
    initial_obj_val = np.mean(x0_postreps)
    opt_obj_val = np.mean(xstar_postreps)
    initial_opt_gap = initial_obj_val - opt_obj_val
    # Store x0 and x* info and compute progress curves for each ProblemSolver.
    for experiment in experiments:
        # DOUBLE-CHECK FOR SHALLOW COPY ISSUES.
        experiment.n_postreps_init_opt = n_postreps_init_opt
        experiment.crn_across_init_opt = crn_across_init_opt
        experiment.x0 = x0
        experiment.x0_postreps = x0_postreps
        if xstar is not None:
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
                    est_objectives.append(
                        experiment.all_est_objectives[mrep][budget]
                    )
            experiment.objective_curves.append(
                Curve(
                    x_vals=experiment.all_intermediate_budgets[mrep],
                    y_vals=est_objectives,
                )
            )
            # Normalize by initial optimality gap.
            if initial_opt_gap == 0:
                print(
                    "Warning: Divide by zero during post-normalization (initial_opt_gap is 0)."
                )
                norm_est_objectives = []
                for est_objective in est_objectives:
                    est_diff = est_objective - opt_obj_val
                    # Follow IEEE 754 standard for division by zero.
                    if est_diff < 0:
                        norm_est_objectives.append(-np.inf)
                    elif est_diff > 0:
                        norm_est_objectives.append(np.inf)
                    else:
                        norm_est_objectives.append(np.nan)
            else:
                norm_est_objectives = [
                    (est_objective - opt_obj_val) / initial_opt_gap
                    for est_objective in est_objectives
                ]
            frac_intermediate_budgets = [
                budget / experiment.problem.factors["budget"]
                for budget in experiment.all_intermediate_budgets[mrep]
            ]
            experiment.progress_curves.append(
                Curve(
                    x_vals=frac_intermediate_budgets, y_vals=norm_est_objectives
                )
            )

        experiment.has_postnormalized = True

        # Save ProblemSolver object to .pickle file if specified.
        if create_pair_pickles:
            file_name = os.path.basename(experiment.file_name_path)
            experiment.record_experiment_results(file_name=file_name)


def bootstrap_sample_all(
    experiments: list[list[ProblemSolver]],
    bootstrap_rng: MRG32k3a,
    normalize: bool = True,
) -> list[list[list[Curve]]]:
    """Generate bootstrap samples of estimated progress curves (normalized and unnormalized) from a set of experiments.

    Parameters
    ----------
    experiments : list [list [``experiment_base.ProblemSolver``]]
        Problem-solver pairs of different solvers and/or problems.
    bootstrap_rng : ``mrg32k3a.mrg32k3a.MRG32k3a``
        Random number generator to use for bootstrapping.
    normalize : bool, default=True
        True if progress curves are to be normalized w.r.t. optimality gaps,
        otherwise False.

    Returns
    -------
    list [list [list [``experiment_base.Curve``]]]
        Bootstrapped estimated objective curves or estimated progress curves
        of all solutions from all macroreplications.

    Raises
    ------
    TypeError

    """
    # Type checking
    if (
        not isinstance(experiments, list)
        or not all(
            isinstance(experiment_list, list) for experiment_list in experiments
        )
        or not all(
            [
                isinstance(experiment, ProblemSolver)
                for experiment in experiment_list
            ]
            for experiment_list in experiments
        )
    ):
        error_msg = (
            "Experiments must be a list of lists of ProblemSolver objects."
        )
        raise TypeError(error_msg)
    if not isinstance(bootstrap_rng, MRG32k3a):
        error_msg = "Bootstrap RNG must be an MRG32k3a object."
        raise TypeError(error_msg)
    if not isinstance(normalize, bool):
        error_msg = "Normalize must be a boolean."
        raise TypeError(error_msg)

    n_solvers = len(experiments)
    n_problems = len(experiments[0])
    bootstrap_curves = [
        [[] for _ in range(n_problems)] for _ in range(n_solvers)
    ]
    # Obtain a bootstrap sample from each experiment.
    for solver_idx in range(n_solvers):
        for problem_idx in range(n_problems):
            experiment = experiments[solver_idx][problem_idx]
            bootstrap_curves[solver_idx][problem_idx] = (
                experiment.bootstrap_sample(bootstrap_rng, normalize)
            )
            # Reset substream for next solver-problem pair.
            bootstrap_rng.reset_substream()
    # Advance substream of random number generator to prepare for next bootstrap sample.
    bootstrap_rng.advance_substream()
    return bootstrap_curves


def bootstrap_procedure(
    experiments: list[list[ProblemSolver]],
    n_bootstraps: int,
    conf_level: float,
    plot_type: Literal[
        "mean",
        "quantile",
        "area_mean",
        "area_std_dev",
        "solve_time_quantile",
        "solve_time_cdf",
        "cdf_solvability",
        "quantile_solvability",
        "diff_cdf_solvability",
        "diff_quantile_solvability",
    ],
    beta: float | None = None,
    solve_tol: float | None = None,
    estimator: float | Curve | None = None,
    normalize: bool = True,
) -> tuple[float, float] | tuple[Curve, Curve]:
    """Obtain bootstrap sample and compute confidence intervals.

    Parameters
    ----------
    experiments : list [list [``experiment_base.ProblemSolver``]]
        Problem-solver pairs of different solvers and/or problems.
    n_bootstraps : int
        Number of times to generate a bootstrap sample of estimated progress curves.
    conf_level : float
        Confidence level for confidence intervals, i.e., 1-gamma; in (0, 1).
    plot_type : str
            "mean" : estimated mean progress curve;

            "quantile" : estimated beta quantile progress curve;

            "area_mean" : mean of area under progress curve;

            "area_std_dev" : standard deviation of area under progress curve;

            "solve_time_quantile" : beta quantile of solve time;

            "solve_time_cdf" : cdf of solve time;

            "cdf_solvability" : cdf solvability profile;

            "quantile_solvability" : quantile solvability profile;

            "diff_cdf_solvability" : difference of cdf solvability profiles;

            "diff_quantile_solvability" : difference of quantile solvability profiles.
    beta : float, optional
        Quantile to plot, e.g., beta quantile; in (0, 1).
    solve_tol : float, optional
        Relative optimality gap definining when a problem is solved; in (0, 1].
    estimator : float or ``experiment_base.Curve``, optional
        Main estimator, e.g., mean convergence curve from an experiment.
    normalize : bool, default=True
        True if progress curves are to be normalized w.r.t. optimality gaps, otherwise False.

    Returns
    -------
    float | ``experiment_base.Curve``
        Lower bound of bootstrap CI, as a float or curve.
    float | ``experiment_base.Curve``]
        Upper bound of bootstrap CI, as a float or curve.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if (
        not isinstance(experiments, list)
        or not all(
            isinstance(experiment_list, list) for experiment_list in experiments
        )
        or not all(
            [
                isinstance(experiment, ProblemSolver)
                for experiment in experiment_list
            ]
            for experiment_list in experiments
        )
    ):
        error_msg = (
            "Experiments must be a list of lists of ProblemSolver objects."
        )
        raise TypeError(error_msg)
    if not isinstance(n_bootstraps, int):
        error_msg = "Number of bootstraps must be an integer."
        raise TypeError(error_msg)
    if not isinstance(conf_level, (int, float)):
        error_msg = "Confidence level must be a float."
        raise TypeError(error_msg)
    if not isinstance(plot_type, str):
        error_msg = "Plot type must be a string."
        raise TypeError(error_msg)
    if not isinstance(beta, (int, float, type(None))):
        error_msg = "Beta quantile must be a float or None."
        raise TypeError(error_msg)
    if not isinstance(solve_tol, (int, float, type(None))):
        error_msg = "Solve tolerance must be a float or None."
        raise TypeError(error_msg)
    if not isinstance(estimator, (int, float, Curve, type(None))):
        error_msg = "Estimator must be a float, Curve, or None."
        raise TypeError(error_msg)
    if not isinstance(normalize, bool):
        error_msg = "Normalize must be a boolean."
        raise TypeError(error_msg)
    # Value checking
    if n_bootstraps < 1:
        error_msg = "Number of bootstraps must be a positive integer."
        raise ValueError(error_msg)
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)
    if plot_type not in [
        "mean",
        "quantile",
        "area_mean",
        "area_std_dev",
        "solve_time_quantile",
        "solve_time_cdf",
        "cdf_solvability",
        "quantile_solvability",
        "diff_cdf_solvability",
        "diff_quantile_solvability",
    ]:
        error_msg = "Plot type must be a valid string."
        raise ValueError(error_msg)
    if beta is not None and not 0 < beta < 1:
        error_msg = "Beta quantile must be in (0, 1)."
        raise ValueError(error_msg)
    if solve_tol is not None and not 0 < solve_tol <= 1:
        error_msg = "Solve tolerance must be in (0, 1]."
        raise ValueError(error_msg)

    # Create random number generator for bootstrap sampling.
    # Stream 1 dedicated for bootstrapping.
    bootstrap_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
    # Obtain n_bootstrap replications.
    bootstrap_replications = []
    for _ in range(n_bootstraps):
        # Generate bootstrap sample of estimated objective/progress curves.
        bootstrap_curves = bootstrap_sample_all(
            experiments, bootstrap_rng=bootstrap_rng, normalize=normalize
        )
        # Apply the functional of the bootstrap sample.
        bootstrap_replications.append(
            functional_of_curves(
                bootstrap_curves, plot_type, beta=beta, solve_tol=solve_tol
            )
        )
    # Distinguish cases where functional returns a scalar vs a curve.
    if plot_type in ["area_mean", "area_std_dev", "solve_time_quantile"]:
        if estimator is None:
            error_msg = "Estimator must be provided for functional that returns a scalar."
            raise ValueError(error_msg)
        if isinstance(estimator, Curve):
            error_msg = "Estimator must be a scalar for functional that returns a scalar."
            raise ValueError(error_msg)
        # Functional returns a scalar.
        computed_bootstrap = compute_bootstrap_conf_int(
            bootstrap_replications,
            conf_level=conf_level,
            bias_correction=True,
            overall_estimator=estimator,
        )
        # Get the first and second float values from the computed bootstrap.
        float_1 = computed_bootstrap[0]
        float_2 = computed_bootstrap[1]
        # Keep indexing into them until they are floats.
        while not isinstance(float_1, (int, float)):
            float_1 = float_1[0]
        while not isinstance(float_2, (int, float)):
            float_2 = float_2[0]
        return float_1, float_2
    else:
        # Functional returns a curve.
        unique_budget_list = list(
            np.unique(
                [
                    budget
                    for curve in bootstrap_replications
                    for budget in curve.x_vals
                ]
            )
        )
        bs_conf_int_lower_bound_list: list[np.ndarray] = []
        bs_conf_int_upper_bound_list: list[np.ndarray] = []
        for budget in unique_budget_list:
            budget_float = float(budget)
            bootstrap_subreplications = [
                curve.lookup(budget_float) for curve in bootstrap_replications
            ]
            if estimator is None:
                error_msg = "Estimator must be provided for functional that returns a curve."
                raise ValueError(error_msg)
            if isinstance(estimator, (int, float)):
                error_msg = "Estimator must be a Curve object for functional that returns a curve."
                raise ValueError(error_msg)
            sub_estimator = estimator.lookup(budget_float)
            bs_conf_int_lower_bound, bs_conf_int_upper_bound = (
                compute_bootstrap_conf_int(
                    bootstrap_subreplications,
                    conf_level=conf_level,
                    bias_correction=True,
                    overall_estimator=sub_estimator,
                )
            )
            bs_conf_int_lower_bound_list.append(bs_conf_int_lower_bound)
            bs_conf_int_upper_bound_list.append(bs_conf_int_upper_bound)
        # Create the curves for the lower and upper bounds of the bootstrap
        # confidence intervals.
        unique_budget_list_floats = [float(val) for val in unique_budget_list]
        lower_bound_list = [float(val) for val in bs_conf_int_lower_bound_list]
        bs_conf_int_lower_bounds = Curve(
            x_vals=unique_budget_list_floats, y_vals=lower_bound_list
        )
        upper_bound_list = [float(val) for val in bs_conf_int_upper_bound_list]
        bs_conf_int_upper_bounds = Curve(
            x_vals=unique_budget_list_floats, y_vals=upper_bound_list
        )
        return bs_conf_int_lower_bounds, bs_conf_int_upper_bounds


def functional_of_curves(
    bootstrap_curves: list[list[list[Curve]]],
    plot_type: Literal[
        "mean",
        "quantile",
        "area_mean",
        "area_std_dev",
        "solve_time_quantile",
        "solve_time_cdf",
        "cdf_solvability",
        "quantile_solvability",
        "diff_cdf_solvability",
        "diff_quantile_solvability",
    ],
    beta: float | None = 0.5,
    solve_tol: float | None = 0.1,
) -> float | Curve:
    """Compute a functional of the bootstrapped objective/progress curves.

    Parameters
    ----------
    bootstrap_curves : list [list [list [``experiment_base.Curve``]]]
        Bootstrapped estimated objective curves or estimated progress curves
        of all solutions from all macroreplications.
    plot_type : str
        String indicating which type of plot to produce:
            "mean" : estimated mean progress curve;

            "quantile" : estimated beta quantile progress curve;

            "area_mean" : mean of area under progress curve;

            "area_std_dev" : standard deviation of area under progress curve;

            "solve_time_quantile" : beta quantile of solve time;

            "solve_time_cdf" : cdf of solve time;

            "cdf_solvability" : cdf solvability profile;

            "quantile_solvability" : quantile solvability profile;

            "diff_cdf_solvability" : difference of cdf solvability profiles;

            "diff_quantile_solvability" : difference of quantile solvability profiles;
    beta : float, default=0.5
        Quantile to plot, e.g., beta quantile; in (0, 1).
    solve_tol : float, default=0.1
        Relative optimality gap definining when a problem is solved; in (0, 1].

    Returns
    -------
    "Curve" | float
        Functional of bootstrapped curves, e.g, mean progress curves,
        mean area under progress curve, quantile of crossing time, etc.

    Raises
    ------
    TypeError
    ValueError

    """
    # Set default arguments
    if beta is None:
        beta = 0.5
    if solve_tol is None:
        solve_tol = 0.1
    # Type checking
    if (
        not isinstance(bootstrap_curves, list)
        or not all(
            isinstance(experiment_list_list, list)
            for experiment_list_list in bootstrap_curves
        )
        or not all(
            [isinstance(curve_list, list) for curve_list in curve_list_list]
            for curve_list_list in bootstrap_curves
        )
        or not all(
            [
                [isinstance(curve, Curve) for curve in curve_list]
                for curve_list in curve_list_list
            ]
            for curve_list_list in bootstrap_curves
        )
    ):
        error_msg = "Bootstrap curves must be a list of lists of lists of Curve objects."
        raise TypeError(error_msg)
    if not isinstance(plot_type, str):
        error_msg = "Plot type must be a string."
        raise TypeError(error_msg)
    if not isinstance(beta, (int, float, type(None))):
        error_msg = "Beta quantile must be a float."
        raise TypeError(error_msg)
    if not isinstance(solve_tol, (int, float, type(None))):
        error_msg = "Solve tolerance must be a float."
        raise TypeError(error_msg)
    # Value checking
    if plot_type not in [
        "mean",
        "quantile",
        "area_mean",
        "area_std_dev",
        "solve_time_quantile",
        "solve_time_cdf",
        "cdf_solvability",
        "quantile_solvability",
        "diff_cdf_solvability",
        "diff_quantile_solvability",
    ]:
        error_msg = f"Plot type '{plot_type}' is not valid."
        raise ValueError(error_msg)
    if not 0 < beta < 1:
        error_msg = "Beta quantile must be in (0, 1)."
        raise ValueError(error_msg)
    if not 0 < solve_tol <= 1:
        error_msg = "Solve tolerance must be in (0, 1]."
        raise ValueError(error_msg)

    if plot_type == "mean":
        # Single experiment --> returns a curve.
        return mean_of_curves(bootstrap_curves[0][0])
    elif plot_type == "quantile":
        # Single experiment --> returns a curve.
        return quantile_of_curves(bootstrap_curves[0][0], beta=beta)
    elif plot_type == "area_mean":
        # Single experiment --> returns a scalar.
        area_mean = np.mean(
            [
                curve.compute_area_under_curve()
                for curve in bootstrap_curves[0][0]
            ]
        )
        return float(area_mean)
    elif plot_type == "area_std_dev":
        # Single experiment --> returns a scalar.
        area_std_dev = np.std(
            [
                curve.compute_area_under_curve()
                for curve in bootstrap_curves[0][0]
            ],
            ddof=1,
        )
        return float(area_std_dev)
    elif plot_type == "solve_time_quantile":
        # Single experiment --> returns a scalar
        solve_time_quantile = np.quantile(
            [
                curve.compute_crossing_time(threshold=solve_tol)
                for curve in bootstrap_curves[0][0]
            ],
            q=beta,
        )
        return float(solve_time_quantile)
    elif plot_type == "solve_time_cdf":
        # Single experiment --> returns a curve.
        return cdf_of_curves_crossing_times(
            bootstrap_curves[0][0], threshold=solve_tol
        )
    elif plot_type == "cdf_solvability":
        # One solver, multiple problems --> returns a curve.
        return mean_of_curves(
            [
                cdf_of_curves_crossing_times(
                    curves=progress_curves, threshold=solve_tol
                )
                for progress_curves in bootstrap_curves[0]
            ]
        )
    elif plot_type == "quantile_solvability":
        # One solver, multiple problems --> returns a curve.
        return mean_of_curves(
            [
                quantile_cross_jump(
                    curves=progress_curves, threshold=solve_tol, beta=beta
                )
                for progress_curves in bootstrap_curves[0]
            ]
        )
    elif plot_type == "diff_cdf_solvability":
        # Two solvers, multiple problems --> returns a curve.
        solvability_profile_1 = mean_of_curves(
            [
                cdf_of_curves_crossing_times(
                    curves=progress_curves, threshold=solve_tol
                )
                for progress_curves in bootstrap_curves[0]
            ]
        )
        solvability_profile_2 = mean_of_curves(
            [
                cdf_of_curves_crossing_times(
                    curves=progress_curves, threshold=solve_tol
                )
                for progress_curves in bootstrap_curves[1]
            ]
        )
        return difference_of_curves(
            solvability_profile_1, solvability_profile_2
        )
    elif plot_type == "diff_quantile_solvability":
        # Two solvers, multiple problems --> returns a curve.
        solvability_profile_1 = mean_of_curves(
            [
                quantile_cross_jump(
                    curves=progress_curves, threshold=solve_tol, beta=beta
                )
                for progress_curves in bootstrap_curves[0]
            ]
        )
        solvability_profile_2 = mean_of_curves(
            [
                quantile_cross_jump(
                    curves=progress_curves, threshold=solve_tol, beta=beta
                )
                for progress_curves in bootstrap_curves[1]
            ]
        )
        return difference_of_curves(
            solvability_profile_1, solvability_profile_2
        )
    else:
        error_msg = "'{plot_type}' is not implemented."
        raise NotImplementedError(error_msg)


# TODO: double check observations type and return type
def compute_bootstrap_conf_int(
    observations: list[float | int],
    conf_level: float,
    bias_correction: bool = True,
    overall_estimator: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a bootstrap confidence interval for an estimator.

    Parameters
    ----------
    observations : list[float | int]
        Estimators from all bootstrap instances.
    conf_level : float
        Confidence level for confidence intervals, i.e., 1-gamma; in (0, 1).
    bias_correction : bool, default=True
        True if bias-corrected bootstrap CIs (via percentile method) are to be used,
        otherwise False.
    overall_estimator : float, optional
        Estimator to compute bootstrap confidence interval of;
        required for bias corrected CI.

    Returns
    -------
    ndarray[float]
        Lower bound of bootstrap CI.
    ndarray[float]
        Upper bound of bootstrap CI.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if not isinstance(observations, list):
        error_msg = "Observations must be a list."
        raise TypeError(error_msg)
    if not isinstance(conf_level, (int, float)):
        error_msg = "Confidence level must be a float."
        raise TypeError(error_msg)
    if not isinstance(bias_correction, bool):
        error_msg = "Bias correction must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(overall_estimator, (int, float, type(None))):
        error_msg = "Overall estimator must be a float or None."
        raise TypeError(error_msg)
    # Value checking
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)
    if bias_correction and overall_estimator is None:
        error_msg = "Overall estimator must be provided for bias correction."
        raise ValueError(error_msg)

    # Compute bootstrapping confidence interval via percentile method.
    # See Efron (1981) "Nonparameteric Standard Errors and Confidence Intervals."
    if bias_correction:
        if overall_estimator is None:
            error_msg = (
                "Overall estimator must be provided for bias correction."
            )
            raise ValueError(error_msg)
        # For biased-corrected CIs, see equation (4.4) on page 146.
        z0 = norm.ppf(
            np.mean([obs < overall_estimator for obs in observations])
        )
        zconflvl = norm.ppf(conf_level)
        q_lower = norm.cdf(2 * z0 - zconflvl)
        q_upper = norm.cdf(2 * z0 + zconflvl)
    else:
        # For uncorrected CIs, see equation (4.3) on page 146.
        q_lower = (1 - conf_level) / 2
        q_upper = 1 - (1 - conf_level) / 2
    bs_conf_int_lower_bound = np.quantile(observations, q=q_lower)
    bs_conf_int_upper_bound = np.quantile(observations, q=q_upper)
    # Sometimes quantile returns a scalar, so convert to array.
    if not isinstance(bs_conf_int_lower_bound, np.ndarray):
        bs_conf_int_lower_bound = np.array([bs_conf_int_lower_bound])
    if not isinstance(bs_conf_int_upper_bound, np.ndarray):
        bs_conf_int_upper_bound = np.array([bs_conf_int_upper_bound])
    return bs_conf_int_lower_bound, bs_conf_int_upper_bound


def plot_bootstrap_conf_ints(
    bs_conf_int_lower_bounds: Curve,
    bs_conf_int_upper_bounds: Curve,
    color_str: str = "C0",
) -> None:
    """Plot bootstrap confidence intervals.

    Parameters
    ----------
    bs_conf_int_lower_bounds : ``experiment_base.Curve``
        Lower bounds of bootstrap CIs, as curves.
    bs_conf_int_upper_bounds : ``experiment_base.Curve``
        Upper bounds of bootstrap CIs, as curves.
    color_str : str, default="C0"
        String indicating line color, e.g., "C0", "C1", etc.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(bs_conf_int_lower_bounds, Curve):
        error_msg = "Lower bounds must be a Curve object."
        raise TypeError(error_msg)
    if not isinstance(bs_conf_int_upper_bounds, Curve):
        error_msg = "Upper bounds must be a Curve object."
        raise TypeError(error_msg)
    if not isinstance(color_str, str):
        error_msg = "Color string must be a string."
        raise TypeError(error_msg)

    bs_conf_int_lower_bounds.plot(color_str=color_str, curve_type="conf_bound")
    bs_conf_int_upper_bounds.plot(color_str=color_str, curve_type="conf_bound")
    # Shade space between curves.
    # Convert to full curves to get piecewise-constant shaded areas.
    plt.fill_between(
        x=bs_conf_int_lower_bounds.curve_to_full_curve().x_vals,
        y1=bs_conf_int_lower_bounds.curve_to_full_curve().y_vals,
        y2=bs_conf_int_upper_bounds.curve_to_full_curve().y_vals,
        color=color_str,
        alpha=0.2,
    )


def report_max_halfwidth(
    curve_pairs: list[list[Curve]],
    normalize: bool,
    conf_level: float,
    difference: bool = False,
) -> None:
    """Compute and print caption for max halfwidth of one or more bootstrap CI curves.

    Parameters
    ----------
    curve_pairs : list [list [``experiment_base.Curve``]]
        List of paired bootstrap CI curves.
    normalize : bool
        True if progress curves are to be normalized w.r.t. optimality gaps,
        otherwise False.
    conf_level : float
        Confidence level for confidence intervals, i.e., 1-gamma; in (0, 1).
    difference : bool
        True if the plot is for difference profiles, otherwise False.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if (
        not isinstance(curve_pairs, list)
        or not all([isinstance(curve_pair, list) for curve_pair in curve_pairs])
        or not all(
            [isinstance(curve, Curve) for curve in curve_pair]
            for curve_pair in curve_pairs
        )
    ):
        error_msg = "Curve pairs must be a list of lists of Curve objects."
        raise TypeError(error_msg)
    if not isinstance(normalize, bool):
        error_msg = "Normalize must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(conf_level, (int, float)):
        error_msg = "Confidence level must be a float."
        raise TypeError(error_msg)
    if not isinstance(difference, bool):
        error_msg = "Difference must be a boolean."
        raise TypeError(error_msg)
    # Value checking
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)
    # Make sure there's something in the list
    if len(curve_pairs) == 0:
        error_msg = "No curve pairs to report on."
        raise ValueError(error_msg)

    # Compute max halfwidth of bootstrap confidence intervals.
    min_lower_bound = np.inf
    max_upper_bound = -np.inf
    max_halfwidths = []
    for curve_pair in curve_pairs:
        min_lower_bound = min(min_lower_bound, min(curve_pair[0].y_vals))
        max_upper_bound = max(max_upper_bound, max(curve_pair[1].y_vals))
        max_halfwidths.append(
            0.5 * max_difference_of_curves(curve_pair[1], curve_pair[0])
        )
    max_halfwidth = max(max_halfwidths)
    # Print caption about max halfwidth.
    if normalize:
        if difference:
            xloc = 0.05
            yloc = -1.35
        else:
            xloc = 0.05
            yloc = -0.35
    else:
        # xloc = 0.05 * budget of the problem
        xloc = 0.05 * curve_pairs[0][0].x_vals[-1]
        yloc = min_lower_bound - 0.25 * (max_upper_bound - min_lower_bound)
    txt = f"The max halfwidth of the bootstrap {round(conf_level * 100)}% CIs is {round(max_halfwidth, 2)}."
    plt.text(x=xloc, y=yloc, s=txt)


def check_common_problem_and_reference(
    experiments: list[ProblemSolver],
) -> None:
    """Check if a collection of experiments have the same problem, x0, and x*.

    Parameters
    ----------
    experiments : list [``experiment_base.ProblemSolver``]
        Problem-solver pairs of different solvers on a common problem.

    Raises
    ------
    TypeError
    ValueError
        If at least two experiments have different problem instances, starting solutions, or optimal solutions.

    """
    # Type checking
    if not isinstance(experiments, list) or not all(
        [isinstance(experiment, ProblemSolver) for experiment in experiments]
    ):
        error_msg = "Experiments must be a list of ProblemSolver objects."
        raise TypeError(error_msg)

    problem_list = [experiment.problem for experiment in experiments]
    problem_err_msg = "All experiments must have the same problem."
    assert all(
        elem == problem_list[0] for elem in problem_list
    ), problem_err_msg

    x0_list = [experiment.x0 for experiment in experiments]
    x0_err_msg = "All experiments must have the same starting solution."
    assert all(elem == x0_list[0] for elem in x0_list), x0_err_msg

    xstar_list = [experiment.xstar for experiment in experiments]
    xstar_err_msg = "All experiments must have the same optimal solution."
    assert all(elem == xstar_list[0] for elem in xstar_list), xstar_err_msg


def plot_progress_curves(
    experiments: list[ProblemSolver],
    plot_type: Literal["all", "mean", "quantile"],
    beta: float = 0.50,
    normalize: bool = True,
    all_in_one: bool = True,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    print_max_hw: bool = True,
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
) -> list[str | os.PathLike]:
    """Plot individual or aggregate progress curves for one or more solvers on a single problem.

    Parameters
    ----------
    experiments : list [``experiment_base.ProblemSolver``]
        Problem-solver pairs of different solvers on a common problem.
    plot_type : str
        String indicating which type of plot to produce:
            "all" : all estimated progress curves
            "mean" : estimated mean progress curve
            "quantile" : estimated beta quantile progress curve.
    beta : float, default=0.50
        Quantile to plot, e.g., beta quantile; in (0, 1).
    normalize : bool, default=True
        True if progress curves are to be normalized w.r.t. optimality gaps,
        otherwise False.
    all_in_one : bool, default=True
        True if curves are to be plotted together, otherwise False.
    n_bootstraps : int, default = 100
        Number of bootstrap samples.
    conf_level : float, default = 0.95
        Confidence level for confidence intervals, i.e., 1-gamma; in (0, 1).
    plot_conf_ints : bool, default=True
        True if bootstrapping confidence intervals are to be plotted, otherwise False.
    print_max_hw : bool, default=True
        True if caption with max half-width is to be printed, otherwise False.
    plot_title : str, opt
        Optional title to override the one that is autmatically generated, only applies if all_in_one is True.
    legend_loc : str, default="best"
        specificies location of legend
    ext: str, default = '.png'
        Extension to add to image file path to change file type
    save_as_pickle: bool, default = False
        True if plot should be saved to pickle file, False otherwise.
    solver_set_name: str, default = "SOLVER_SET"
        Use to change name of solver groups for plot titles.

    Returns
    -------
    file_list : list [str]
        List compiling path names for plots produced.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if not isinstance(experiments, list) or not all(
        [isinstance(experiment, ProblemSolver) for experiment in experiments]
    ):
        error_msg = "Experiments must be a list of ProblemSolver objects."
        raise TypeError(error_msg)
    if not isinstance(plot_type, str):
        error_msg = "Plot type must be a string."
        raise TypeError(error_msg)
    if not isinstance(beta, (int, float)):
        error_msg = "Beta quantile must be a float."
        raise TypeError(error_msg)
    if not isinstance(normalize, bool):
        error_msg = "Normalize must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(all_in_one, bool):
        error_msg = "All in one must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(n_bootstraps, int):
        error_msg = "Number of bootstraps must be an integer."
        raise TypeError(error_msg)
    if not isinstance(conf_level, (int, float)):
        error_msg = "Confidence level must be a float."
        raise TypeError(error_msg)
    if not isinstance(plot_conf_ints, bool):
        error_msg = "Plot confidence intervals must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(print_max_hw, bool):
        error_msg = "Print max halfwidth must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(plot_title, (str, type(None))):
        error_msg = "Plot title must be a string or None."
        raise TypeError(error_msg)
    if not isinstance(legend_loc, (str, type(None))):
        error_msg = "Legend location must be a string or None."
        raise TypeError(error_msg)
    if not isinstance(ext, str):
        error_msg = "Extension must be a string."
        raise TypeError(error_msg)
    if not isinstance(save_as_pickle, bool):
        error_msg = "Save as pickle must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(solver_set_name, str):
        error_msg = "Solver set name must be a string."
        raise TypeError(error_msg)
    # Value checking
    if plot_type not in ["all", "mean", "quantile"]:
        error_msg = f"Plot type '{plot_type}' is not valid."
        raise ValueError(error_msg)
    if not 0 < beta < 1:
        error_msg = "Beta quantile must be in (0, 1)."
        raise ValueError(error_msg)
    if n_bootstraps < 1:
        error_msg = "Number of bootstraps must be a positive integer."
        raise ValueError(error_msg)
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)

    if legend_loc is None:
        legend_loc = "best"

    # Check if problems are the same with the same x0 and x*.
    check_common_problem_and_reference(experiments)
    file_list = []
    # Set up plot.
    n_experiments = len(experiments)
    if all_in_one:
        ref_experiment = experiments[0]
        setup_plot(
            plot_type=plot_type,
            solver_name=solver_set_name,
            problem_name=ref_experiment.problem.name,
            normalize=normalize,
            budget=ref_experiment.problem.factors["budget"],
            beta=beta,
            plot_title=plot_title,
        )
        solver_curve_handles = []
        curve_pairs = []
        for exp_idx in range(n_experiments):
            experiment = experiments[exp_idx]
            color_str = "C" + str(exp_idx)
            estimator = None
            if plot_type == "all":
                # Plot all estimated progress curves.
                if normalize:
                    handle = experiment.progress_curves[0].plot(
                        color_str=color_str
                    )
                    for curve in experiment.progress_curves[1:]:
                        curve.plot(color_str=color_str)
                else:
                    handle = experiment.objective_curves[0].plot(
                        color_str=color_str
                    )
                    for curve in experiment.objective_curves[1:]:
                        curve.plot(color_str=color_str)
            elif plot_type == "mean":
                # Plot estimated mean progress curve.
                if normalize:
                    estimator = mean_of_curves(experiment.progress_curves)
                else:
                    estimator = mean_of_curves(experiment.objective_curves)
                handle = estimator.plot(color_str=color_str)
            else:  # Must be quantile.
                # Plot estimated beta-quantile progress curve.
                if normalize:
                    estimator = quantile_of_curves(
                        experiment.progress_curves, beta
                    )
                else:
                    estimator = quantile_of_curves(
                        experiment.objective_curves, beta
                    )
                handle = estimator.plot(color_str=color_str)
            solver_curve_handles.append(handle)
            if (plot_conf_ints or print_max_hw) and plot_type != "all":
                # Note: "experiments" needs to be a list of list of ProblemSolver objects.
                bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                    bootstrap_procedure(
                        experiments=[[experiment]],
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_type=plot_type,
                        beta=beta,
                        estimator=estimator,
                        normalize=normalize,
                    )
                )
                if plot_conf_ints:
                    if isinstance(
                        bs_conf_int_lb_curve, (int, float)
                    ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                        error_msg = "Bootstrap confidence intervals are not available for scalar estimators."
                        raise ValueError(error_msg)
                    plot_bootstrap_conf_ints(
                        bs_conf_int_lb_curve,
                        bs_conf_int_ub_curve,
                        color_str=color_str,
                    )
                if print_max_hw:
                    curve_pairs.append(
                        [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                    )
        plt.legend(
            handles=solver_curve_handles,
            labels=[experiment.solver.name for experiment in experiments],
            loc=legend_loc,
        )
        if print_max_hw and plot_type != "all":
            report_max_halfwidth(
                curve_pairs=curve_pairs,
                normalize=normalize,
                conf_level=conf_level,
            )
        file_list.append(
            save_plot(
                solver_name=solver_set_name,
                problem_name=ref_experiment.problem.name,
                plot_type=plot_type,
                normalize=normalize,
                extra=beta,
                plot_title=plot_title,
                ext=ext,
                save_as_pickle=save_as_pickle,
            )
        )
    else:  # Plot separately.
        for experiment in experiments:
            setup_plot(
                plot_type=plot_type,
                solver_name=experiment.solver.name,
                problem_name=experiment.problem.name,
                normalize=normalize,
                budget=experiment.problem.factors["budget"],
                beta=beta,
            )
            estimator = None
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
            else:  # Must be quantile.
                # Plot estimated beta-quantile progress curve.
                if normalize:
                    estimator = quantile_of_curves(
                        experiment.progress_curves, beta
                    )
                else:
                    estimator = quantile_of_curves(
                        experiment.objective_curves, beta
                    )
                estimator.plot()
            if (plot_conf_ints or print_max_hw) and plot_type != "all":
                # Note: "experiments" needs to be a list of list of ProblemSolvers.
                bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                    bootstrap_procedure(
                        experiments=[[experiment]],
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_type=plot_type,
                        beta=beta,
                        estimator=estimator,
                        normalize=normalize,
                    )
                )
                if plot_conf_ints:
                    if isinstance(
                        bs_conf_int_lb_curve, (int, float)
                    ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                        error_msg = "Bootstrap confidence intervals are not available for scalar estimators."
                        raise ValueError(error_msg)
                    plot_bootstrap_conf_ints(
                        bs_conf_int_lb_curve, bs_conf_int_ub_curve
                    )
                if print_max_hw:
                    if isinstance(
                        bs_conf_int_lb_curve, (int, float)
                    ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                        error_msg = "Max halfwidth is not available for scalar estimators."
                        raise ValueError(error_msg)
                    report_max_halfwidth(
                        curve_pairs=[
                            [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                        ],
                        normalize=normalize,
                        conf_level=conf_level,
                    )
            file_list.append(
                save_plot(
                    solver_name=experiment.solver.name,
                    problem_name=experiment.problem.name,
                    plot_type=plot_type,
                    normalize=normalize,
                    extra=beta,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )
    return file_list


def plot_solvability_cdfs(
    experiments: list[ProblemSolver],
    solve_tol: float = 0.1,
    all_in_one: bool = True,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    print_max_hw: bool = True,
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
) -> list[str | os.PathLike]:
    """Plot the solvability cdf for one or more solvers on a single problem.

    Parameters
    ----------
    experiments : list [``experiment_base.ProblemSolver``]
        Problem-solver pairs of different solvers on a common problem.
    solve_tol : float, default=0.1
        Relative optimality gap definining when a problem is solved; in (0, 1].
    all_in_one : bool, default=True
        True if curves are to be plotted together, otherwise False.
    n_bootstraps : int, default=100
        Number of bootstrap samples.
    conf_level : float, default=0.95
        Confidence level for confidence intervals, i.e., 1-gamma; in (0, 1).
    plot_conf_ints : bool, default=True
        True if bootstrapping confidence intervals are to be plotted, otherwise False.
    print_max_hw : bool, default=True
        True if caption with max half-width is to be printed, otherwise False.
    plot_title : str, opt
        Optional title to override the one that is autmatically generated, only applies if all_in_one is True.
    legend_loc : str, default="best"
        specificies location of legend
    ext: str, default = '.png'
        Extension to add to image file path to change file type
    save_as_pickle: bool, default = False
        True if plot should be saved to pickle file, False otherwise.
    solver_set_name: str, default = "SOLVER_SET"
        Use to change name of solver groups for plot titles.

    Returns
    -------
    file_list : list [str | os.PathLike]
        List compiling path names for plots produced.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if not isinstance(experiments, list) or not all(
        [isinstance(experiment, ProblemSolver) for experiment in experiments]
    ):
        error_msg = "Experiments must be a list of ProblemSolver objects."
        raise TypeError(error_msg)
    if not isinstance(solve_tol, (int, float)):
        error_msg = "Solve tolerance must be a float."
        raise TypeError(error_msg)
    if not isinstance(all_in_one, bool):
        error_msg = "All in one must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(n_bootstraps, int):
        error_msg = "Number of bootstraps must be an integer."
        raise TypeError(error_msg)
    if not isinstance(conf_level, (int, float)):
        error_msg = "Confidence level must be a float."
        raise TypeError(error_msg)
    if not isinstance(plot_conf_ints, bool):
        error_msg = "Plot confidence intervals must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(print_max_hw, bool):
        error_msg = "Print max halfwidth must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(plot_title, (str, type(None))):
        error_msg = "Plot title must be a string or None."
        raise TypeError(error_msg)
    if not isinstance(legend_loc, (str, type(None))):
        error_msg = "Legend location must be a string or None."
        raise TypeError(error_msg)
    if not isinstance(ext, str):
        error_msg = "Extension must be a string."
        raise TypeError(error_msg)
    if not isinstance(save_as_pickle, bool):
        error_msg = "Save as pickle must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(solver_set_name, str):
        error_msg = "Solver set name must be a string."
        raise TypeError(error_msg)
    # Value checking
    if not 0 < solve_tol <= 1:
        error_msg = "Solve tolerance must be in (0, 1]."
        raise ValueError(error_msg)
    if n_bootstraps < 1:
        error_msg = "Number of bootstraps must be a positive integer."
        raise ValueError(error_msg)
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)

    if legend_loc is None:
        legend_loc = "best"

    # Check if problems are the same with the same x0 and x*.
    check_common_problem_and_reference(experiments)
    file_list = []
    # Set up plot.
    n_experiments = len(experiments)
    if all_in_one:
        ref_experiment = experiments[0]
        setup_plot(
            plot_type="solve_time_cdf",
            solver_name=solver_set_name,
            problem_name=ref_experiment.problem.name,
            solve_tol=solve_tol,
            plot_title=plot_title,
        )
        solver_curve_handles = []
        curve_pairs = []
        for exp_idx in range(n_experiments):
            experiment = experiments[exp_idx]
            color_str = "C" + str(exp_idx)
            # Plot cdf of solve times.
            estimator = cdf_of_curves_crossing_times(
                experiment.progress_curves, threshold=solve_tol
            )
            handle = estimator.plot(color_str=color_str)
            solver_curve_handles.append(handle)
            if plot_conf_ints or print_max_hw:
                # Note: "experiments" needs to be a list of list of ProblemSolver objects.
                bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                    bootstrap_procedure(
                        experiments=[[experiment]],
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_type="solve_time_cdf",
                        solve_tol=solve_tol,
                        estimator=estimator,
                        normalize=True,
                    )
                )
                if plot_conf_ints:
                    if isinstance(
                        bs_conf_int_lb_curve, (int, float)
                    ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                        error_msg = "Bootstrap confidence intervals are not available for scalar estimators."
                        raise ValueError(error_msg)
                    plot_bootstrap_conf_ints(
                        bs_conf_int_lb_curve,
                        bs_conf_int_ub_curve,
                        color_str=color_str,
                    )
                if print_max_hw:
                    curve_pairs.append(
                        [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                    )
        plt.legend(
            handles=solver_curve_handles,
            labels=[experiment.solver.name for experiment in experiments],
            loc=legend_loc,
        )
        if print_max_hw:
            report_max_halfwidth(
                curve_pairs=curve_pairs, normalize=True, conf_level=conf_level
            )
        file_list.append(
            save_plot(
                solver_name="SOLVER SET",
                problem_name=ref_experiment.problem.name,
                plot_type="solve_time_cdf",
                normalize=True,
                extra=solve_tol,
                plot_title=plot_title,
                ext=ext,
                save_as_pickle=save_as_pickle,
            )
        )
    else:  # Plot separately.
        for experiment in experiments:
            setup_plot(
                plot_type="solve_time_cdf",
                solver_name=experiment.solver.name,
                problem_name=experiment.problem.name,
                solve_tol=solve_tol,
            )
            estimator = cdf_of_curves_crossing_times(
                experiment.progress_curves, threshold=solve_tol
            )
            estimator.plot()
            if plot_conf_ints or print_max_hw:
                # Note: "experiments" needs to be a list of list of Problem-Solver objects.
                bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                    bootstrap_procedure(
                        experiments=[[experiment]],
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_type="solve_time_cdf",
                        solve_tol=solve_tol,
                        estimator=estimator,
                        normalize=True,
                    )
                )
                if plot_conf_ints:
                    if isinstance(
                        bs_conf_int_lb_curve, (int, float)
                    ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                        error_msg = "Bootstrap confidence intervals are not available for scalar estimators."
                        raise ValueError(error_msg)
                    plot_bootstrap_conf_ints(
                        bs_conf_int_lb_curve, bs_conf_int_ub_curve
                    )
                if print_max_hw:
                    if isinstance(
                        bs_conf_int_lb_curve, (int, float)
                    ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                        error_msg = "Max halfwidth is not available for scalar estimators."
                        raise ValueError(error_msg)
                    report_max_halfwidth(
                        curve_pairs=[
                            [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                        ],
                        normalize=True,
                        conf_level=conf_level,
                    )
            file_list.append(
                save_plot(
                    solver_name=experiment.solver.name,
                    problem_name=experiment.problem.name,
                    plot_type="solve_time_cdf",
                    normalize=True,
                    extra=solve_tol,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )
    return file_list


# TODO: Add the capability to compute and print the max halfwidth of the bootstrapped CI intervals.
def plot_area_scatterplots(
    experiments: list[
        list[ProblemSolver]
    ],  # TODO: check if this should be list[ProblemSolver]
    all_in_one: bool = True,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    print_max_hw: bool = True,
    plot_title: str | None = None,
    legend_loc: str = "best",
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
    problem_set_name: str = "PROBLEM_SET",
) -> list[str | os.PathLike]:
    """Plot a scatter plot of mean and standard deviation of area under progress curves.

    Either one plot for each solver or one plot for all solvers.

    Notes
    -----
    TODO: Add the capability to compute and print the max halfwidth of
    the bootstrapped CI intervals.

    Parameters
    ----------
    experiments : list [list [``experiment_base.ProblemSolver``]]
        Problem-solver pairs used to produce plots.
    all_in_one : bool, default=True
        True if curves are to be plotted together, otherwise False.
    n_bootstraps : int, default=100
        Number of bootstrap samples.
    conf_level : float
        Confidence level for confidence intervals, i.e., 1-gamma; in (0, 1).
    plot_conf_ints : bool, default=True
        True if bootstrapping confidence intervals are to be plotted, otherwise False.
    print_max_hw : bool, default=True
        True if caption with max half-width is to be printed, otherwise False.
    plot_title : str, opt
        Optional title to override the one that is autmatically generated, only applies if all_in_one is True.
    legend_loc : str, default="best"
        specificies location of legend
    ext: str, default = '.png'
        Extension to add to image file path to change file type
    save_as_pickle: bool, default = False
        True if plot should be saved to pickle file, False otherwise.
    solver_set_name: str, default = "SOLVER_SET"
        Use to change name of solver groups for plot titles.
    problem_set_name: str, default = "PROBLEM_SET"
        USe to change name of problem groups for plot titles.

    Returns
    -------
    file_list : list [str]
        List compiling path names for plots produced.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if not isinstance(experiments, list) or not all(
        [isinstance(experiment, list) for experiment in experiments]
    ):
        error_msg = (
            "Experiments must be a list of lists of ProblemSolver objects."
        )
        raise TypeError(error_msg)
    if not isinstance(all_in_one, bool):
        error_msg = "All in one must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(n_bootstraps, int):
        error_msg = "Number of bootstraps must be an integer."
        raise TypeError(error_msg)
    if not isinstance(conf_level, (int, float)):
        error_msg = "Confidence level must be a float."
        raise TypeError(error_msg)
    if not isinstance(plot_conf_ints, bool):
        error_msg = "Plot confidence intervals must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(print_max_hw, bool):
        error_msg = "Print max halfwidth must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(plot_title, (str, type(None))):
        error_msg = "Plot title must be a string or None."
        raise TypeError(error_msg)
    if not isinstance(legend_loc, str):
        error_msg = "Legend location must be a string."
        raise TypeError(error_msg)
    if not isinstance(ext, str):
        error_msg = "Extension must be a string."
        raise TypeError(error_msg)
    if not isinstance(save_as_pickle, bool):
        error_msg = "Save as pickle must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(solver_set_name, str):
        error_msg = "Solver set name must be a string."
        raise TypeError(error_msg)
    if not isinstance(problem_set_name, str):
        error_msg = "Problem set name must be a string."
        raise TypeError(error_msg)
    # Value checking
    if n_bootstraps < 1:
        error_msg = "Number of bootstraps must be a positive integer."
        raise ValueError(error_msg)
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)

    file_list = []
    # Set up plot.
    n_solvers = len(experiments)
    n_problems = len(experiments[0])
    if all_in_one:
        marker_list = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]
        setup_plot(
            plot_type="area",
            solver_name=solver_set_name,
            problem_name=problem_set_name,
            plot_title=plot_title,
        )
        solver_names = [
            solver_experiments[0].solver.name
            for solver_experiments in experiments
        ]
        solver_curve_handles = []
        # TODO: Build up capability to print max half-width.
        # if print_max_hw:
        #     curve_pairs = []
        handle = None
        for solver_idx in range(n_solvers):
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                color_str = "C" + str(solver_idx)
                marker_str = marker_list[
                    solver_idx % len(marker_list)
                ]  # Cycle through list of marker types.
                # Plot mean and standard deviation of area under progress curve.
                areas = [
                    curve.compute_area_under_curve()
                    for curve in experiment.progress_curves
                ]
                mean_estimator = float(np.mean(areas))
                std_dev_estimator = float(np.std(areas, ddof=1))
                if plot_conf_ints:
                    # Note: "experiments" needs to be a list of list of ProblemSolver objects.
                    mean_bs_conf_int_lb, mean_bs_conf_int_ub = (
                        bootstrap_procedure(
                            experiments=[[experiment]],
                            n_bootstraps=n_bootstraps,
                            conf_level=conf_level,
                            plot_type="area_mean",
                            estimator=mean_estimator,
                            normalize=True,
                        )
                    )
                    std_dev_bs_conf_int_lb, std_dev_bs_conf_int_ub = (
                        bootstrap_procedure(
                            experiments=[[experiment]],
                            n_bootstraps=n_bootstraps,
                            conf_level=conf_level,
                            plot_type="area_std_dev",
                            estimator=std_dev_estimator,
                            normalize=True,
                        )
                    )
                    # if print_max_hw:
                    #     curve_pairs.append([bs_CI_lb_curve, bs_CI_ub_curve])
                    if isinstance(mean_bs_conf_int_lb, (Curve)) or isinstance(
                        mean_bs_conf_int_ub, (Curve)
                    ):
                        error_msg = (
                            "Mean confidence intervals should be scalar values."
                        )
                        raise ValueError(error_msg)
                    if isinstance(
                        std_dev_bs_conf_int_lb, (Curve)
                    ) or isinstance(std_dev_bs_conf_int_ub, (Curve)):
                        error_msg = "Standard deviation confidence intervals should be scalar values."
                        raise ValueError(error_msg)
                    x_err = [
                        [mean_estimator - mean_bs_conf_int_lb],
                        [mean_bs_conf_int_ub - mean_estimator],
                    ]
                    y_err = [
                        [std_dev_estimator - std_dev_bs_conf_int_lb],
                        [std_dev_bs_conf_int_ub - std_dev_estimator],
                    ]
                    handle = plt.errorbar(
                        x=mean_estimator,
                        y=std_dev_estimator,
                        xerr=x_err,
                        yerr=y_err,
                        color=color_str,
                        marker=marker_str,
                        elinewidth=1,
                    )
                else:
                    handle = plt.scatter(
                        x=mean_estimator,
                        y=std_dev_estimator,
                        color=color_str,
                        marker=marker_str,
                    )
            solver_curve_handles.append(handle)
        plt.legend(
            handles=solver_curve_handles, labels=solver_names, loc=legend_loc
        )
        file_list.append(
            save_plot(
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                plot_type="area",
                normalize=True,
                plot_title=plot_title,
                ext=ext,
                save_as_pickle=save_as_pickle,
            )
        )
    else:
        for solver_idx in range(n_solvers):
            ref_experiment = experiments[solver_idx][0]
            setup_plot(
                plot_type="area",
                solver_name=ref_experiment.solver.name,
                problem_name=problem_set_name,
            )
            # if print_max_hw:
            #     curve_pairs = []
            experiment = None
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                # Plot mean and standard deviation of area under progress curve.
                areas = [
                    curve.compute_area_under_curve()
                    for curve in experiment.progress_curves
                ]
                mean_estimator = float(np.mean(areas))
                std_dev_estimator = float(np.std(areas, ddof=1))
                if plot_conf_ints:
                    # Note: "experiments" needs to be a list of list of ProblemSolver objects.
                    mean_bs_conf_int_lb, mean_bs_conf_int_ub = (
                        bootstrap_procedure(
                            experiments=[[experiment]],
                            n_bootstraps=n_bootstraps,
                            conf_level=conf_level,
                            plot_type="area_mean",
                            estimator=mean_estimator,
                            normalize=True,
                        )
                    )
                    std_dev_bs_conf_int_lb, std_dev_bs_conf_int_ub = (
                        bootstrap_procedure(
                            experiments=[[experiment]],
                            n_bootstraps=n_bootstraps,
                            conf_level=conf_level,
                            plot_type="area_std_dev",
                            estimator=std_dev_estimator,
                            normalize=True,
                        )
                    )
                    # if print_max_hw:
                    #     curve_pairs.append([bs_CI_lb_curve, bs_CI_ub_curve])
                    if isinstance(mean_bs_conf_int_lb, (Curve)) or isinstance(
                        mean_bs_conf_int_ub, (Curve)
                    ):
                        error_msg = (
                            "Mean confidence intervals should be scalar values."
                        )
                        raise ValueError(error_msg)
                    if isinstance(
                        std_dev_bs_conf_int_lb, (Curve)
                    ) or isinstance(std_dev_bs_conf_int_ub, (Curve)):
                        error_msg = "Standard deviation confidence intervals should be scalar values."
                        raise ValueError(error_msg)
                    x_err = [
                        [mean_estimator - mean_bs_conf_int_lb],
                        [mean_bs_conf_int_ub - mean_estimator],
                    ]
                    y_err = [
                        [std_dev_estimator - std_dev_bs_conf_int_lb],
                        [std_dev_bs_conf_int_ub - std_dev_estimator],
                    ]
                    handle = plt.errorbar(
                        x=mean_estimator,
                        y=std_dev_estimator,
                        xerr=x_err,
                        yerr=y_err,
                        marker="o",
                        color="C0",
                        elinewidth=1,
                    )
                else:
                    handle = plt.scatter(
                        x=mean_estimator,
                        y=std_dev_estimator,
                        color="C0",
                        marker="o",
                    )
            if experiment is not None:
                file_list.append(
                    save_plot(
                        solver_name=experiment.solver.name,
                        problem_name=problem_set_name,
                        plot_type="area",
                        normalize=True,
                        ext=ext,
                        save_as_pickle=save_as_pickle,
                    )
                )
    return file_list


def plot_solvability_profiles(
    experiments: list[
        list[ProblemSolver]
    ],  # TODO: check if this should be list[ProblemSolver]
    plot_type: Literal[
        "cdf_solvability",
        "quantile_solvability",
        "diff_cdf_solvability",
        "diff_quantile_solvability",
    ],
    all_in_one: bool = True,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    print_max_hw: bool = True,
    solve_tol: float = 0.1,
    beta: float = 0.5,
    ref_solver: str | None = None,
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
    problem_set_name: str = "PROBLEM_SET",
) -> list[str | os.PathLike]:
    """Plot the (difference of) solvability profiles for each solver on a set of problems.

    Parameters
    ----------
    experiments : list [list [``experiment_base.ProblemSolver``]]
        Problem-solver pairs used to produce plots.
    plot_type : str
        String indicating which type of plot to produce:
            "cdf_solvability" : cdf-solvability profile;

            "quantile_solvability" : quantile-solvability profile;

            "diff_cdf_solvability" : difference of cdf-solvability profiles;

            "diff_quantile_solvability" : difference of quantile-solvability profiles.
    all_in_one : bool, default=True
        True if curves are to be plotted together, otherwise False.
    n_bootstraps : int, default=100
        Number of bootstrap samples.
    conf_level : float
        Confidence level for confidence intervals, i.e., 1-gamma; in (0, 1).
    plot_conf_ints : bool, default=True
        True if bootstrapping confidence intervals are to be plotted, otherwise False.
    print_max_hw : bool, default=True
        True if caption with max half-width is to be printed, otherwise False.
    solve_tol : float, default=0.1
        Relative optimality gap definining when a problem is solved; in (0, 1].
    beta : float, default=0.5
        Quantile to compute, e.g., beta quantile; in (0, 1).
    ref_solver : str, optional
        Name of solver used as benchmark for difference profiles.
    plot_title : str, optional
        Optional title to override the one that is autmatically generated, only applies if all_in_one is True.
    legend_loc : str, default="best"
        specificies location of legend
    ext: str, default = '.png'
         Extension to add to image file path to change file type
    save_as_pickle: bool, default = False
         True if plot should be saved to pickle file, False otherwise.
    solver_set_name: str, default = "SOLVER_SET"
        Use to change name of solver groups for plot titles.
    problem_set_name: str, default = "PROBLEM_SET"
        USe to change name of problem groups for plot titles.

    Returns
    -------
    file_list : list [str]
        List compiling path names for plots produced.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if (
        not isinstance(experiments, list)
        or not all(
            [
                isinstance(experiment_list, list)
                for experiment_list in experiments
            ]
        )
        or not all(
            [
                isinstance(experiment, ProblemSolver)
                for experiment in experiment_list
            ]
            for experiment_list in experiments
        )
    ):
        error_msg = (
            "Experiments must be a list of lists of ProblemSolver objects."
        )
        raise TypeError(error_msg)
    if not isinstance(plot_type, str):
        error_msg = "Plot type must be a string."
        raise TypeError(error_msg)
    if not isinstance(all_in_one, bool):
        error_msg = "All in one must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(n_bootstraps, int):
        error_msg = "Number of bootstraps must be an integer."
        raise TypeError(error_msg)
    if not isinstance(conf_level, (int, float)):
        error_msg = "Confidence level must be a float."
        raise TypeError(error_msg)
    if not isinstance(plot_conf_ints, bool):
        error_msg = "Plot confidence intervals must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(print_max_hw, bool):
        error_msg = "Print max halfwidth must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(solve_tol, (int, float)):
        error_msg = "Solve tolerance must be a float."
        raise TypeError(error_msg)
    if not isinstance(beta, (int, float)):
        error_msg = "Beta quantile must be a float."
        raise TypeError(error_msg)
    if not isinstance(ref_solver, (str, type(None))):
        error_msg = "Reference solver must be a string or None."
        raise TypeError(error_msg)
    if not isinstance(plot_title, (str, type(None))):
        error_msg = "Plot title must be a string or None."
        raise TypeError(error_msg)
    if not isinstance(legend_loc, (str, type(None))):
        error_msg = "Legend location must be a string."
        raise TypeError(error_msg)
    if not isinstance(ext, str):
        error_msg = "Extension must be a string."
        raise TypeError(error_msg)
    if not isinstance(save_as_pickle, bool):
        error_msg = "Save as pickle must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(solver_set_name, str):
        error_msg = "Solver set name must be a string."
        raise TypeError(error_msg)
    if not isinstance(problem_set_name, str):
        error_msg = "Problem set name must be a string."
        raise TypeError(error_msg)
    # Value checking
    if n_bootstraps < 1:
        error_msg = "Number of bootstraps must be a positive integer."
        raise ValueError(error_msg)
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)
    if not 0 < solve_tol <= 1:
        error_msg = "Solve tolerance must be in (0, 1]."
        raise ValueError(error_msg)
    if not 0 < beta < 1:
        error_msg = "Beta quantile must be in (0, 1)."
        raise ValueError(error_msg)

    if legend_loc is None:
        legend_loc = "best"

    file_list = []
    # Set up plot.
    n_solvers = len(experiments)
    n_problems = len(experiments[0])
    if all_in_one:
        if plot_type == "cdf_solvability":
            setup_plot(
                plot_type=plot_type,
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                solve_tol=solve_tol,
                plot_title=plot_title,
            )
        elif plot_type == "quantile_solvability":
            setup_plot(
                plot_type=plot_type,
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                beta=beta,
                solve_tol=solve_tol,
                plot_title=plot_title,
            )
        elif plot_type == "diff_cdf_solvability":
            setup_plot(
                plot_type=plot_type,
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                solve_tol=solve_tol,
                plot_title=plot_title,
            )
        elif plot_type == "diff_quantile_solvability":
            setup_plot(
                plot_type=plot_type,
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                beta=beta,
                solve_tol=solve_tol,
                plot_title=plot_title,
            )
        curve_pairs = []
        solver_names = [
            solver_experiments[0].solver.name
            for solver_experiments in experiments
        ]
        solver_curves = []
        solver_curve_handles = []
        for solver_idx in range(n_solvers):
            solver_sub_curves = []
            color_str = "C" + str(solver_idx)
            # For each problem compute the cdf or quantile of solve times.
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                sub_curve = None
                if plot_type in {"cdf_solvability", "diff_cdf_solvability"}:
                    sub_curve = cdf_of_curves_crossing_times(
                        curves=experiment.progress_curves, threshold=solve_tol
                    )
                if plot_type in {
                    "quantile_solvability",
                    "diff_quantile_solvability",
                }:
                    sub_curve = quantile_cross_jump(
                        curves=experiment.progress_curves,
                        threshold=solve_tol,
                        beta=beta,
                    )
                if sub_curve is not None:
                    solver_sub_curves.append(sub_curve)
            # Plot solvability profile for the solver.
            # Exploit the fact that each solvability profile is an average of more basic curves.
            solver_curve = mean_of_curves(solver_sub_curves)
            # CAUTION: Using mean above requires an equal number of macro-replications per problem.
            solver_curves.append(solver_curve)
            if plot_type in {"cdf_solvability", "quantile_solvability"}:
                handle = solver_curve.plot(color_str=color_str)
                solver_curve_handles.append(handle)
                if plot_conf_ints or print_max_hw:
                    # Note: "experiments" needs to be a list of list of ProblemSolver objects.
                    bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                        bootstrap_procedure(
                            experiments=[experiments[solver_idx]],
                            n_bootstraps=n_bootstraps,
                            conf_level=conf_level,
                            plot_type=plot_type,  # type: ignore
                            solve_tol=solve_tol,
                            beta=beta,
                            estimator=solver_curve,
                            normalize=True,
                        )
                    )
                    if plot_conf_ints:
                        if isinstance(
                            bs_conf_int_lb_curve, (int, float)
                        ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                            error_msg = "Bootstrap confidence intervals are not available for scalar estimators."
                            raise ValueError(error_msg)
                        plot_bootstrap_conf_ints(
                            bs_conf_int_lb_curve,
                            bs_conf_int_ub_curve,
                            color_str=color_str,
                        )
                    if print_max_hw:
                        curve_pairs.append(
                            [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                        )
        if plot_type == "cdf_solvability":
            plt.legend(
                handles=solver_curve_handles,
                labels=solver_names,
                loc=legend_loc,
            )
            if print_max_hw:
                report_max_halfwidth(
                    curve_pairs=curve_pairs,
                    normalize=True,
                    conf_level=conf_level,
                )
            file_list.append(
                save_plot(
                    solver_name=solver_set_name,
                    problem_name=problem_set_name,
                    plot_type=plot_type,
                    normalize=True,
                    extra=solve_tol,
                    plot_title=plot_title,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )
        elif plot_type == "quantile_solvability":
            plt.legend(
                handles=solver_curve_handles,
                labels=solver_names,
                loc=legend_loc,
            )
            if print_max_hw:
                report_max_halfwidth(
                    curve_pairs=curve_pairs,
                    normalize=True,
                    conf_level=conf_level,
                )
            file_list.append(
                save_plot(
                    solver_name=solver_set_name,
                    problem_name=problem_set_name,
                    plot_type=plot_type,
                    normalize=True,
                    extra=[solve_tol, beta],
                    plot_title=plot_title,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )
        elif plot_type in {"diff_cdf_solvability", "diff_quantile_solvability"}:
            if ref_solver is None:
                error_msg = "Reference solver must be specified for difference profiles."
                raise ValueError(error_msg)
            non_ref_solvers = [
                solver_name
                for solver_name in solver_names
                if solver_name != ref_solver
            ]
            ref_solver_idx = solver_names.index(ref_solver)
            for solver_idx in range(n_solvers):
                if solver_idx is not ref_solver_idx:
                    diff_solver_curve = difference_of_curves(
                        solver_curves[solver_idx], solver_curves[ref_solver_idx]
                    )
                    color_str = "C" + str(solver_idx)
                    handle = diff_solver_curve.plot(color_str=color_str)
                    solver_curve_handles.append(handle)
                    if plot_conf_ints or print_max_hw:
                        # Note: "experiments" needs to be a list of list of ProblemSolver objects.
                        bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                            bootstrap_procedure(
                                experiments=[
                                    experiments[solver_idx],
                                    experiments[ref_solver_idx],
                                ],
                                n_bootstraps=n_bootstraps,
                                conf_level=conf_level,
                                plot_type=plot_type,  # type: ignore
                                solve_tol=solve_tol,
                                beta=beta,
                                estimator=diff_solver_curve,
                                normalize=True,
                            )
                        )
                        if plot_conf_ints:
                            if isinstance(
                                bs_conf_int_lb_curve, (int, float)
                            ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                                error_msg = "Bootstrap confidence intervals are not available for scalar estimators."
                                raise ValueError(error_msg)
                            plot_bootstrap_conf_ints(
                                bs_conf_int_lb_curve,
                                bs_conf_int_ub_curve,
                                color_str=color_str,
                            )
                        if print_max_hw:
                            curve_pairs.append(
                                [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                            )
            offset_labels = [
                f"{non_ref_solver} - {ref_solver}"
                for non_ref_solver in non_ref_solvers
            ]
            plt.legend(
                handles=solver_curve_handles,
                labels=offset_labels,
                loc=legend_loc,
            )
            if print_max_hw:
                report_max_halfwidth(
                    curve_pairs=curve_pairs,
                    normalize=True,
                    conf_level=conf_level,
                    difference=True,
                )
            if plot_type == "diff_cdf_solvability":
                file_list.append(
                    save_plot(
                        solver_name=solver_set_name,
                        problem_name=problem_set_name,
                        plot_type=plot_type,
                        normalize=True,
                        extra=solve_tol,
                        plot_title=plot_title,
                        ext=ext,
                        save_as_pickle=save_as_pickle,
                    )
                )
            elif plot_type == "diff_quantile_solvability":
                file_list.append(
                    save_plot(
                        solver_name=solver_set_name,
                        problem_name=problem_set_name,
                        plot_type=plot_type,
                        normalize=True,
                        extra=[solve_tol, beta],
                        plot_title=plot_title,
                        ext=ext,
                        save_as_pickle=save_as_pickle,
                    )
                )
    else:
        solver_names = [
            solver_experiments[0].solver.name
            for solver_experiments in experiments
        ]
        solver_curves = []
        for solver_idx in range(n_solvers):
            solver_sub_curves = []
            # For each problem compute the cdf or quantile of solve times.
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                sub_curve = None
                if plot_type in {"cdf_solvability", "diff_cdf_solvability"}:
                    sub_curve = cdf_of_curves_crossing_times(
                        curves=experiment.progress_curves, threshold=solve_tol
                    )
                if plot_type in {
                    "quantile_solvability",
                    "diff_quantile_solvability",
                }:
                    sub_curve = quantile_cross_jump(
                        curves=experiment.progress_curves,
                        threshold=solve_tol,
                        beta=beta,
                    )
                if sub_curve is not None:
                    solver_sub_curves.append(sub_curve)
            # Plot solvability profile for the solver.
            # Exploit the fact that each solvability profile is an average of more basic curves.
            solver_curve = mean_of_curves(solver_sub_curves)
            solver_curves.append(solver_curve)
            if plot_type in {"cdf_solvability", "quantile_solvability"}:
                # Set up plot.
                if plot_type == "cdf_solvability":
                    file_list.append(
                        setup_plot(
                            plot_type=plot_type,
                            solver_name=experiments[solver_idx][0].solver.name,
                            problem_name=problem_set_name,
                            solve_tol=solve_tol,
                        )
                    )
                elif plot_type == "quantile_solvability":
                    file_list.append(
                        setup_plot(
                            plot_type=plot_type,
                            solver_name=experiments[solver_idx][0].solver.name,
                            problem_name=problem_set_name,
                            beta=beta,
                            solve_tol=solve_tol,
                        )
                    )
                handle = solver_curve.plot()
                if plot_conf_ints or print_max_hw:
                    # Note: "experiments" needs to be a list of list of ProblemSolver objects.
                    bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                        bootstrap_procedure(
                            experiments=[experiments[solver_idx]],
                            n_bootstraps=n_bootstraps,
                            conf_level=conf_level,
                            plot_type=plot_type,  # type: ignore
                            solve_tol=solve_tol,
                            beta=beta,
                            estimator=solver_curve,
                            normalize=True,
                        )
                    )
                    if plot_conf_ints:
                        if isinstance(
                            bs_conf_int_lb_curve, (int, float)
                        ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                            error_msg = "Bootstrap confidence intervals are not available for scalar estimators."
                            raise ValueError(error_msg)
                        plot_bootstrap_conf_ints(
                            bs_conf_int_lb_curve, bs_conf_int_ub_curve
                        )
                    if print_max_hw:
                        if isinstance(
                            bs_conf_int_lb_curve, (int, float)
                        ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                            error_msg = "Max halfwidth is not available for scalar estimators."
                            raise ValueError(error_msg)
                        report_max_halfwidth(
                            curve_pairs=[
                                [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                            ],
                            normalize=True,
                            conf_level=conf_level,
                        )
                if plot_type == "cdf_solvability":
                    file_list.append(
                        save_plot(
                            solver_name=experiments[solver_idx][0].solver.name,
                            problem_name=problem_set_name,
                            plot_type=plot_type,
                            normalize=True,
                            extra=solve_tol,
                            ext=ext,
                            save_as_pickle=save_as_pickle,
                        )
                    )
                elif plot_type == "quantile_solvability":
                    file_list.append(
                        save_plot(
                            solver_name=experiments[solver_idx][0].solver.name,
                            problem_name=problem_set_name,
                            plot_type=plot_type,
                            normalize=True,
                            extra=[solve_tol, beta],
                            ext=ext,
                            save_as_pickle=save_as_pickle,
                        )
                    )
        if plot_type in {"diff_cdf_solvability", "diff_quantile_solvability"}:
            if ref_solver is None:
                error_msg = "Reference solver must be specified for difference profiles."
                raise ValueError(error_msg)
            non_ref_solvers = [
                solver_name
                for solver_name in solver_names
                if solver_name != ref_solver
            ]
            ref_solver_idx = solver_names.index(ref_solver)
            for solver_idx in range(n_solvers):
                if solver_idx is not ref_solver_idx:
                    if plot_type == "diff_cdf_solvability":
                        file_list.append(
                            setup_plot(
                                plot_type=plot_type,
                                solver_name=experiments[solver_idx][
                                    0
                                ].solver.name,
                                problem_name=problem_set_name,
                                solve_tol=solve_tol,
                            )
                        )
                    elif plot_type == "diff_quantile_solvability":
                        file_list.append(
                            setup_plot(
                                plot_type=plot_type,
                                solver_name=experiments[solver_idx][
                                    0
                                ].solver.name,
                                problem_name=problem_set_name,
                                beta=beta,
                                solve_tol=solve_tol,
                            )
                        )
                    diff_solver_curve = difference_of_curves(
                        solver_curves[solver_idx], solver_curves[ref_solver_idx]
                    )
                    handle = diff_solver_curve.plot()
                    if plot_conf_ints or print_max_hw:
                        # Note: "experiments" needs to be a list of list of ProblemSolver objects.
                        bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                            bootstrap_procedure(
                                experiments=[
                                    experiments[solver_idx],
                                    experiments[ref_solver_idx],
                                ],
                                n_bootstraps=n_bootstraps,
                                conf_level=conf_level,
                                plot_type=plot_type,  # type: ignore
                                solve_tol=solve_tol,
                                beta=beta,
                                estimator=diff_solver_curve,
                                normalize=True,
                            )
                        )
                        if plot_conf_ints:
                            if isinstance(
                                bs_conf_int_lb_curve, (int, float)
                            ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                                error_msg = "Bootstrap confidence intervals are not available for scalar estimators."
                                raise ValueError(error_msg)
                            plot_bootstrap_conf_ints(
                                bs_conf_int_lb_curve, bs_conf_int_ub_curve
                            )
                        if print_max_hw:
                            if isinstance(
                                bs_conf_int_lb_curve, (int, float)
                            ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                                error_msg = "Max halfwidth is not available for scalar estimators."
                                raise ValueError(error_msg)
                            report_max_halfwidth(
                                curve_pairs=[
                                    [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                                ],
                                normalize=True,
                                conf_level=conf_level,
                                difference=True,
                            )
                    if plot_type == "diff_cdf_solvability":
                        file_list.append(
                            save_plot(
                                solver_name=experiments[solver_idx][
                                    0
                                ].solver.name,
                                problem_name=problem_set_name,
                                plot_type=plot_type,
                                normalize=True,
                                extra=solve_tol,
                                ext=ext,
                                save_as_pickle=save_as_pickle,
                            )
                        )
                    elif plot_type == "diff_quantile_solvability":
                        file_list.append(
                            save_plot(
                                solver_name=experiments[solver_idx][
                                    0
                                ].solver.name,
                                problem_name=problem_set_name,
                                plot_type=plot_type,
                                normalize=True,
                                extra=[solve_tol, beta],
                                ext=ext,
                                save_as_pickle=save_as_pickle,
                            )
                        )
    return file_list


def plot_terminal_progress(
    experiments: list[ProblemSolver],
    plot_type: Literal["box", "violin"] = "violin",
    normalize: bool = True,
    all_in_one: bool = True,
    plot_title: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
) -> list[str | os.PathLike]:
    """Plot individual or aggregate terminal progress for one or more solvers on a single problem.

    Parameters
    ----------
    experiments : list [``experiment_base.ProblemSolver``]
        ProblemSolver pairs of different solvers on a common problem.
    plot_type : str, default="violin"
        String indicating which type of plot to produce:

            "box" : comparative box plots;

            "violin" : comparative violin plots.
    normalize : bool, default=True
        True if progress curves are to be normalized w.r.t. optimality gaps,
        otherwise False.
    all_in_one : bool, default=True
        True if curves are to be plotted together, otherwise False.
    plot_title : str, opt
        Optional title to override the one that is autmatically generated, only applies if all_in_one is True.
    ext: str, default = '.png'
        Extension to add to image file path to change file type
    save_as_pickle: bool, default = False
        True if plot should be saved to pickle file, False otherwise.
    solver_set_name: str, default = "SOLVER_SET"
        Use to change name of solver groups for plot titles.

    Returns
    -------
    file_list : list [str]
        List compiling path names for plots produced.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if not isinstance(experiments, list) or not all(
        [isinstance(experiment, ProblemSolver) for experiment in experiments]
    ):
        error_msg = "Experiments must be a list of ProblemSolver objects."
        raise TypeError(error_msg)
    if not isinstance(plot_type, str):
        error_msg = "Plot type must be a string."
        raise TypeError(error_msg)
    if not isinstance(normalize, bool):
        error_msg = "Normalize must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(all_in_one, bool):
        error_msg = "All in one must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(plot_title, (str, type(None))):
        error_msg = "Plot title must be a string or None."
        raise TypeError(error_msg)
    if not isinstance(ext, str):
        error_msg = "Extension must be a string."
        raise TypeError(error_msg)
    if not isinstance(save_as_pickle, bool):
        error_msg = "Save as pickle must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(solver_set_name, str):
        error_msg = "Solver set name must be a string."
        raise TypeError(error_msg)
    # Value checking
    if plot_type not in ["box", "violin"]:
        error_msg = "Plot type must be either 'box' or 'violin'."
        raise ValueError(error_msg)

    # Check if problems are the same with the same x0 and x*.
    check_common_problem_and_reference(experiments)
    file_list = []
    # Set up plot.
    n_experiments = len(experiments)
    if all_in_one:
        ref_experiment = experiments[0]
        setup_plot(
            plot_type=plot_type,
            solver_name=solver_set_name,
            problem_name=ref_experiment.problem.name,
            normalize=normalize,
            budget=ref_experiment.problem.factors["budget"],
            plot_title=plot_title,
        )
        # solver_curve_handles = []
        if normalize:
            terminal_data = [
                [
                    experiment.progress_curves[mrep].y_vals[-1]
                    for mrep in range(experiment.n_macroreps)
                ]
                for experiment in experiments
            ]
        else:
            terminal_data = [
                [
                    experiment.objective_curves[mrep].y_vals[-1]
                    for mrep in range(experiment.n_macroreps)
                ]
                for experiment in experiments
            ]
        if plot_type == "box":
            plt.boxplot(terminal_data)
            plt.xticks(
                range(1, n_experiments + 1),
                labels=[experiment.solver.name for experiment in experiments],
            )
        elif plot_type == "violin":
            solver_names = [
                experiments[exp_idx].solver.name
                for exp_idx in range(n_experiments)
                for td in terminal_data[exp_idx]
            ]
            terminal_values = [
                td
                for exp_idx in range(n_experiments)
                for td in terminal_data[exp_idx]
            ]
            terminal_data_dict = {
                "Solvers": solver_names,
                "Terminal": terminal_values,
            }
            terminal_data_df = pd.DataFrame(terminal_data_dict)
            # sns.violinplot(x="Solvers", y="Terminal", data=terminal_data_df, inner="stick", scale="width", showmeans=True, bw = 0.2,  cut=2)
            sns.violinplot(
                x="Solvers",
                y="Terminal",
                data=terminal_data_df,
                inner="stick",
                density_norm="width",
                cut=0.1,
                hue="Solvers",
            )
            if normalize:
                plt.ylabel("Terminal Progress")
            else:
                plt.ylabel("Terminal Objective")
        file_list.append(
            save_plot(
                solver_name=solver_set_name,
                problem_name=ref_experiment.problem.name,
                plot_type=plot_type,
                normalize=normalize,
                plot_title=plot_title,
                ext=ext,
                save_as_pickle=save_as_pickle,
            )
        )
    else:  # Plot separately.
        for experiment in experiments:
            setup_plot(
                plot_type=plot_type,
                solver_name=experiment.solver.name,
                problem_name=experiment.problem.name,
                normalize=normalize,
                budget=experiment.problem.factors["budget"],
            )
            if normalize:
                terminal_data = [
                    experiment.progress_curves[mrep].y_vals[-1]
                    for mrep in range(experiment.n_macroreps)
                ]
            else:
                terminal_data = [
                    experiment.objective_curves[mrep].y_vals[-1]
                    for mrep in range(experiment.n_macroreps)
                ]
            if plot_type == "box":
                plt.boxplot(terminal_data)
                plt.xticks([1], labels=[experiment.solver.name])
            if plot_type == "violin":
                solver_name_rep = [
                    experiment.solver.name for td in terminal_data
                ]
                terminal_data_dict = {
                    "Solver": solver_name_rep,
                    "Terminal": terminal_data,
                }
                terminal_data_df = pd.DataFrame(terminal_data_dict)
                sns.violinplot(
                    x="Solver",
                    y="Terminal",
                    data=terminal_data_df,
                    inner="stick",
                )
            if normalize:
                plt.ylabel("Terminal Progress")
            else:
                plt.ylabel("Terminal Objective")
            file_list.append(
                save_plot(
                    solver_name=experiment.solver.name,
                    problem_name=experiment.problem.name,
                    plot_type=plot_type,
                    normalize=normalize,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )
    return file_list


def plot_terminal_scatterplots(
    experiments: list[list[ProblemSolver]],
    all_in_one: float = True,
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
    problem_set_name: str = "PROBLEM_SET",
) -> list[str | os.PathLike]:
    """Plot a scatter plot of mean and standard deviation of terminal progress.

    Either one plot for each solver or one plot for all solvers.

    Parameters
    ----------
    experiments : list [list [``experiment_base.ProblemSolver``]]
        ProblemSolver pairs used to produce plots.
    all_in_one : bool, default=True
        True if curves are to be plotted together, otherwise False.
    plot_title : str, opt
        Optional title to override the one that is autmatically generated, only applies if all_in_one is True.
    legend_loc : str, default="best"
        specificies location of legend
    ext: str, default = '.png'
        Extension to add to image file path to change file type
    save_as_pickle: bool, default = False
        True if plot should be saved to pickle file, False otherwise.
    solver_set_name: str, default = "SOLVER_SET"
        Use to change name of solver groups for plot titles.
    problem_set_name: str, default = "PROBLEM_SET"
        USe to change name of problem groups for plot titles.

    Returns
    -------
    file_list : list [str]
        List compiling path names for plots produced.

    Raises
    ------
    TypeError

    """
    # Type checking
    if (
        not isinstance(experiments, list)
        or not all(
            [
                isinstance(experiment_list, list)
                for experiment_list in experiments
            ]
        )
        or not all(
            [
                isinstance(experiment, ProblemSolver)
                for experiment in experiment_list
            ]
            for experiment_list in experiments
        )
    ):
        error_msg = (
            "Experiments must be a list of lists of ProblemSolver objects."
        )
        raise TypeError(error_msg)
    if not isinstance(all_in_one, bool):
        error_msg = "All in one must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(plot_title, (str, type(None))):
        error_msg = "Plot title must be a string or None."
        raise TypeError(error_msg)
    if not isinstance(legend_loc, (str, type(None))):
        error_msg = "Legend location must be a string."
        raise TypeError(error_msg)
    if not isinstance(ext, str):
        error_msg = "Extension must be a string."
        raise TypeError(error_msg)
    if not isinstance(save_as_pickle, bool):
        error_msg = "Save as pickle must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(solver_set_name, str):
        error_msg = "Solver set name must be a string."
        raise TypeError(error_msg)
    if not isinstance(problem_set_name, str):
        error_msg = "Problem set name must be a string."
        raise TypeError(error_msg)

    if legend_loc is None:
        legend_loc = "best"

    file_list = []
    # Set up plot.
    n_solvers = len(experiments)
    n_problems = len(experiments[0])
    if all_in_one:
        marker_list = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]
        setup_plot(
            plot_type="terminal_scatter",
            solver_name=solver_set_name,
            problem_name=problem_set_name,
            plot_title=plot_title,
        )
        solver_names = [
            solver_experiments[0].solver.name
            for solver_experiments in experiments
        ]
        solver_curve_handles = []
        handle = None
        for solver_idx in range(n_solvers):
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                color_str = "C" + str(solver_idx)
                marker_str = marker_list[
                    solver_idx % len(marker_list)
                ]  # Cycle through list of marker types.
                # Plot mean and standard deviation of terminal progress.
                terminals = [
                    curve.y_vals[-1] for curve in experiment.progress_curves
                ]
                mean_estimator = np.mean(terminals)
                std_dev_estimator = np.std(terminals, ddof=1)
                handle = plt.scatter(
                    x=mean_estimator,
                    y=std_dev_estimator,
                    color=color_str,
                    marker=marker_str,
                )
            solver_curve_handles.append(handle)
        plt.legend(
            handles=solver_curve_handles, labels=solver_names, loc=legend_loc
        )
        file_list.append(
            save_plot(
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                plot_type="terminal_scatter",
                normalize=True,
                plot_title=plot_title,
                ext=ext,
                save_as_pickle=save_as_pickle,
            )
        )
    else:
        for solver_idx in range(n_solvers):
            ref_experiment = experiments[solver_idx][0]
            setup_plot(
                plot_type="terminal_scatter",
                solver_name=ref_experiment.solver.name,
                problem_name=problem_set_name,
            )
            experiment = None
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                # Plot mean and standard deviation of terminal progress.
                terminals = [
                    curve.y_vals[-1] for curve in experiment.progress_curves
                ]
                mean_estimator = np.mean(terminals)
                std_dev_estimator = np.std(terminals, ddof=1)
                handle = plt.scatter(
                    x=mean_estimator,
                    y=std_dev_estimator,
                    color="C0",
                    marker="o",
                )
            if experiment is not None:
                file_list.append(
                    save_plot(
                        solver_name=experiment.solver.name,
                        problem_name=problem_set_name,
                        plot_type="terminal_scatter",
                        normalize=True,
                        ext=ext,
                        save_as_pickle=save_as_pickle,
                    )
                )
    return file_list


def setup_plot(
    plot_type: Literal[
        "all",
        "mean",
        "quantile",
        "solve_time_cdf",
        "cdf_solvability",
        "quantile_solvability",
        "diff_cdf_solvability",
        "diff_quantile_solvability",
        "area",
        "box",
        "violin",
        "terminal_scatter",
    ],
    solver_name: str = "SOLVER SET",
    problem_name: str = "PROBLEM SET",
    normalize: bool = True,
    budget: int | None = None,
    beta: float | None = None,
    solve_tol: float | None = None,
    plot_title: str | None = None,
) -> None:
    """Create new figure. Add labels to plot and reformat axes.

    Parameters
    ----------
    plot_type : str
        String indicating which type of plot to produce:
            "all" : all estimated progress curves;

            "mean" : estimated mean progress curve;

            "quantile" : estimated beta quantile progress curve;

            "solve_time_cdf" : cdf of solve time;

            "cdf_solvability" : cdf solvability profile;

            "quantile_solvability" : quantile solvability profile;

            "diff_cdf_solvability" : difference of cdf solvability profiles;

            "diff_quantile_solvability" : difference of quantile solvability profiles;

            "area" : area scatterplot;

            "box" : box plot of terminal progress;

            "violin" : violin plot of terminal progress;

            "terminal_scatter" : scatterplot of mean and std dev of terminal progress.
    solver_name : str, default="SOLVER_SET"
        Name of solver.
    problem_name : str, default="PROBLEM_SET"
        Name of problem.
    normalize : bool, default=True
        True if progress curves are to be normalized w.r.t. optimality gaps,
        otherwise False.
    budget : int, optional
        Budget of problem, measured in function evaluations.
    beta : float, optional
        Quantile to compute, e.g., beta quantile; in (0, 1).
    solve_tol : float, optional
        Relative optimality gap definining when a problem is solved; in (0, 1].
    plot_title : str, optional
        Optional title to override the one that is autmatically generated.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if not isinstance(plot_type, str):
        error_msg = "Plot type must be a string."
        raise TypeError(error_msg)
    if not isinstance(solver_name, str):
        error_msg = "Solver name must be a string."
        raise TypeError(error_msg)
    if not isinstance(problem_name, str):
        error_msg = "Problem name must be a string."
        raise TypeError(error_msg)
    if not isinstance(normalize, bool):
        error_msg = "Normalize must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(budget, (int, type(None))):
        error_msg = "Budget must be an integer or None."
        raise TypeError(error_msg)
    if not isinstance(beta, (float, type(None))):
        error_msg = "Beta must be a float or None."
        raise TypeError(error_msg)
    if not isinstance(solve_tol, (float, type(None))):
        error_msg = "Solve tolerance must be a float or None."
        raise TypeError(error_msg)
    if not isinstance(plot_title, (str, type(None))):
        error_msg = "Plot title must be a string or None."
        raise TypeError(error_msg)
    # Value checking
    if plot_type not in [
        "all",
        "mean",
        "quantile",
        "solve_time_cdf",
        "cdf_solvability",
        "quantile_solvability",
        "diff_cdf_solvability",
        "diff_quantile_solvability",
        "area",
        "box",
        "violin",
        "terminal_scatter",
    ]:
        error_msg = f"Plot type '{plot_type}' is not recognized."
        raise ValueError(error_msg)
    if isinstance(beta, float) and not 0 < beta < 1:
        error_msg = "Beta must be in (0, 1)."
        raise ValueError(error_msg)
    if isinstance(solve_tol, float) and not 0 < solve_tol <= 1:
        error_msg = "Solve tolerance must be in (0, 1]."
        raise ValueError(error_msg)

    plt.figure()
    # Set up axes and axis labels.
    if normalize:
        plt.ylabel("Fraction of Initial Optimality Gap", size=14)
        if plot_type != "box" and plot_type != "violin":
            plt.xlabel("Fraction of Budget", size=14)
            plt.xlim((0, 1))
            plt.ylim((-0.1, 1.1))
            plt.tick_params(axis="both", which="major", labelsize=12)
    else:
        plt.ylabel("Objective Function Value", size=14)
        if plot_type != "box" and plot_type != "violin":
            plt.xlabel("Budget", size=14)
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
        if beta is None:
            error_msg = "Beta must be specified for quantile plot."
            raise ValueError(error_msg)
        if normalize:
            title = f"{solver_name} on {problem_name}\n{round(beta, 2)}-Quantile Progress Curve"
        else:
            title = f"{solver_name} on {problem_name}\n{round(beta, 2)}-Quantile Objective Curve"
    elif plot_type == "solve_time_cdf":
        if solve_tol is None:
            error_msg = "Solve tolerance must be specified for cdf plot."
            raise ValueError(error_msg)
        plt.ylabel("Fraction of Macroreplications Solved", size=14)
        title = f"{solver_name} on {problem_name}\nCDF of {round(solve_tol, 2)}-Solve Times"
    elif plot_type == "cdf_solvability":
        if solve_tol is None:
            error_msg = (
                "Solve tolerance must be specified for cdf solvability plot."
            )
            raise ValueError(error_msg)
        plt.ylabel("Problem Averaged Solve Fraction", size=14)
        title = f"CDF-Solvability Profile for {solver_name}\nProfile of CDFs of {round(solve_tol, 2)}-Solve Times"
    elif plot_type == "quantile_solvability":
        if beta is None:
            error_msg = "Beta must be specified for quantile solvability plot."
            raise ValueError(error_msg)
        if solve_tol is None:
            error_msg = "Solve tolerance must be specified for quantile solvability plot."
            raise ValueError(error_msg)
        plt.ylabel("Fraction of Problems Solved", size=14)
        title = f"Quantile Solvability Profile for {solver_name}\nProfile of {round(beta, 2)}-Quantiles of {round(solve_tol, 2)}-Solve Times"
    elif plot_type == "diff_cdf_solvability":
        if solve_tol is None:
            error_msg = (
                "Solve tolerance must be specified for cdf solvability plot."
            )
            raise ValueError(error_msg)
        plt.ylabel("Difference in Problem Averaged Solve Fraction", size=14)
        title = f"Difference of CDF-Solvability Profile for {solver_name}\nDifference of Profiles of CDFs of {round(solve_tol, 2)}-Solve Times"
        plt.plot([0, 1], [0, 0], color="black", linestyle="--")
        plt.ylim((-1, 1))
    elif plot_type == "diff_quantile_solvability":
        if beta is None:
            error_msg = "Beta must be specified for quantile solvability plot."
            raise ValueError(error_msg)
        if solve_tol is None:
            error_msg = "Solve tolerance must be specified for quantile solvability plot."
            raise ValueError(error_msg)
        plt.ylabel("Difference in Fraction of Problems Solved", size=14)
        title = f"Difference of Quantile Solvability Profile for {solver_name}\nDifference of Profiles of {round(beta, 2)}-Quantiles of {round(solve_tol, 2)}-Solve Times"
        plt.plot([0, 1], [0, 0], color="black", linestyle="--")
        plt.ylim((-1, 1))
    elif plot_type == "area":
        plt.xlabel("Mean Area", size=14)
        plt.ylabel("Std Dev of Area")
        # plt.xlim((0, 1))
        # plt.ylim((0, 0.5))
        title = f"{solver_name}\nAreas Under Progress Curves"
    elif plot_type == "box" or plot_type == "violin":
        plt.xlabel("Solvers")
        if normalize:
            plt.ylabel("Terminal Progress")
            title = f"{solver_name} on {problem_name}"
        else:
            plt.ylabel("Terminal Objective")
            title = f"{solver_name} on {problem_name}"
    elif plot_type == "terminal_scatter":
        plt.xlabel("Mean Terminal Progress", size=14)
        plt.ylabel("Std Dev of Terminal Progress")
        # plt.xlim((0, 1))
        # plt.ylim((0, 0.5))
        title = f"{solver_name}\nTerminal Progress"
    else:
        error_msg = f"'{plot_type}' is not implemented."
        raise NotImplementedError(error_msg)
    # if title argument provided, overide prevous title assignment
    if plot_title is not None:
        title = plot_title
    plt.title(title, size=14)


def save_plot(
    solver_name: str,
    problem_name: str,
    plot_type: Literal[
        "all",
        "mean",
        "quantile",
        "solve_time_cdf",
        "cdf_solvability",
        "quantile_solvability",
        "diff_cdf_solvability",
        "diff_quantile_solvability",
        "area",
        "box",
        "violin",
        "terminal_scatter",
    ],
    normalize: bool,
    extra: float | list[float] | None = None,
    plot_title: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
) -> str | os.PathLike:
    """Create new figure. Add labels to plot and reformat axes.

    Parameters
    ----------
    solver_name : str
        Name of solver.
    problem_name : str
        Name of problem.
    plot_type : str
        String indicating which type of plot to produce:
            "all" : all estimated progress curves;

            "mean" : estimated mean progress curve;

            "quantile" : estimated beta quantile progress curve;

            "solve_time_cdf" : cdf of solve time;

            "cdf_solvability" : cdf solvability profile;

            "quantile_solvability" : quantile solvability profile;

            "diff_cdf_solvability" : difference of cdf solvability profiles;

            "diff_quantile_solvability" : difference of quantile solvability profiles;

            "area" : area scatterplot;

            "terminal_scatter" : scatterplot of mean and std dev of terminal progress.
    normalize : bool
        True if progress curves are to be normalized w.r.t. optimality gaps,
        otherwise False.
    extra : float | list [float], optional
        Extra number(s) specifying quantile (e.g., beta) and/or solve tolerance.
    plot_title: str, opt
        Will change name of save file if generic title is overwritten.
    ext: str, default = '.png'
        Extension to add to image file path to change file type
    save_as_pickle: bool, default = False
        True if plot should be saved to pickle file, False otherwise.

    Returns
    -------
    path_name : str
        Path name pointing to location where plot will be saved.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(solver_name, str):
        error_msg = "Solver name must be a string."
        raise TypeError(error_msg)
    if not isinstance(problem_name, str):
        error_msg = "Problem name must be a string."
        raise TypeError(error_msg)
    if plot_type not in [
        "all",
        "mean",
        "quantile",
        "solve_time_cdf",
        "cdf_solvability",
        "quantile_solvability",
        "diff_cdf_solvability",
        "diff_quantile_solvability",
        "area",
        "box",
        "violin",
        "terminal_scatter",
    ]:
        error_msg = f"Plot type '{plot_type}' is not recognized."
        raise ValueError(error_msg)
    if not isinstance(normalize, bool):
        error_msg = "Normalize must be a boolean."
        raise TypeError(error_msg)
    if not isinstance(extra, (float, list, type(None))) or (
        isinstance(extra, list) and not all(isinstance(e, float) for e in extra)
    ):
        error_msg = "Extra must be a float, list of floats, or None."
        raise TypeError(error_msg)
    if not isinstance(plot_title, (str, type(None))):
        error_msg = "Plot title must be a string or None."
        raise TypeError(error_msg)
    if not isinstance(ext, str):
        error_msg = "Extension must be a string."
        raise TypeError(error_msg)
    if not isinstance(save_as_pickle, bool):
        error_msg = "Save as pickle must be a boolean."
        raise TypeError(error_msg)

    # Form string name for plot filename.
    if plot_type == "all":
        plot_name = "all_prog_curves"
    elif plot_type == "mean":
        plot_name = "mean_prog_curve"
    elif plot_type == "quantile":
        plot_name = f"{extra}_quantile_prog_curve"
    elif plot_type == "solve_time_cdf":
        plot_name = f"cdf_{extra}_solve_times"
    elif plot_type == "cdf_solvability":
        plot_name = f"profile_cdf_{extra}_solve_times"
    elif plot_type == "quantile_solvability":
        if (
            isinstance(extra, list)
            and len(extra) == 2
            and isinstance(extra[0], float)
            and isinstance(extra[1], float)
        ):
            plot_name = f"profile_{extra[1]}_quantile_{extra[0]}_solve_times"
        else:
            error_msg = "Extra must be a list of two floats for 'quantile_solvability' plot type."
            raise ValueError(error_msg)
    elif plot_type == "diff_cdf_solvability":
        plot_name = f"diff_profile_cdf_{extra}_solve_times"
    elif plot_type == "diff_quantile_solvability":
        if (
            isinstance(extra, list)
            and len(extra) == 2
            and isinstance(extra[0], float)
            and isinstance(extra[1], float)
        ):
            plot_name = (
                f"diff_profile_{extra[1]}_quantile_{extra[0]}_solve_times"
            )
        else:
            error_msg = "Extra must be a list of two floats for 'diff_quantile_solvability' plot type."
            raise ValueError(error_msg)
    elif plot_type == "area":
        plot_name = "area_scatterplot"
    elif plot_type == "box":
        plot_name = "terminal_box"
    elif plot_type == "violin":
        plot_name = "terminal_violin"
    elif plot_type == "terminal_scatter":
        plot_name = "terminal_scatter"
    else:
        raise NotImplementedError(f"'{plot_type}' is not implemented.")

    plot_dir = os.path.join(EXPERIMENT_DIR, "plots")
    # Create directories if they do no exist.
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if not normalize:
        plot_name = plot_name + "_unnorm"

    # Reformat plot_name to be suitable as a string literal.
    plot_name = plot_name.replace("\\", "")
    plot_name = plot_name.replace("$", "")
    plot_name = plot_name.replace(" ", "_")

    if plot_title is None:
        path_name = os.path.join(
            plot_dir, f"{solver_name}_on_{problem_name}_{plot_name}"
        )
    else:
        path_name = os.path.join(plot_dir, plot_title)
    # add extension to path name
    extended_path_name = f"{path_name}{ext}"
    # Create directories if they do no exist.
    if not os.path.exists("./experiments/plots"):
        os.makedirs("./experiments", exist_ok=True)
        os.makedirs("./experiments/plots")
    # plt.savefig(extended_path_name, bbox_inches="tight")

    # Check to make sure file does not override previous images
    counter = 1
    new_path_name = path_name  # set incase counter not needed
    while os.path.exists(extended_path_name):
        extended_path_name = f"{path_name} ({counter}){ext}"
        new_path_name = f"{path_name} ({counter})"  # use for pickle
        counter += 1
    plt.savefig(extended_path_name, bbox_inches="tight")

    # save plot as pickle
    if save_as_pickle:
        fig = plt.gcf()
        pickle_path = f"{new_path_name}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(fig, f)
    # Return path_name for use in GUI.
    return extended_path_name


class ProblemsSolvers:
    """Base class for running one or more solver on one or more problem.

    Attributes
    ----------
    solver_names : list [str]
        List of solver names.
    n_solvers : int
        Number of solvers.
    problem_names : list [str]
        List of problem names.
    n_problems : int
        Number of problems.
    solvers : list [``base.Solver``]
        List of solvers.
    problems : list [``base.Problem``]
        List of problems.
    all_solver_fixed_factors : dict [dict]
        Fixed solver factors for each solver:
            outer key is solver name;
            inner key is factor name.
    all_problem_fixed_factors : dict [dict]
        Fixed problem factors for each problem:
            outer key is problem name;
            inner key is factor name.
    all_model_fixed_factors : dict of dict
        Fixed model factors for each problem:
            outer key is problem name;
            inner key is factor name.
    experiments : list [list [``experiment_base.ProblemSolver``]]
        All problem-solver pairs.
    file_name_path : str
        Path of .pickle file for saving ``experiment_base.ProblemsSolvers`` object.

    Parameters
    ----------
    solver_factors: list [dict], optional
        List of dictionaries that contain solver factors at different design points.
        Each variant of solver with be crossed together with each vairant of problem.
        (Requires solver_names with a name provided for each index in solver_factors.)
    problem_factors: list [dict], optional
        List of dictionaries that contain problem and model factors at different design points.
        Each variant of problem will be crossed together with each variant of solver.
        (Requires problem_names with a name provided for each index in problem_factors.)
    solver_names : list [str], optional
        List of solver names.
    problem_names : list [str], optional
        List of problem names.
    solver_renames : list [str], optional
        User-specified names for solvers.
    problem_renames : list [str], optional
        User-specified names for problems.
    fixed_factors_filename : str, optional
        Name of .py file containing dictionaries of fixed factors
        for solvers/problems/models.
    solvers : list [``base.Solver``], optional
        List of solvers.
    problems : list [``base.Problem``], optional
        List of problems.
    experiments : list [list [``experiment_base.ProblemSolver``]], optional
        All problem-solver pairs.
    file_name_path : str
        Path of .pickle file for saving ``experiment_base.ProblemsSolvers`` object.
    create_pair_pickles : bool, optional
        True if creating pickle files for each problem-solver pair, False otherwise.
    experiment_name: str, optional
        Name of experiment to be appended to the beginning of output files.

    """

    @property
    def solver_names(self) -> list[str]:
        """List of solver names."""
        return self.__solver_names

    @solver_names.setter
    def solver_names(self, solver_names: list[str]) -> None:
        self.__solver_names = solver_names

    @property
    def n_solvers(self) -> int:
        """Number of solvers."""
        return len(self.solver_names)

    @property
    def problem_names(self) -> list[str]:
        """List of problem names."""
        return self.__problem_names

    @problem_names.setter
    def problem_names(self, problem_names: list[str]) -> None:
        self.__problem_names = problem_names

    @property
    def n_problems(self) -> int:
        """Number of problems."""
        return len(self.problem_names)

    @property
    def solvers(self) -> list[Solver]:
        """List of solvers."""
        return self.__solvers

    @solvers.setter
    def solvers(self, solvers: list[Solver]) -> None:
        self.__solvers = solvers

    @property
    def problems(self) -> list[Problem]:
        """List of problems."""
        return self.__problems

    @problems.setter
    def problems(self, problems: list[Problem]) -> None:
        self.__problems = problems

    @property
    def all_solver_fixed_factors(self) -> dict[str, dict]:
        """Fixed solver factors for each solver."""
        return self.__all_solver_fixed_factors

    @all_solver_fixed_factors.setter
    def all_solver_fixed_factors(
        self, all_solver_fixed_factors: dict[str, dict]
    ) -> None:
        self.__all_solver_fixed_factors = all_solver_fixed_factors

    @property
    def all_problem_fixed_factors(self) -> dict[str, dict]:
        """Fixed problem factors for each problem."""
        return self.__all_problem_fixed_factors

    @all_problem_fixed_factors.setter
    def all_problem_fixed_factors(
        self, all_problem_fixed_factors: dict[str, dict]
    ) -> None:
        self.__all_problem_fixed_factors = all_problem_fixed_factors

    @property
    def all_model_fixed_factors(self) -> dict[str, dict]:
        """Fixed model factors for each problem."""
        return self.__all_model_fixed_factors

    @all_model_fixed_factors.setter
    def all_model_fixed_factors(
        self, all_model_fixed_factors: dict[str, dict]
    ) -> None:
        self.__all_model_fixed_factors = all_model_fixed_factors

    @property
    def experiments(self) -> list[list[ProblemSolver]]:
        """All problem-solver pairs."""
        return self.__experiments

    @experiments.setter
    def experiments(self, experiments: list[list[ProblemSolver]]) -> None:
        self.__experiments = experiments

    @property
    def file_name_path(self) -> str:
        """Path of .pickle file for saving ``experiment_base.ProblemsSolvers`` object."""
        return self.__file_name_path

    @file_name_path.setter
    def file_name_path(self, file_name_path: str) -> None:
        self.__file_name_path = file_name_path

    @property
    def create_pair_pickles(self) -> bool:
        """True if creating pickle files for each problem-solver pair, False otherwise."""
        return self.__create_pair_pickles

    @create_pair_pickles.setter
    def create_pair_pickles(self, create_pair_pickles: bool) -> None:
        self.__create_pair_pickles = create_pair_pickles

    @property
    def experiment_name(self) -> str:
        """Name of experiment to be appended to the beginning of output files."""
        return self.__experiment_name

    @experiment_name.setter
    def experiment_name(self, experiment_name: str) -> None:
        self.__experiment_name = experiment_name

    # TODO: If loading some ProblemSolver objects from file, check that their factors match those in the overall ProblemsSolvers.
    def __init__(
        self,
        solver_factors: list[dict] | None = None,
        problem_factors: list[dict] | None = None,
        solver_names: list[str] | None = None,
        problem_names: list[str] | None = None,
        solver_renames: list[str] | None = None,
        problem_renames: list[str] | None = None,
        fixed_factors_filename: str | None = None,
        solvers: list[Solver] | None = None,
        problems: list[Problem] | None = None,
        experiments: list[list[ProblemSolver]] | None = None,
        file_name_path: str | None = None,
        create_pair_pickles: bool = False,
        experiment_name: str | None = None,
    ) -> None:
        """Initialize ProblemsSolvers object.

        There are three ways to create a ProblemsSolvers object:
        1. Provide the names of the solvers and problems to look up in directory.py.
        2. Provide the lists of unique solver and problem objects to pair.
        3. Provide a list of list of ProblemSolver objects.

        Parameters
        ----------
        solver_factors: list [dict], optional
            List of dictionaries that contain solver factors at different design points.
            Each variant of solver with be crossed together with each vairant of problem.
            (Requires solver_names with a name provided for each index in solver_factors.)
        problem_factors: list [dict], optional
            List of dictionaries that contain problem and model factors at different design points.
            Each variant of problem will be crossed together with each variant of solver.
            (Requires problem_names with a name provided for each index in problem_factors.)
        solver_names : list [str], optional
            List of solver names.
        problem_names : list [str], optional
            List of problem names.
        solver_renames : list [str], optional
            User-specified names for solvers.
        problem_renames : list [str], optional
            User-specified names for problems.
        fixed_factors_filename : str, optional
            Name of .py file containing dictionaries of fixed factors
            for solvers/problems/models.
        solvers : list [``base.Solver``], optional
            List of solvers.
        problems : list [``base.Problem``], optional
            List of problems.
        experiments : list [list [``experiment_base.ProblemSolver``]], optional
            All problem-solver pairs.
        file_name_path : str, optional
            Path of .pickle file for saving ``experiment_base.ProblemsSolvers`` object.
        create_pair_pickles : bool, optional
            True if creating pickle files for each problem-solver pair, False otherwise.
        experiment_name: str, optional
            Name of experiment to be appended to the beginning of output files.

        Raises
        ------
        TypeError

        """
        # Type checking
        if not isinstance(solver_factors, (list, type(None))) or (
            isinstance(solver_factors, list)
            and not all(isinstance(dp, dict) for dp in solver_factors)
        ):
            error_msg = "Solver factors must be a list of dictionaries or None."
            raise TypeError(error_msg)
        if not isinstance(problem_factors, (list, type(None))) or (
            isinstance(problem_factors, list)
            and not all(isinstance(dp, dict) for dp in problem_factors)
        ):
            error_msg = (
                "Problem factors must be a list of dictionaries or None."
            )
            raise TypeError(error_msg)
        if not isinstance(solver_names, (list, type(None))) or (
            isinstance(solver_names, list)
            and not all(isinstance(name, str) for name in solver_names)
        ):
            error_msg = "Solver names must be a list of strings or None."
            raise TypeError(error_msg)
        if not isinstance(problem_names, (list, type(None))) or (
            isinstance(problem_names, list)
            and not all(isinstance(name, str) for name in problem_names)
        ):
            error_msg = "Problem names must be a list of strings or None."
            raise TypeError(error_msg)
        if not isinstance(solver_renames, (list, type(None))) or (
            isinstance(solver_renames, list)
            and not all(isinstance(name, str) for name in solver_renames)
        ):
            error_msg = "Solver renames must be a list of strings or None."
            raise TypeError(error_msg)
        if not isinstance(problem_renames, (list, type(None))) or (
            isinstance(problem_renames, list)
            and not all(isinstance(name, str) for name in problem_renames)
        ):
            error_msg = "Problem renames must be a list of strings or None."
            raise TypeError(error_msg)
        if not isinstance(fixed_factors_filename, (str, type(None))):
            error_msg = "Fixed factors filename must be a string or None."
            raise TypeError(error_msg)
        if not isinstance(solvers, (list, type(None))) or (
            isinstance(solvers, list)
            and not all(isinstance(solver, Solver) for solver in solvers)
        ):
            error_msg = "Solvers must be a list of Solver objects or None."
            raise TypeError(error_msg)
        if not isinstance(problems, (list, type(None))) or (
            isinstance(problems, list)
            and not all(isinstance(problem, Problem) for problem in problems)
        ):
            error_msg = "Problems must be a list of Problem objects or None."
            raise TypeError(error_msg)
        if not isinstance(experiments, (list, type(None))) or (
            isinstance(experiments, list)
            and not all(
                isinstance(experiment_list, list)
                for experiment_list in experiments
            )
            and not all(
                isinstance(experiment, ProblemSolver)
                for experiment_list in experiments
                for experiment in experiment_list
            )
        ):
            error_msg = "Experiments must be a list of lists of ProblemSolver objects or None."
            raise TypeError(error_msg)
        if not isinstance(file_name_path, (str, type(None))):
            error_msg = "File name path must be a string or None."
            raise TypeError(error_msg)
        if not isinstance(create_pair_pickles, bool):
            error_msg = "Create pair pickles must be a boolean."
            raise TypeError(error_msg)
        if not isinstance(experiment_name, (str, type(None))):
            error_msg = "Experiment name must be a string or None."
            raise TypeError(error_msg)
        # Value checking
        # TODO: Implement this

        # set attributes for pickle create and experiment file names
        self.create_pair_pickles = create_pair_pickles
        if experiment_name is not None:
            self.file_header = f"{experiment_name}_"
            self.experiment_name = experiment_name
        else:
            self.file_header = ""
            self.experiment_name = ""
        # For some reason some of these variables weren't being assigned to the
        # class attributes. TODO: Fix this.

        output_dir = os.path.join(EXPERIMENT_DIR, "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if experiments is not None:  # Method #3
            self.experiments = experiments
            self.solvers = [
                experiments[idx][0].solver for idx in range(len(experiments))
            ]
            self.problems = [
                experiment.problem for experiment in experiments[0]
            ]
            self.solver_names = [solver.name for solver in self.solvers]
            self.problem_names = [problem.name for problem in self.problems]
            self.solver_set = self.solver_names
            self.problem_set = self.problem_names

        elif solver_factors is not None and problem_factors is not None:
            # Create solvers list.
            solvers = []
            for index, dp in enumerate(solver_factors):
                if solver_names is None:
                    error_msg = "Solver names must be provided."
                    raise ValueError(error_msg)
                # Get corresponding name of solver from names.
                solver_name = solver_names[index]
                # Assign all factor values from current dp to solver object.
                solver = solver_directory[solver_name](fixed_factors=dp)
                solvers.append(solver)

            # Create problems list.
            problems = []
            for index, dp in enumerate(problem_factors):
                if problem_names is None:
                    error_msg = "Problem names must be provided."
                    raise ValueError(error_msg)
                # Get corresponding name of problem from names.
                problem_name = problem_names[index]
                fixed_factors = {}  # Will hold problem factor values for current dp.
                model_fixed_factors = {}  # Will hold model factor values for current dp.
                # Create default instances of problem and model to compare factor names.
                default_problem = problem_directory[problem_name]()
                default_model = default_problem.model

                # Set factor values for current dp using problem/model specifications
                # to determine if problem or model factor.
                for factor in dp:
                    if factor in default_problem.specifications:
                        fixed_factors[factor] = dp[factor]
                    if factor in default_model.specifications:
                        model_fixed_factors[factor] = dp[factor]
                # Create instance of problem and append to problems list.
                problem = problem_directory[problem_name](
                    fixed_factors=fixed_factors,
                    model_fixed_factors=model_fixed_factors,
                )
                problems.append(problem)
            # rename problems and solvers if applicable
            if solver_renames is not None:
                self.solver_renames = solver_renames
            else:
                if solver_names is None:
                    error_msg = "Solver names must be provided."
                    raise ValueError(error_msg)
                self.solver_renames = solver_names
            if problem_renames is not None:
                self.problem_renames = problem_renames
            else:
                if problem_names is None:
                    error_msg = "Problem names must be provided."
                    raise ValueError(error_msg)
                self.problem_renames = problem_names
            self.experiments = [
                [
                    ProblemSolver(
                        solver=solver,
                        problem=problem,
                        solver_rename=self.solver_renames[sol_indx],
                        problem_rename=self.problem_renames[prob_indx],
                        create_pickle=self.create_pair_pickles,
                        file_name_path=os.path.join(
                            output_dir,
                            f"{self.file_header}{self.solver_renames[sol_indx]}_on_{self.problem_renames[prob_indx]}",
                        ),
                    )
                    for prob_indx, problem in enumerate(problems)
                ]
                for sol_indx, solver in enumerate(solvers)
            ]
            self.solvers = solvers
            self.problems = problems
            if solver_names is not None:
                self.solver_names = solver_names
            if problem_names is not None:
                self.problem_names = problem_names

            if solver_names is None:
                self.solver_set = None
            else:
                self.solver_set = set(solver_names)
            if problem_names is None:
                self.problem_set = None
            else:
                self.problem_set = set(problem_names)

        elif solvers is not None and problems is not None:  # Method #2
            self.experiments = [
                [
                    ProblemSolver(solver=solver, problem=problem)
                    for problem in problems
                ]
                for solver in solvers
            ]
            self.solvers = solvers
            self.problems = problems
            self.solver_names = [solver.name for solver in self.solvers]
            self.problem_names = [problem.name for problem in self.problems]
            self.solver_set = self.solver_names
            self.problem_set = self.problem_names

        else:  # Method #1
            if solver_renames is None:
                if solver_names is None:
                    error_msg = "Solver names must be provided."
                    raise ValueError(error_msg)
                self.solver_names = solver_names
            else:
                self.solver_names = solver_renames
            if problem_renames is None:
                if problem_names is None:
                    error_msg = "Problem names must be provided."
                    raise ValueError(error_msg)
                self.problem_names = problem_names
            else:
                self.problem_names = problem_renames

            # Use this for naming file.
            self.solver_set = solver_names
            self.problem_set = problem_names
            # Read in fixed solver/problem/model factors from .py file in the experiments folder.
            # File should contain three dictionaries of dictionaries called
            #   - all_solver_fixed_factors
            #   - all_problem_fixed_factors
            #   - all_model_fixed_factors
            if fixed_factors_filename is None:
                self.all_solver_fixed_factors = {
                    solver_name: {} for solver_name in self.solver_names
                }
                self.all_problem_fixed_factors = {
                    problem_name: {} for problem_name in self.problem_names
                }
                self.all_model_fixed_factors = {
                    problem_name: {} for problem_name in self.problem_names
                }
            else:
                fixed_factors_filename = (
                    "experiments.inputs." + fixed_factors_filename
                )
                all_factors = importlib.import_module(fixed_factors_filename)
                self.all_solver_fixed_factors = (
                    all_factors.all_solver_fixed_factors
                )
                self.all_problem_fixed_factors = (
                    all_factors.all_problem_fixed_factors
                )
                self.all_model_fixed_factors = (
                    all_factors.all_model_fixed_factors
                )
            # Create all problem-solver pairs (i.e., instances of ProblemSolver class).
            self.experiments = []
            for solver_idx in range(self.n_solvers):
                solver_experiments = []
                for problem_idx in range(self.n_problems):
                    filename = os.path.join(
                        output_dir,
                        f"{self.solver_names[solver_idx]}_on_{self.problem_names[problem_idx]}.pickle",
                    )
                    if os.path.exists(filename):
                        with open(filename, "rb") as file:
                            next_experiment = pickle.load(file)
                        # TODO: Check if the solver/problem/model factors in the file match
                        # those for the ProblemsSolvers.
                    else:
                        if solver_names is None:
                            error_msg = "Solver names must be provided if no file exists."
                            raise ValueError(error_msg)
                        if problem_names is None:
                            error_msg = "Problem names must be provided if no file exists."
                            raise ValueError(error_msg)
                        # If no file exists, create new ProblemSolver object.
                        print(
                            f"No experiment file exists for {self.solver_names[solver_idx]} on {self.problem_names[problem_idx]}. Creating new experiment."
                        )
                        next_experiment = ProblemSolver(
                            solver_name=solver_names[solver_idx],
                            problem_name=problem_names[problem_idx],
                            solver_rename=self.solver_names[solver_idx],
                            problem_rename=self.problem_names[problem_idx],
                            solver_fixed_factors=self.all_solver_fixed_factors[
                                self.solver_names[solver_idx]
                            ],
                            problem_fixed_factors=self.all_problem_fixed_factors[
                                self.problem_names[problem_idx]
                            ],
                            model_fixed_factors=self.all_model_fixed_factors[
                                self.problem_names[problem_idx]
                            ],
                        )
                    solver_experiments.append(next_experiment)
                self.experiments.append(solver_experiments)
                self.solvers = [
                    self.experiments[idx][0].solver
                    for idx in range(len(self.experiments))
                ]
                self.problems = [
                    experiment.problem for experiment in self.experiments[0]
                ]
        # Initialize file path.
        if file_name_path is None:
            solver_names_string = ""
            problem_names_string = ""
            if self.solver_set is not None:
                solver_names_string = "_".join(self.solver_set)
            if self.problem_set is not None:
                problem_names_string = "_".join(self.problem_set)
            self.file_name_path = os.path.join(
                output_dir,
                f"{self.file_header}group_{solver_names_string}_on_{problem_names_string}.pickle",
            )
        else:
            self.file_name_path = file_name_path

            self.solver_set = self.solver_names
            self.problem_set = self.problem_names

    def check_compatibility(self) -> str:
        """Check whether all experiments' solvers and problems are compatible.

        Returns
        -------
        error_str : str
            Error message in the event any problem and solver are incompatible.

        """
        error_str = ""
        for solver_idx in range(self.n_solvers):
            for problem_idx in range(self.n_problems):
                new_error_str = self.experiments[solver_idx][
                    problem_idx
                ].check_compatibility()
                if new_error_str != "":
                    error_str += f"For solver {self.solver_names[solver_idx]} and problem {self.problem_names[problem_idx]}... {new_error_str}"
        return error_str

    def run(self, n_macroreps: int) -> None:
        """Run `n_macroreps` of each solver on each problem.

        Parameters
        ----------
        n_macroreps : int
            Number of macroreplications of the solver to run on the problem.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(n_macroreps, int):
            error_msg = "Number of macroreplications must be an integer."
            raise TypeError(error_msg)
        # Value checking
        if n_macroreps <= 0:
            error_msg = "Number of macroreplications must be positive."
            raise ValueError(error_msg)

        for solver_idx in range(self.n_solvers):
            for problem_idx in range(self.n_problems):
                experiment = self.experiments[solver_idx][problem_idx]
                # If the problem-solver pair has not been run in this way before,
                # run it now and save result to .pickle file.
                if not experiment.has_run:
                    print(
                        f"Running {n_macroreps} macro-replications of {experiment.solver.name} on {experiment.problem.name}."
                    )
                    experiment.run(n_macroreps)
        # Save ProblemsSolvers object to .pickle file.
        self.record_group_experiment_results()

    def post_replicate(
        self,
        n_postreps: int,
        crn_across_budget: bool = True,
        crn_across_macroreps: bool = False,
    ) -> None:
        """For each problem-solver pair, run postreplications at solutions recommended by the solver on each macroreplication.

        Parameters
        ----------
        n_postreps : int
            Number of postreplications to take at each recommended solution.
        crn_across_budget : bool, default=True
            True if CRN used for post-replications at solutions recommended at different times,
            otherwise False.
        crn_across_macroreps : bool, default=False
            True if CRN used for post-replications at solutions recommended on different
            macroreplications, otherwise False.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(n_postreps, int):
            error_msg = "Number of postreplications must be an integer."
            raise TypeError(error_msg)
        if not isinstance(crn_across_budget, bool):
            error_msg = "CRN across budget must be a boolean."
            raise TypeError(error_msg)
        if not isinstance(crn_across_macroreps, bool):
            error_msg = "CRN across macroreplications must be a boolean."
            raise TypeError(error_msg)
        # Value checking
        if n_postreps <= 0:
            error_msg = "Number of postreplications must be positive."
            raise ValueError(error_msg)

        for solver_index in range(self.n_solvers):
            for problem_index in range(self.n_problems):
                experiment = self.experiments[solver_index][problem_index]
                # If the problem-solver pair has not been post-replicated in this way before,
                # post-process it now.
                if not experiment.has_postreplicated:
                    print(
                        f"Post-processing {experiment.solver.name} on {experiment.problem.name}."
                    )
                    experiment.post_replicate(
                        n_postreps, crn_across_budget, crn_across_macroreps
                    )
        # Save ProblemsSolvers object to .pickle file.
        self.record_group_experiment_results()

    def post_normalize(
        self, n_postreps_init_opt: int, crn_across_init_opt: bool = True
    ) -> None:
        """Construct objective curves and (normalized) progress curves for all collections of experiments on all given problem.

        Parameters
        ----------
        n_postreps_init_opt : int
            Number of postreplications to take at initial x0 and optimal x*.
        crn_across_init_opt : bool, default=True
            True if CRN used for post-replications at solutions x0 and x*,
            otherwise False.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(n_postreps_init_opt, int):
            error_msg = "Number of postreplications must be an integer."
            raise TypeError(error_msg)
        if not isinstance(crn_across_init_opt, bool):
            error_msg = (
                "CRN across initial and optimal solutions must be a boolean."
            )
            raise TypeError(error_msg)
        # Value checking
        if n_postreps_init_opt <= 0:
            error_msg = "Number of postreplications must be positive."
            raise ValueError(error_msg)

        for problem_idx in range(self.n_problems):
            experiments_same_problem = [
                self.experiments[solver_idx][problem_idx]
                for solver_idx in range(self.n_solvers)
            ]
            post_normalize(
                experiments=experiments_same_problem,
                n_postreps_init_opt=n_postreps_init_opt,
                crn_across_init_opt=crn_across_init_opt,
            )
        # Save ProblemsSolvers object to .pickle file.
        self.record_group_experiment_results()

    def check_postreplicate(self) -> bool:
        """Check to see if all experiments have been postreplicated.

        Return:
        ------
        bool
            True if all experiments have been postreplicated, false otherwise.

        """
        for solver_idx in range(self.n_solvers):
            for problem_idx in range(self.n_problems):
                experiment = self.experiments[solver_idx][problem_idx]
                if not experiment.has_postreplicated:
                    return False
        return True

    def check_postnormalize(self) -> bool:
        """Check to see if all experiments have been postnormalized.

        Return:
        ------
        bool
            True if all experiments have been postnormalized, false otherwise.

        """
        for solver_idx in range(self.n_solvers):
            for problem_idx in range(self.n_problems):
                experiment = self.experiments[solver_idx][problem_idx]
                if not experiment.has_postreplicated:
                    return False
        return True

    def record_group_experiment_results(self) -> None:
        """Save ``experiment_base.ProblemsSolvers`` object to .pickle file."""
        output_dir = os.path.join(EXPERIMENT_DIR, "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(self.file_name_path, "wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def log_group_experiment_results(self) -> None:
        """Create readable .txt file describing the solvers and problems that make up the ProblemSolvers object."""
        # Create a new text file in experiment/{date/time of launch}/logs folder with correct name.
        new_path = self.file_name_path.replace(
            "outputs", "logs"
        )  # Adjust file_path_name to correct folder.
        new_path = new_path.replace(
            ".pickle", "_group_experiment_results.txt"
        )  # Remove .pickle from .txt file name.

        # Create directory if it does no exist.
        log_dir = os.path.dirname(new_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create text file.
        with open(new_path, "w") as file:
            # Title text file with experiment information.
            file.write(self.file_name_path)
            file.write("\n")
            # Write the name of each problem.
            file.write(
                "----------------------------------------------------------------------------------------------"
            )
            file.write("\nProblems:\n\n")
            for i in range(self.n_problems):
                file.write(f"{self.problem_renames[i]}\n\t")
                # Write model factors for each problem.
                file.write("Model Factors:\n")
                for key, value in self.problems[i].model.factors.items():
                    # Excluding model factors corresponding to decision variables.
                    if key not in self.problems[i].model_decision_factors:
                        file.write(f"\t\t{key}: {value}\n")
                # Write problem factors for each problem.
                file.write("\n\tProblem Factors:\n")
                for key, value in self.problems[i].factors.items():
                    file.write(f"\t\t{key}: {value}\n")
                file.write("\n")
            file.write(
                "----------------------------------------------------------------------------------------------"
            )
            # Write the name of each Solver.
            file.write("\nSolvers:\n\n")
            # Write solver factors for each solver.
            for j in range(self.n_solvers):
                file.write(f"{self.solver_renames[j]}\n\t")
                file.write("Solver Factors:\n")
                for key, value in self.solvers[
                    j
                ].factors.items():  # changed from i to j
                    file.write(f"\t\t{key}: {value}\n")
                file.write("\n")
            file.write(
                "----------------------------------------------------------------------------------------------"
            )
            # Write the name of pickle files for each Problem-Solver pair if created.
            if self.create_pair_pickles:
                file.write(
                    "\nThe .pickle files for the associated Problem-Solver pairs are:\n"
                )
                for solver_group in self.experiments:
                    for experiment in solver_group:
                        directory, file_name = os.path.split(
                            experiment.file_name_path
                        )
                        file.write(f"{file_name}\n")
            # for p in self.problem_names:
            #     for s in self.solver_names:
            #         file.write(f"\t{s}_on_{p}.pickle\n")
        file.close()

    def report_group_statistics(
        self,
        solve_tols: list[float] | None = None,
        csv_filename: str = "df_solver_results",
    ) -> None:
        """Report statistics for all solvers on all problems.

        Parameters
        ----------
        solve_tols : list [float], optional
            Relative optimality gap(s) definining when a problem is solved; in (0,1].
        csv_filename : str, optional
            Name of .csv file to print output to. Do not include '.csv' extension.

        Raises
        ------
        TypeError
        ValueError

        """
        # Assign default values
        if solve_tols is None:
            solve_tols = [0.05, 0.10, 0.20, 0.50]
        # Type checking
        if not isinstance(solve_tols, list) or not all(
            isinstance(tol, float) for tol in solve_tols
        ):
            error_msg = "Solve tols must be a list of floats or None."
            raise TypeError(error_msg)
        if not isinstance(csv_filename, str):
            error_msg = "CSV filename must be a string."
            raise TypeError(error_msg)
        # Value checking
        if not all(0 < tol <= 1 for tol in solve_tols):
            error_msg = "Solve tols must be in (0,1]."
            raise ValueError(error_msg)
        # TODO: figure out if we should also check for increasing order of solve_tols

        # create dictionary of common solvers and problems
        pair_dict = {}  # used to hold pairs of

        for sublist in self.experiments:
            for obj in sublist:
                solver = type(obj.solver).__name__
                problem = type(obj.problem).__name__
                key = (solver, problem)
                if key not in pair_dict:
                    pair_dict[key] = [obj]
                else:
                    pair_dict[key].append(obj)
        for (solver, problem), pair_list in pair_dict.items():
            csv_filename = f"{self.file_header}{solver}_on_{problem}_results"
            self.report_statistics(
                pair_list=pair_list,
                solve_tols=solve_tols,
                csv_filename=csv_filename,
            )

    def report_statistics(
        self,
        pair_list: list[ProblemSolver],
        solve_tols: list[float] | None = None,
        csv_filename: str = "df_solver_results",
    ) -> None:
        """For each design point, calculate statistics from each macoreplication and print to csv.

        Parameters
        ----------
        pair_list: list [``experiment_base.ProblemSolver``]
            List of ProblemSolver objects.
        solve_tols : list [float], default = [0.05, 0.10, 0.20, 0.50]
            Relative optimality gap(s) definining when a problem is solved; in (0,1].
        csv_filename : str, default="df_solver_results"
            Name of .csv file to print output to.

        Raises
        ------
        TypeError
        ValueError

        """
        # Assign default values
        if solve_tols is None:
            solve_tols = [0.05, 0.10, 0.20, 0.50]
        # Type checking
        if not isinstance(pair_list, list) or not all(
            isinstance(obj, ProblemSolver) for obj in pair_list
        ):
            error_msg = "Pair list must be a list of ProblemSolver objects."
            raise TypeError(error_msg)
        if not isinstance(solve_tols, list) or not all(
            isinstance(tol, float) for tol in solve_tols
        ):
            error_msg = "Solve tols must be a list of floats or None."
            raise TypeError(error_msg)
        if not isinstance(csv_filename, str):
            error_msg = "CSV filename must be a string."
            raise TypeError(error_msg)
        # Value checking
        if not all(0 < tol <= 1 for tol in solve_tols):
            error_msg = "Solve tols must be in (0,1]."
            raise ValueError(error_msg)
        # TODO: figure out if we should also check for increasing order of solve_tols

        # Create directory if it does no exist.
        log_dir = os.path.join(EXPERIMENT_DIR, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_path = os.path.join(log_dir, f"{csv_filename}.csv")
        with open(file_path, mode="w", newline="") as output_file:
            csv_writer = csv.writer(
                output_file,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            base_experiment = pair_list[0]
            solver_factor_names = list(
                base_experiment.solver.specifications.keys()
            )
            problem_factor_names = list(
                base_experiment.problem.specifications.keys()
            )
            model_factor_names = list(
                set(base_experiment.problem.model.specifications.keys())
                - base_experiment.problem.model_decision_factors
            )
            # Concatenate solve time headers.
            solve_time_headers = [
                [f"{solve_tol}-Solve Time", f"{solve_tol}-Solved? (Y/N)"]
                for solve_tol in solve_tols
            ]
            solve_time_headers = list(
                itertools.chain.from_iterable(solve_time_headers)
            )
            # Print headers.
            csv_writer.writerow(
                [
                    "DesignPt#",
                    "SolverName",
                    *solver_factor_names,
                    "ProblemName",
                    *problem_factor_names,
                    *model_factor_names,
                    "MacroRep#",
                    "Final Relative Optimality Gap",
                    "Area Under Progress Curve",
                    *solve_time_headers,
                    "Initial Solution",
                    "Initial Objective Function Value",
                    "Optimal Solution",
                    "Optimal Objective Function Value",
                ]
            )
            # Compute performance metrics.
            for designpt_index in range(len(pair_list)):
                experiment = pair_list[designpt_index]
                solver_name = experiment.solver.name
                problem_name = experiment.problem.name
                # Parse lists of factors.
                solver_factor_list = [
                    experiment.solver.factors[solver_factor_name]
                    for solver_factor_name in solver_factor_names
                ]
                problem_factor_list = [
                    experiment.problem.factors[problem_factor_name]
                    for problem_factor_name in problem_factor_names
                ]
                model_factor_list = [
                    experiment.problem.model.factors[model_factor_name]
                    for model_factor_name in model_factor_names
                ]
                for mrep in range(experiment.n_macroreps):
                    progress_curve = experiment.progress_curves[mrep]
                    # Parse list of statistics.
                    solve_time_values = [
                        [
                            progress_curve.compute_crossing_time(
                                threshold=solve_tol
                            ),
                            int(
                                progress_curve.compute_crossing_time(
                                    threshold=solve_tol
                                )
                                < np.inf
                            ),
                        ]
                        for solve_tol in solve_tols
                    ]
                    solve_time_values = list(
                        itertools.chain.from_iterable(solve_time_values)
                    )
                    statistics_list = [
                        progress_curve.y_vals[-1],
                        progress_curve.compute_area_under_curve(),
                        *solve_time_values,
                    ]
                    init_sol = tuple([round(x, 4) for x in experiment.x0])
                    int_obj = experiment.x0_postreps[mrep]
                    opt_sol = tuple(
                        [
                            round(x, 4)
                            for x in experiment.all_recommended_xs[mrep][-1]
                        ]
                    )
                    opt_obj = experiment.all_est_objectives[mrep][-1]
                    solution_list = [init_sol, int_obj, opt_sol, opt_obj]
                    print_list = [
                        designpt_index,
                        solver_name,
                        *solver_factor_list,
                        problem_name,
                        *problem_factor_list,
                        *model_factor_list,
                        mrep,
                        *statistics_list,
                        *solution_list,
                    ]
                    csv_writer.writerow(print_list)


def read_group_experiment_results(
    file_name_path: str | os.PathLike,
) -> ProblemsSolvers:
    """Read in ``experiment_base.ProblemsSolvers`` object from .pickle file.

    Parameters
    ----------
    file_name_path : str | os.PathLike
        Path of .pickle file for reading ``experiment_base.ProblemsSolvers`` object.

    Returns
    -------
    groupexperiment : ``experiment_base.ProblemsSolvers``
        Problem-solver group that has been run or has been post-processed.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if not isinstance(file_name_path, (str, os.PathLike)):
        error_msg = "File name path must be a string or os.PathLike object."
        raise TypeError(error_msg)
    # Value checking
    if not os.path.exists(file_name_path):
        error_msg = "File name path does not exist."
        raise ValueError(error_msg)

    with open(file_name_path, "rb") as file:
        groupexperiment = pickle.load(file)
    return groupexperiment


def find_unique_solvers_problems(
    experiments: list[ProblemSolver],
) -> tuple[list[Solver], list[Problem]]:
    """Identify the unique problems and solvers in a collection of experiments.

    Parameters
    ----------
    experiments : list [``experiment_base.ProblemSolver``]
        ProblemSolver pairs of different solvers on different problems.

    Returns
    -------
    list [``base.Solver``]
        Unique solvers.
    list [``base.Problem``]
        Unique problems.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(experiments, list) or not all(
        isinstance(experiment, ProblemSolver) for experiment in experiments
    ):
        error_msg = "Experiments must be a list of ProblemSolver objects."
        raise TypeError(error_msg)

    unique_solvers = list(
        set([experiment.solver for experiment in experiments])
    )
    unique_problems = list(
        set([experiment.problem for experiment in experiments])
    )
    # unique_solvers = []
    # unique_problems = []
    # for experiment in experiments:
    #     if experiment.solver not in unique_solvers:
    #         unique_solvers.append(experiment.solver)
    #     if experiment.problem not in unique_problems:
    #         unique_problems.append(experiment.problem)
    return unique_solvers, unique_problems


def find_missing_experiments(
    experiments: list[ProblemSolver],
) -> tuple[list[Solver], list[Problem], list[tuple[Solver, Problem]]]:
    """Identify problem-solver pairs that are not part of a list of experiments.

    Parameters
    ----------
    experiments : list [``experiment_base.ProblemSolver``]
        Problem-solver pairs of different solvers on different problems.

    Returns
    -------
    list [``base.Solver``]
        List of solvers present in the list of experiments
    list [``base.Problem``]
        List of problems present in the list of experiments.
    list [tuple [``base.Solver``, ``base.Problem``]]
        List of names of missing problem-solver pairs.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(experiments, list) or not all(
        isinstance(experiment, ProblemSolver) for experiment in experiments
    ):
        error_msg = "Experiments must be a list of ProblemSolver objects."
        raise TypeError(error_msg)

    pairs = [
        (experiment.solver, experiment.problem) for experiment in experiments
    ]
    unique_solvers, unique_problems = find_unique_solvers_problems(experiments)
    missing = []
    for solver in unique_solvers:
        for problem in unique_problems:
            if (solver, problem) not in pairs:
                missing.append((solver, problem))
    return unique_solvers, unique_problems, missing


def make_full_metaexperiment(
    existing_experiments: list[ProblemSolver],
    unique_solvers: list[Solver],
    unique_problems: list[Problem],
    missing_experiments: list[tuple[Solver, Problem]],
) -> ProblemsSolvers:
    """Create experiment objects for missing problem-solver pairs and run them.

    Parameters
    ----------
    existing_experiments : list [``experiment_base.ProblemSolver``]
        Problem-solver pairs of different solvers on different problems.
    unique_solvers : list [``base.Solver objects``]
        List of solvers present in the list of experiments.
    unique_problems : list [``base.Problem``]
        List of problems present in the list of experiments.
    missing_experiments : list [tuple [``base.Solver``, ``base.Problem``]]
        List of missing problem-solver pairs.

    Returns
    -------
    metaexperiment : ``experiment_base.ProblemsSolvers``
        New ProblemsSolvers object.

    Raises
    ------
    TypeError

    """
    # Type checking
    if not isinstance(existing_experiments, list) or not all(
        isinstance(experiment, ProblemSolver)
        for experiment in existing_experiments
    ):
        error_msg = (
            "Existing experiments must be a list of ProblemSolver objects."
        )
        raise TypeError(error_msg)
    if not isinstance(unique_solvers, list) or not all(
        isinstance(solver, Solver) for solver in unique_solvers
    ):
        error_msg = "Unique solvers must be a list of Solver objects."
        raise TypeError(error_msg)
    if not isinstance(unique_problems, list) or not all(
        isinstance(problem, Problem) for problem in unique_problems
    ):
        error_msg = "Unique problems must be a list of Problem objects."
        raise TypeError(error_msg)
    if (
        not isinstance(missing_experiments, list)
        or not all(isinstance(pair, tuple) for pair in missing_experiments)
        or not all(len(pair) == 2 for pair in missing_experiments)
        or not all(
            isinstance(pair[0], Solver) and isinstance(pair[1], Problem)
            for pair in missing_experiments
        )
    ):
        error_msg = "Missing experiments must be a list of tuples of Solver and Problem objects."
        raise TypeError(error_msg)

    # Ordering of solvers and problems in unique_solvers and unique_problems
    # is used to construct experiments.
    full_experiments = [
        [] * len(unique_problems) for _ in range(len(unique_solvers))
    ]
    for experiment in existing_experiments:
        solver_idx = unique_solvers.index(experiment.solver)
        problem_idx = unique_problems.index(experiment.problem)
        full_experiments[solver_idx][problem_idx] = experiment
    for pair in missing_experiments:
        solver_idx = unique_solvers.index(pair[0])
        problem_idx = unique_problems.index(pair[1])
        full_experiments[solver_idx][problem_idx] = ProblemSolver(
            solver=pair[0], problem=pair[1]
        )
    metaexperiment = ProblemsSolvers(experiments=full_experiments)
    return metaexperiment


def validate_ruby_install() -> bool:
    """
    Check if Ruby is installed on the system or is on the system path.

    Returns
    -------
    bool
        False is ruby is installed on the system path
        True if ruby is installed via WSL

    Raises
    ------
    Exception
        If Ruby is not installed on the system or is not on the system path.
    """
    # Check if Ruby is on the system path
    results = subprocess.run(
        "ruby -v",
        shell=True,
        capture_output=True,
    )
    # Return false if it is
    if results.returncode == 0:
        return False

    # If Ruby is not on the system path, check if it is installed on WSL
    if os.name == "nt" and shutil.which("wsl") is not None:
        # If WSL is installed, check if Ruby is installed on WSL
        result = subprocess.run(
            'wsl -e bash -lic "ruby -v"',
            shell=True,
            capture_output=True,
        )
        # If Ruby is installed on WSL, return True
        if result.returncode == 0:
            return True
        # The internet claims that WSL will shut down automatically if not
        # being used, so we shouldn't risk accidentally closing it while
        # the user is using it elsewhere. The system should take care of it
        # automatically.

    # If we're here, Ruby is not installed on the system or on WSL
    # Raise an exception
    error_msg = "Ruby is not installed on the system or is not on the system path. Please install Ruby or add it to the system path."
    raise Exception(error_msg)


def validate_gem_install(design_type: str, installed_via_wsl: bool) -> str:
    # Set command prefix based on if WSL is being used
    if installed_via_wsl:
        command_prefix = 'wsl -e bash -lic "'
        command_suffix = '"'
    else:
        command_prefix = ""
        command_suffix = ""

    datafarming_stack = ["nolhs"]

    # Check the design type
    if design_type in datafarming_stack:
        command = (
            f"{command_prefix}stack_{design_type}.rb --help{command_suffix}"
        )
        results = subprocess.run(
            command,
            shell=True,
            capture_output=True,
        )
        # If the return code is 0, then the command was successful
        if results.returncode == 0:
            return f"stack_{design_type}.rb"

        # The command was not successful, so check to see if the gem is installed
        # Check to see if the datafarming gem is installed
        command = f"{command_prefix}gem list{command_suffix}"
        results = subprocess.run(
            command,
            shell=True,
            capture_output=True,
        )
        # If the return code is not 0, then the command was not successful
        # Let's figure out what error we're throwing
        # If the datafarming gem is not present, then tell the user
        # that they need to install it
        if "datafarming" not in results.stdout.decode("utf-8"):
            error_msg = "Datafarming gem is not installed. Please install it by running:\n"
            error_msg += f"`{command_prefix}gem install datafarming -v 1.4{command_suffix}`"
            raise Exception(error_msg)
        installed_gems = results.stdout.decode("utf-8").split("\n")
        # If the datafarming gem is present, then check to see if the version is correct
        # Strip away all the information except for version(s)
        datafarming_gem_installs = [
            gem.split(" ")[1]
            for gem in installed_gems
            if gem.startswith("datafarming ")
        ]
        # Strip away anything that isn't a period or a number
        datafarming_versions = [
            re.sub(r"[^0-9.]", "", version)
            for version in datafarming_gem_installs
        ]
        # Check for valid versions (min <= version < max)
        min_version = "1.0.0"
        max_version = "2.0.0"
        version_check_results = [
            min_version <= version < max_version
            for version in datafarming_versions
        ]
        if not any(version_check_results):
            # Write the correct error message depending on plurality
            if len(version_check_results) == 1:
                error_msg = f"Datafarming gem is installed, but the installed version {datafarming_versions} is not supported."
            else:
                error_msg = f"Datafarming gem is installed, but the installed versions {datafarming_versions.sort()} are not supported."
            error_msg += (
                f" Please install version {min_version} <= x < {max_version}."
            )
            error_msg += f" This can be done by running: `{command_prefix}gem install datafarming -v 1.4{command_suffix}`"
            raise Exception(error_msg)
        # Not totally sure what error to catch here, in theory it should only
        # ever error due to the datafarming gem not being installed or having
        # an incompatible version
        error_msg = "An error occurred while checking if the datafarming gem is installed."
        raise Exception(error_msg)
    else:
        error_msg = "Invalid design type."
        raise ValueError(error_msg)


def create_design_list_from_table(design_table: pd.DataFrame) -> list:
    # Create list of solver or problem objects for each dp using design_table.
    design_list = []
    dp_dict = design_table.to_dict(
        orient="list"
    )  # Creates dictonary of table to convert values to proper datatypes.
    for dp in range(len(design_table)):
        dp_factors = {}
        for factor in dp_dict:
            factor_str = str(dp_dict[factor][dp])
            dp_factors[factor] = ast.literal_eval(factor_str)
        design_list.append(dp_factors)
    return design_list


def create_design(
    name: str,
    factor_headers: list[str],
    factor_settings_filename: str,
    fixed_factors: dict,
    class_type: Literal["solver", "problem", "model"],
    n_stacks: int = 1,
    design_type: Literal["nolhs"] = "nolhs",
    cross_design_factors: dict | None = None,
    csv_filename: str | None = None,
) -> list:
    """Create a design of solver or problem factors using Ruby.

    Parameters
    ----------
    name : str
        Name of solver, problem, or model.
    factor_headers : list[str]
        List of factor names that are changing in the design.
    factor_settings_filename : str
        name of factor settings file within data_farming_experiments folder.
    fixed_factors : dict
        dict of fixed factor values that are different that defaults.
    n_stacks : int, optional
        number of stacks for ruby calculation. The default is '1'.
    design_type : str, optional
        design type for ruby calculation. The default is 'nolhs'.
    cross_design_factors : dict, optional
        dict of lists of values of factors to include in cross design. The default is None.
    class_type: str, optional
        determines class type (solver, problem, or model) that design is over. Problem automatically combines problem factors with model factors. Choose model to run without any associated problem(s).
        default is 'solver'
    csv_filename: str, optional
        override default csv file name

    Returns
    -------
    design_list : list
        list that contains a dict of factor values for every design point.

    Throws
    ------
    Exception
        If ruby is not installed on the system or if the design type is not valid.

    Raises
    ------
    TypeError
    ValueError

    """
    # Default values
    if cross_design_factors is None:
        cross_design_factors = {}
    if csv_filename is None:
        csv_filename = factor_settings_filename

    # Type checking
    if not isinstance(name, str):
        error_msg = "Name must be a string."
        raise TypeError(error_msg)
    if not isinstance(factor_headers, list) or not all(
        isinstance(header, str) for header in factor_headers
    ):
        error_msg = "Factor headers must be a list of strings."
        raise TypeError(error_msg)
    if not isinstance(factor_settings_filename, str):
        error_msg = "Factor settings filename must be a string."
        raise TypeError(error_msg)
    if not isinstance(fixed_factors, dict):
        error_msg = "Fixed factors must be a dictionary."
        raise TypeError(error_msg)
    if not isinstance(n_stacks, int):
        error_msg = "Number of stacks must be an integer."
        raise TypeError(error_msg)
    if not isinstance(design_type, str):
        error_msg = "Design type must be a string."
        raise TypeError(error_msg)
    if not isinstance(cross_design_factors, dict):
        error_msg = "Cross design factors must be a dictionary or None."
        raise TypeError(error_msg)
    if not isinstance(class_type, str):
        error_msg = "Class type must be a string."
        raise TypeError(error_msg)
    if not isinstance(csv_filename, str):
        error_msg = "CSV filename must be a string or None."
        raise TypeError(error_msg)

    # TODO: add additional checking
    # Value checking
    if n_stacks <= 0:
        error_msg = "Number of stacks must be positive."
        raise ValueError(error_msg)
    if design_type not in ["nolhs"]:
        error_msg = "Invalid design type."
        raise ValueError(error_msg)

    # Search directories to create object based on name provided.
    if class_type == "solver":
        if name not in solver_directory:
            error_msg = f"Solver name {name} not found in solver directory."
            raise ValueError(error_msg)
        design_object = solver_directory[name]()
    elif class_type == "problem":
        if name not in problem_directory:
            error_msg = f"Problem name {name} not found in problem directory."
            raise ValueError(error_msg)
        design_object = problem_directory[name]()
    elif class_type == "model":
        if name not in model_directory:
            error_msg = f"Model name {name} not found in model directory."
            raise ValueError(error_msg)
        design_object = model_directory[name]()

    # Make directory to store the current design file.
    data_farming_path = os.path.join(EXPERIMENT_DIR, "data_farming")
    if not os.path.exists(data_farming_path):
        os.makedirs(data_farming_path)

    source_file = os.path.join(
        data_farming_path, f"{factor_settings_filename}.txt"
    )
    design_file = os.path.join(
        data_farming_path, f"{factor_settings_filename}_design.txt"
    )
    # If the dest file already exists, delete it
    # TODO: investigate if this may cause issues with multiple concurrent designs
    if os.path.exists(design_file):
        os.remove(design_file)

    # Only run the Ruby script if there are factors to change
    if len(factor_headers) > 0:
        # Check if Ruby is installed on the system.
        installed_via_wsl: bool = validate_ruby_install()
        # Check if the datafarming gem is installed
        command_file: str = validate_gem_install(design_type, installed_via_wsl)

        # Create solver factor design from .txt file of factor settings.
        if installed_via_wsl:
            # Replace the drive letter with the WSL equivalent
            windows_drive_letter = os.path.splitdrive(data_farming_path)[0]
            linux_drive_letter = (
                f"/mnt/{windows_drive_letter.strip(':').lower()}"
            )
            source_file_wsl = source_file.replace(
                windows_drive_letter, linux_drive_letter
            )
            design_file_wsl = design_file.replace(
                windows_drive_letter, linux_drive_letter
            )
            # Replace backslashes with forward slashes
            source_file_wsl = source_file_wsl.replace("\\", "/")
            design_file_wsl = design_file_wsl.replace("\\", "/")
            command = f"{command_file} -s {n_stacks} {source_file_wsl} > {design_file_wsl}"
            command = f'wsl -e bash -lic "{command}"'
        else:
            command = (
                f"{command_file} -s {n_stacks} {source_file} > {design_file}"
            )
        completed_process = subprocess.run(
            command, capture_output=True, shell=True
        )
        # If the design file doesn't exist, there was an error in the Ruby script.
        if not os.path.exists(design_file):
            error_msg = completed_process.stderr.decode("utf-8")
            raise Exception(
                f"Ruby script did not complete successfully.\nError:\n{error_msg}"
            )

        # Read in design matrix from .txt file. Result is a pandas DataFrame.
        try:
            design_table = pd.read_csv(
                design_file,
                header=None,
                delimiter="\t",
                encoding="utf-8",
            )
        except pd.errors.EmptyDataError:
            error_msg = "Error in Ruby script. No data in design file.\nMake sure to select factors for data farming."
            raise Exception(error_msg) from pd.errors.EmptyDataError
        design_table.columns = factor_headers  # Add factor headers names to dt.
    else:
        # Grab one key/value pair from the specifications
        first_item = design_object.specifications
        # Create a DataFrame with the key/value pair
        design_table = pd.DataFrame(first_item, index=[0])

    # Combine model and problem specifications for problems
    if isinstance(design_object, Problem):
        specifications = {
            **design_object.specifications,
            **design_object.model.specifications,
        }
    else:
        specifications = design_object.specifications

    # Add default values to str dict for unspecified factors.
    for factor in specifications:
        default = specifications[factor].get("default")
        if factor not in fixed_factors and factor not in factor_headers:
            fixed_factors[factor] = default

    # Add all the fixed factors to the design table
    for factor in fixed_factors:
        design_table[factor] = str(
            fixed_factors[factor]
        )  # Change to string to ensure correct addition of tuples & list data types.

    # Add cross design factors to design table.
    if len(cross_design_factors) != 0:
        # num_cross = 0 # number of times cross design is run

        # Create combination of categorical factor options.
        cross_factor_names = list(cross_design_factors.keys())
        combinations = itertools.product(
            *(cross_design_factors[opt] for opt in cross_factor_names)
        )

        new_design_table = pd.DataFrame()  # Temp empty value.
        for combination in combinations:
            combination_dict = dict(
                zip(cross_factor_names, combination)
            )  # Dictionary containing current combination of cross design factor values.
            working_design_table = design_table.copy()

            for factor in combination_dict:
                str_factor_val = str(combination_dict[factor])
                working_design_table[factor] = str_factor_val

            new_design_table = pd.concat(
                [new_design_table, working_design_table], ignore_index=True
            )

        design_table = new_design_table

    design_list = create_design_list_from_table(design_table)

    # check factors for each design point
    for dp in design_list:
        if class_type == "solver":
            # initialize temporary solver to run factor checks
            temp = solver_directory[name](fixed_factors=dp)
        if class_type == "model":
            # initialize temporary model to run factor checks
            temp = model_directory[name](fixed_factors=dp)
            # run check function on temp model
            temp.run_all_checks(factor_names=dp.keys())
        if class_type == "problem":
            # seperate problem and model factors in dp
            problem_factor_names = design_object.specifications.keys()
            problem_factors = {}
            model_factors = {}
            for factor in dp:
                if factor in problem_factor_names:
                    problem_factors[factor] = dp[factor]
                else:
                    model_factors[factor] = dp[factor]
            # initialize temporary problem to run factor checks
            temp_problem = problem_directory[name](
                fixed_factors=problem_factors, model_fixed_factors=model_factors
            )
            # initialize temporary model to run factor checks
            model_factor_names = list(temp_problem.model.specifications.keys())
            temp_problem.model.run_all_checks(factor_names=model_factor_names)

    # Write extra design information to design table.
    design_table.insert(0, "design_num", range(len(design_table)))
    design_table["name"] = design_object.name
    design_table["design_type"] = design_type
    design_table["num_stacks"] = str(n_stacks)
    # Dump the design table to a csv file.
    design_file_csv = design_file.replace(".txt", ".csv")
    design_table.to_csv(design_file_csv, mode="w", header=True, index=False)

    # Now return the list from earlier
    return design_list
