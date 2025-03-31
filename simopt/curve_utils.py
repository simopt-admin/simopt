from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from simopt.curve import Curve

# Imports exclusively used when type checking
# Prevents imports from being executed at runtime
if TYPE_CHECKING:
    from collections.abc import Iterable


def mean_of_curves(curves: Iterable[Curve]) -> Curve:
    """Compute pointwise (w.r.t. x-values) mean of curves.

    Starting and ending x-values must coincide for all curves.

    Parameters
    ----------
    curves : Iterable [``experiment_base.Curve``]
        Collection of curves to aggregate.

    Returns
    -------
    ``experiment_base.Curve object``
        Mean curve.

    Raises
    ------
    TypeError
    """
    from statistics import mean

    try:
        # Collect unique x-values across all curves
        unique_x_vals = sorted(
            {x_val for curve in curves for x_val in curve.x_vals}
        )

        # Compute pointwise means using generator expressions
        mean_y_vals = [
            mean(curve.lookup(x_val) for curve in curves)
            for x_val in unique_x_vals
        ]

        return Curve(x_vals=unique_x_vals, y_vals=mean_y_vals)

    except AttributeError as e:
        error_msg = "Curves must be an iterable of Curve objects."
        logging.error(error_msg)
        raise TypeError(error_msg) from e


def quantile_of_curves(curves: Iterable[Curve], beta: float) -> Curve:
    """Compute pointwise (w.r.t. x values) quantile of curves.

    Starting and ending x values must coincide for all curves.

    Parameters
    ----------
    curves : Iterable [``experiment_base.Curve``]
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
    from statistics import quantiles

    try:
        # Collect unique x-values across all curves
        unique_x_vals = sorted(
            {x_val for curve in curves for x_val in curve.x_vals}
        )

        # Precompute quantile index
        quantile_idx = int(beta * 99)

        # Compute pointwise quantiles
        quantile_y_vals = [
            quantiles((curve.lookup(x_val) for curve in curves), n=100)[
                quantile_idx
            ]
            for x_val in unique_x_vals
        ]

        return Curve(x_vals=unique_x_vals, y_vals=quantile_y_vals)

    except AttributeError as e:
        error_msg = "Curves must be an iterable of Curve objects."
        logging.error(error_msg)
        raise TypeError(error_msg) from e
    except TypeError as e:
        error_msg = "Beta must be a numeric value (int or float)."
        logging.error(error_msg)
        raise TypeError(error_msg) from e


def cdf_of_curves_crossing_times(
    curves: Iterable[Curve], threshold: float
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
    from bisect import bisect_right

    try:
        # Compute crossing times once (errors will naturally raise if `curves` is invalid)
        crossing_times = [
            curve.compute_crossing_time(threshold) for curve in curves
        ]

        # Collect unique crossing times (excluding infinity)
        finite_crossing_times = {t for t in crossing_times if t < float("inf")}

        # Construct sorted unique x-values with 0 and 1 at the edges
        unique_x_vals = [0, *sorted(finite_crossing_times), 1]

        # Use binary search (`bisect_right`) for efficient cumulative sum calculation
        n_curves = len(crossing_times)
        sorted_crossings = sorted(crossing_times)

        cdf_y_vals = [
            bisect_right(sorted_crossings, x) / n_curves for x in unique_x_vals
        ]

        return Curve(x_vals=unique_x_vals, y_vals=cdf_y_vals)

    except AttributeError as e:
        error_msg = "Curves must be an iterable of Curve objects."
        logging.error(error_msg)
        raise TypeError(error_msg) from e
    except TypeError as e:
        error_msg = "Threshold must be a float."
        logging.error(error_msg)
        raise TypeError(error_msg) from e


def quantile_cross_jump(
    curves: Iterable[Curve], threshold: float, beta: float
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
    from statistics import quantiles

    """Computes the quantile crossing time curve based on the given threshold and beta quantile."""

    try:
        # Compute crossing times once
        crossing_times = [
            curve.compute_crossing_time(threshold=threshold) for curve in curves
        ]

        # Compute quantile using built-in `statistics.quantiles()` instead of `np.quantile()`
        quantile_idx = int(
            beta * 99
        )  # Convert beta into an index (assuming n=100 quantiles)
        solve_time_quantile = quantiles(crossing_times, n=100)[quantile_idx]

        # Handle NaN and infinity cases
        if math.isinf(solve_time_quantile) or math.isnan(solve_time_quantile):
            return Curve(x_vals=[0, 1], y_vals=[0, 0])
        else:
            return Curve(x_vals=[0, solve_time_quantile, 1], y_vals=[0, 1, 1])

    except AttributeError as e:
        error_msg = "Curves must be an iterable of Curve objects."
        logging.error(error_msg)
        raise TypeError(error_msg) from e
    except TypeError as e:
        error_msg = "Threshold and Beta must be numeric (int or float)."
        logging.error(error_msg)
        raise TypeError(error_msg) from e


def difference_of_curves(curve_1: Curve, curve_2: Curve) -> Curve:
    """Compute the difference of two curves (Curve 1 - Curve 2).

    Parameters
    ----------
    curve_1: ``experiment_base.Curve``
        First curve to take the difference of.
    curve_2 : ``experiment_base.Curve``
        Second curve to take the difference of.

    Returns
    -------
    ``experiment_base.Curve``
        Difference of curves.

    Raises
    ------
    TypeError

    """
    try:
        # Collect unique x-values from both curves
        unique_x_vals = sorted(set(curve_1.x_vals) | set(curve_2.x_vals))

        # Compute difference in y-values
        difference_y_vals = [
            curve_1.lookup(x_val) - curve_2.lookup(x_val)
            for x_val in unique_x_vals
        ]

        return Curve(x_vals=unique_x_vals, y_vals=difference_y_vals)

    except AttributeError as e:
        error_msg = "Both curve_1 and curve_2 must be Curve objects."
        logging.error(error_msg)
        raise TypeError(error_msg) from e


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

    try:
        # Compute the difference curve and return the max y-value
        return max(difference_of_curves(curve_1, curve_2).y_vals)

    except AttributeError as e:
        error_msg = "Both curve_1 and curve_2 must be Curve objects."
        logging.error(error_msg)
        raise TypeError(error_msg) from e
