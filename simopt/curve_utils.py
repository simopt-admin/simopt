"""Curve utility functions.

This module provides utility functions for manipulating and analyzing curves.
"""

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
    """Compute the pointwise mean of a collection of curves.

    All curves must have identical starting and ending x-values.

    Args:
        curves (Iterable[Curve]): A collection of curves to aggregate.

    Returns:
        Curve: A curve representing the pointwise mean across all input curves.

    Raises:
        TypeError: If the input is not an iterable of Curve objects.
    """
    from statistics import mean

    try:
        # Collect unique x-values across all curves
        unique_x_vals = sorted({x_val for curve in curves for x_val in curve.x_vals})

        # Compute pointwise means using generator expressions
        mean_y_vals = [
            mean(curve.lookup(x_val) for curve in curves) for x_val in unique_x_vals
        ]

        return Curve(x_vals=unique_x_vals, y_vals=mean_y_vals)

    except AttributeError as e:
        error_msg = "Curves must be an iterable of Curve objects."
        logging.error(error_msg)
        raise TypeError(error_msg) from e


def quantile_of_curves(curves: Iterable[Curve], beta: float) -> Curve:
    """Compute the pointwise quantile of a collection of curves.

    All curves must have identical starting and ending x-values.

    Args:
        curves (Iterable[Curve]): A collection of curves to aggregate.
        beta (float): The quantile level to compute (e.g., 0.5 for median).

    Returns:
        Curve: A curve representing the pointwise quantile across the input curves.

    Raises:
        TypeError: If input is not a valid collection of Curve objects.
    """
    from statistics import quantiles

    try:
        # Collect unique x-values across all curves
        unique_x_vals = sorted({x_val for curve in curves for x_val in curve.x_vals})

        # Precompute quantile index
        quantile_idx = int(beta * 99)

        # Compute pointwise quantiles
        quantile_y_vals = [
            quantiles((curve.lookup(x_val) for curve in curves), n=100)[quantile_idx]
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


def cdf_of_curves_crossing_times(curves: Iterable[Curve], threshold: float) -> Curve:
    """Compute the CDF of crossing times from a collection of curves.

    The crossing time is defined as the first x-value where a curve crosses a given
    threshold.

    Args:
        curves (list[Curve]): A list of curves to analyze.
        threshold (float): The y-value at which to detect the first crossing.

    Returns:
        Curve: A curve representing the cumulative distribution of crossing times.

    Raises:
        TypeError: If input is not a list of Curve objects.
    """
    from bisect import bisect_right

    try:
        # Compute crossing times once
        crossing_times = [curve.compute_crossing_time(threshold) for curve in curves]

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
    """Compute a curve with a jump at the quantile of the crossing times.

    The curve is piecewise-constant with a single jump located at the specified quantile
    of the first crossing times across the input curves.

    Args:
        curves (list[Curve]): A list of curves to analyze.
        threshold (float): The y-value at which to detect the first crossing.
        beta (float): The quantile level (e.g., 0.5 for median crossing time).

    Returns:
        Curve: A piecewise-constant curve with a jump at the quantile crossing time,
        if finite.

    Raises:
        TypeError: If input types are incorrect.
    """
    from statistics import quantiles

    try:
        # Compute crossing times once
        crossing_times = [
            curve.compute_crossing_time(threshold=threshold) for curve in curves
        ]

        # Convert beta into an index (assuming n=100 quantiles)
        quantile_idx = int(beta * 99)
        solve_time_quantile = quantiles(crossing_times, n=100)[quantile_idx]

        # Handle NaN and infinity cases
        if math.isinf(solve_time_quantile) or math.isnan(solve_time_quantile):
            return Curve(x_vals=[0, 1], y_vals=[0, 0])
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
    """Compute the difference between two curves (curve_1 - curve_2).

    The x-values of both curves must align exactly.

    Args:
        curve_1 (Curve): The first curve (minuend).
        curve_2 (Curve): The second curve (subtrahend).

    Returns:
        Curve: A curve representing the pointwise difference.

    Raises:
        TypeError: If inputs are not Curve instances or are incompatible.
    """
    try:
        # Collect unique x-values from both curves
        unique_x_vals = sorted(set(curve_1.x_vals) | set(curve_2.x_vals))

        # Compute difference in y-values
        difference_y_vals = [
            curve_1.lookup(x_val) - curve_2.lookup(x_val) for x_val in unique_x_vals
        ]

        return Curve(x_vals=unique_x_vals, y_vals=difference_y_vals)

    except AttributeError as e:
        error_msg = "Both curve_1 and curve_2 must be Curve objects."
        logging.error(error_msg)
        raise TypeError(error_msg) from e


def max_difference_of_curves(curve_1: Curve, curve_2: Curve) -> float:
    """Compute the maximum pointwise difference between two curves (curve_1 - curve_2).

    Args:
        curve_1 (Curve): The first curve (minuend).
        curve_2 (Curve): The second curve (subtrahend).

    Returns:
        float: The maximum difference between the two curves at any x-value.

    Raises:
        TypeError: If the inputs are not Curve instances or are incompatible.
    """
    try:
        # Compute the difference curve and return the max y-value
        return max(difference_of_curves(curve_1, curve_2).y_vals)

    except AttributeError as e:
        error_msg = "Both curve_1 and curve_2 must be Curve objects."
        logging.error(error_msg)
        raise TypeError(error_msg) from e
