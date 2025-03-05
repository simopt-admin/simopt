from __future__ import annotations

import logging
import math
from enum import Enum
from typing import TYPE_CHECKING

# Imports exclusively used when type checking
# Prevents imports from being executed at runtime
if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.lines import Line2D as Line2D


class CurveType(Enum):
    """Enumeration for different curve styles."""

    REGULAR = "regular"
    CONF_BOUND = "conf_bound"

    @property
    def style(self) -> tuple[str, int]:
        """Returns linestyle and linewidth for the curve type."""
        return {
            CurveType.REGULAR: ("-", 2),
            CurveType.CONF_BOUND: ("--", 1),
        }[self]


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
    def x_vals(self) -> tuple[float, ...]:
        """Values of horizontal components."""
        return self.__x_vals

    @property
    def y_vals(self) -> tuple[float, ...]:
        """Values of vertical components."""
        return self.__y_vals

    @property
    def n_points(self) -> int:
        """Number of points in the curve."""
        return self.__n_points

    def __init__(
        self, x_vals: Sequence[int | float], y_vals: Sequence[int | float]
    ) -> None:
        """Initialize a curve with x- and y-values.

        Parameters
        ----------
        x_vals : Sequence[int | float]
            Values of horizontal components.
        y_vals : Sequence[int | float]
            Values of vertical components.

        Raises
        ------
        TypeError
        ValueError
        """
        try:
            # Ensure x_vals and y_vals have the same length before conversion
            if len(x_vals) != len(y_vals):
                error_msg = f"Length of x ({len(x_vals)}) and y ({len(y_vals)}) must be equal."
                raise ValueError(error_msg)

            # Convert to immutable tuples only after validation
            self.__x_vals = tuple(float(x) for x in x_vals)
            self.__y_vals = tuple(float(y) for y in y_vals)

            # Store the number of points
            self.__n_points = len(self.__x_vals)

        except (TypeError, ValueError) as e:
            error_msg = f"Invalid input for Curve initialization: {e}"
            logging.error(error_msg)
            raise ValueError(error_msg) from e  # Keep the original error type

    def lookup(self, x_val: float) -> float:
        """Lookup the y-value of the curve at an intermediate x-value.

        Parameters
        ----------
        x_val : float
            X-value at which to lookup the y-value.

        Returns
        -------
        float
            Y-value corresponding to x, or NaN if x_val is out of range.

        Raises
        ------
        TypeError
            If x_val is not numeric.
        """
        from bisect import bisect_right

        try:
            # Return NaN if x_val is out of range (before first or after last x-value)
            if x_val < self.x_vals[0] or x_val > self.x_vals[-1]:
                return math.nan

            # Use binary search (O(log n)) instead of linear search (O(n))
            idx = bisect_right(self.x_vals, x_val) - 1
            return self.y_vals[idx]

        except TypeError as e:
            raise TypeError(f"x_val must be a numeric value: {e}") from e

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
            If threshold is not numeric.
        """
        from bisect import bisect_right

        try:
            # Find the first index where y_vals < threshold using binary search
            index = bisect_right(self.y_vals, threshold)

            # If all y-values are above the threshold, return infinity
            if index == self.n_points:
                return math.inf

            # Return corresponding x-value
            return self.x_vals[index]

        except TypeError as e:
            raise TypeError(f"Threshold must be a numeric value: {e}") from e

    def compute_area_under_curve(self) -> float:
        """Compute the area under a curve.

        Returns
        -------
        float
            Area under the curve.
        """
        x_diffs = (
            x_next - x for x, x_next in zip(self.x_vals[:-1], self.x_vals[1:])
        )
        area_contributions = (
            y * dx for y, dx in zip(self.y_vals[:-1], x_diffs)
        )

        return sum(area_contributions)

    def curve_to_mesh(self, mesh: Iterable[float]) -> Curve:
        """Create a curve defined at equally spaced x values.

        Parameters
        ----------
        mesh : Iterable[float]
            Collection of uniformly spaced x-values.

        Returns
        -------
        ``experiment_base.Curve``
            Curve with equally spaced x-values.

        Raises
        ------
        TypeError
            If mesh is not an iterable of numeric values.
        """
        try:
            # Ensure mesh contains valid numeric values
            mesh_x_vals = tuple(float(x) for x in mesh)

            # Generate corresponding y-values using lookup
            mesh_y_vals = tuple(self.lookup(x) for x in mesh_x_vals)

            return Curve(x_vals=mesh_x_vals, y_vals=mesh_y_vals)

        except (TypeError, ValueError) as e:
            error_msg = "Mesh must be an iterable of numeric values."
            logging.error(error_msg)
            raise TypeError(error_msg) from e

    def curve_to_full_curve(self) -> Curve:
        """Create a curve with duplicate x- and y-values to indicate steps.

        Returns
        -------
        ``experiment_base.Curve``
            Curve with duplicate x- and y-values.

        """
        from itertools import chain, repeat

        full_curve = Curve(
            x_vals=chain.from_iterable(repeat(x, 2) for x in self.x_vals),
            y_vals=chain.from_iterable(repeat(y, 2) for y in self.y_vals),
        )
        return Curve(
            x_vals=list(full_curve.x_vals)[1:],
            y_vals=list(full_curve.y_vals)[:-1],
        )

    def plot(
        self,
        color_str: str = "C0",
        curve_type: CurveType = CurveType.REGULAR,
    ) -> Line2D:
        """Plot a curve.

        Parameters
        ----------
        color_str : str, default="C0"
            String indicating line color, e.g., "C0", "C1", etc.
        curve_type : CurveType, default=CurveType.REGULAR
            Type of line: REGULAR (solid) or CONF_BOUND (dashed).

        Returns
        -------
        matplotlib.lines.Line2D
            Curve handle, to use when creating legends.

        Raises
        ------
        ValueError
            If an invalid curve type is provided.
        """
        from matplotlib.pyplot import step

        try:
            # Ensure curve_type is a valid Enum member
            if not isinstance(curve_type, CurveType):
                error_msg = f"Invalid curve type: {curve_type}. Must be a member of CurveType."
                raise ValueError(error_msg)

            linestyle, linewidth = curve_type.style

            # Plot the step curve
            handle = step(
                self.x_vals,
                self.y_vals,
                color=color_str,
                linestyle=linestyle,
                linewidth=linewidth,
                where="post",
            )[0]

            return handle

        except Exception as e:
            error_msg = f"Error in plot function: {e}"
            logging.error(error_msg)
            raise
