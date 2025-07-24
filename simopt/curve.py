"""Curve class for plotting and analysis."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

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
    """Base class for all curves."""

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

        Args:
            x_vals (Sequence[int | float]): Values of horizontal components.
            y_vals (Sequence[int | float]): Values of vertical components.

        Raises:
            TypeError: If x_vals or y_vals are not numeric.
            ValueError: If x_vals and y_vals have different lengths or if they contain
                non-numeric values.
        """
        try:
            # Ensure x_vals and y_vals have the same length before conversion
            if len(x_vals) != len(y_vals):
                error_msg = (
                    f"Length of x ({len(x_vals)}) and y ({len(y_vals)}) must be equal."
                )
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

        Args:
            x_val (float): X-value to lookup.

        Returns:
            float: Y-value corresponding to x, or NaN if x_val is out of range.

        Raises:
            TypeError: If x_val is not numeric.
        """
        # Type checking
        if not isinstance(x_val, (int, float)):
            error_msg = "x_val must be a float."
            raise TypeError(error_msg)

        if x_val < self.x_vals[0]:
            return np.nan
        idx = np.max(np.where(np.array(self.x_vals) <= x_val))
        return self.y_vals[idx]

    def compute_crossing_time(self, threshold: float) -> float:
        """Compute the first time at which a curve drops below a given threshold.

        Args:
            threshold (float): Value for which to find the first crossing time.

        Returns:
            float: First time at which the curve drops below the threshold.

        Raises:
            TypeError: If threshold is not numeric.
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

        Returns:
            float: Area under the curve.
        """
        area = np.dot(self.y_vals[:-1], np.diff(self.x_vals))
        return area

    def curve_to_mesh(self, mesh: Iterable[float]) -> Curve:
        """Create a curve defined at equally spaced x values.

        Args:
            mesh (Iterable[float]): Collection of uniformly spaced x-values.

        Returns:
            Curve: A curve with equally spaced x-values.

        Raises:
            TypeError: If mesh is not an iterable of numeric values.
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

        Returns:
            Curve: A curve with duplicate x- and y-values.
        """
        duplicate_x_vals = [x for x in self.x_vals for _ in (0, 1)]
        duplicate_y_vals = [y for y in self.y_vals for _ in (0, 1)]
        full_curve = Curve(x_vals=duplicate_x_vals[1:], y_vals=duplicate_y_vals[:-1])
        return full_curve

    def plot(
        self,
        color_str: str = "C0",
        curve_type: CurveType = CurveType.REGULAR,
    ) -> Line2D:
        """Plot a curve.

        Args:
            color_str (str): String indicating line color, e.g., "C0", "C1", etc.
            curve_type (CurveType): Type of line.

        Returns:
            Line2D: Curve handle, useful when creating legends.

        Raises:
            ValueError: If an invalid curve type is provided.
        """
        from matplotlib.pyplot import step

        try:
            # Ensure curve_type is a valid Enum member
            if not isinstance(curve_type, CurveType):
                error_msg = (
                    f"Invalid curve type: {curve_type}. Must be a member of CurveType."
                )
                raise ValueError(error_msg)

            linestyle, linewidth = curve_type.style

            # Plot the step curve
            return step(
                self.x_vals,
                self.y_vals,
                color=color_str,
                linestyle=linestyle,
                linewidth=linewidth,
                where="post",
            )[0]

        except Exception as e:
            error_msg = f"Error in plot function: {e}"
            logging.error(error_msg)
            raise
