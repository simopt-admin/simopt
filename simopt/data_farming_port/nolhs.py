"""Module for generating Nearly Orthogonal Latin Hypercube Samples (NOLHS)."""

import bisect
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from simopt.data_farming_port.nolhs_designs import DESIGN_TABLE


class Scaler:
    """Class to scale values from one range to another."""

    @property
    def original_min(self) -> float:
        """The minimum value of the original range."""
        return self._original_min

    @original_min.setter
    def original_min(self, value: float) -> None:
        self._original_min = value
        self._cached_scale = False

    @property
    def original_max(self) -> float:
        """The maximum value of the original range."""
        return self._original_max

    @original_max.setter
    def original_max(self, value: float) -> None:
        self._original_max = value
        self._cached_scale = False

    @property
    def scaled_min(self) -> float:
        """The minimum value of the scaled range."""
        return self._scaled_min

    @scaled_min.setter
    def scaled_min(self, value: float) -> None:
        self._scaled_min = value
        self._cached_scale = False

    @property
    def scaled_max(self) -> float:
        """The maximum value of the scaled range."""
        return self._scaled_max

    @scaled_max.setter
    def scaled_max(self, value: float) -> None:
        self._scaled_max = value
        self._cached_scale = False

    @property
    def precision(self) -> int:
        """The number of decimal places to round to."""
        return self._precision

    @property
    def scale_factor(self) -> int:
        """The factor by which to scale values."""
        return self._scale_factor

    @precision.setter
    def precision(self, value: int) -> None:
        self._precision = value
        self._scale_factor = 10**value
        self._cached_scale = False

    @property
    def scale(self) -> float:
        """The scale factor between the original and scaled ranges."""
        if not self._cached_scale:
            original_range = self.original_max - self.original_min
            scaled_range = self.scaled_max - self.scaled_min
            self._scale = (scaled_range / original_range) if original_range else 0
            self._offset = self.scaled_max - (self.original_min * self._scale)
            self._cached_scale = True
        return self._scale

    def __init__(
        self,
        original_min: float,
        original_max: float,
        scaled_min: float,
        scaled_max: float,
        precision: int = 0,
    ) -> None:
        """Initialize the Scaler.

        Args:
            original_min (float): The minimum value of the original range.
            original_max (float): The maximum value of the original range.
            scaled_min (float): The minimum value of the scaled range.
            scaled_max (float): The maximum value of the scaled range.
            precision (int, optional): The number of decimal places to round to.
                Defaults to 0.
        """
        self._original_min = original_min
        self._original_max = original_max
        self._scaled_min = scaled_min
        self._scaled_max = scaled_max
        self.precision = precision
        self._cached_scale = False

    def scale_value(self, value: float) -> int | float:
        """Scale a value from the original range to the scaled range.

        Returns an int if the value has no decimal component after rounding,
        otherwise returns a float.
        """
        if not (self.original_min <= value <= self.original_max):
            raise ValueError(
                f"Value {value} is out of bounds "
                f"({self.original_min}, {self.original_max})"
            )
        new_value = self.scaled_min + (self.scale * (value - self.original_min))
        rounded_val = round(new_value, self.precision)
        # TODO: simplify int return logic
        return int(rounded_val) if rounded_val == int(rounded_val) else rounded_val


class DesignType(ABC):
    """Abstract base class for design types."""

    @abstractmethod
    def __init__(
        self,
        designs: list[tuple[float, float, int]] | None = None,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the DesignType class."""
        pass

    @abstractmethod
    def generate_design(self) -> list[list[float]]:
        """Generate a design as a 2D list.

        Returns:
            list[list[float]]: A 2D list containing the scaled design points.
        """
        pass

    def save_design(
        self,
        output_file: Path,
    ) -> None:
        """Save the NOLHS design to a file.

        Args:
            output_file (Path): The path to the output file.
        """
        # Generate the design
        scaled_designs = self.generate_design()

        # Remove the output file if it exists
        if output_file.exists():
            output_file.unlink()

        with output_file.open("w") as f:
            for row in scaled_designs:
                f.write("\t".join(map(str, row)) + "\n")


class NOLHS(DesignType):
    """Class to generate Nearly Orthogonal Latin Hypercube Samples (NOLHS)."""

    def __init__(
        self,
        designs: list[tuple[float, float, int]] | None = None,
        num_stacks: int = 1,
    ) -> None:
        """Initialize the NOLHS class."""
        self._set_designs(designs)
        self._num_stacks = num_stacks

    @property
    def num_stacks(self) -> int:
        """Number of stacks in the design."""
        return self._num_stacks

    @num_stacks.setter
    def num_stacks(self, value: int) -> None:
        self._num_stacks = value

    def _set_designs(
        self,
        designs: list[tuple[float, float, int]] | None,
    ) -> None:
        """Set the design configurations.

        Args:
            designs (list[tuple[float, float, int]]): A list of tuples where each tuple
                contains (min, max, precision) for a design variable.
        """
        self.designs = designs or []
        self.design_size = len(self.designs)
        self.nolhs_size = self._determine_table_key(self.design_size)
        self.scalers = [
            Scaler(
                original_min=1,
                original_max=self.nolhs_size,
                scaled_min=min_val,
                scaled_max=max_val,
                precision=num_digits,
            )
            for min_val, max_val, num_digits in self.designs
        ]

    def generate_design(self) -> list[list[float]]:
        """Generate the scaled NOLHS design as a 2D list.

        Returns:
            list[list[float]]: A 2D list containing the scaled design points.
        """
        if not self.designs:
            raise ValueError("Designs have not been set.")

        lh_max = self.nolhs_size
        design_size = self.design_size
        factor = self.scalers
        design = DESIGN_TABLE[lh_max].copy()
        mid_range = lh_max // 2

        all_scaled_designs = []

        for stack_num in range(self._num_stacks):
            for i, dp in enumerate(design):
                scaled_dp = [
                    factor[k].scale_value(x) for k, x in enumerate(dp[:design_size])
                ]

                condition = stack_num > 0 and i == mid_range and lh_max < 512
                if not condition:
                    all_scaled_designs.append(scaled_dp)

                # Rotate the data point for the next iteration
                design[i] = dp[1:] + dp[:1]

        return all_scaled_designs

    def import_design_config(
        self,
        file_path: Path,
    ) -> None:
        """Import the design config from a file.

        Args:
            file_path (Path): The path to the file containing the design config.
        """
        # Read the design config from the specified file.
        design_config = []
        with file_path.open("r") as f:
            for line in f:
                line_data = re.split(r"\s*[,;:]\s*|\s+", line.strip())
                # Skip empty lines
                if len(line_data) == 0:
                    continue
                # Each line must contain exactly three values
                if len(line_data) != 3:
                    raise ValueError(
                        f"Error importing design config at Path: {file_path}. "
                        f"Each line must contain exactly three values: {line.strip()}"
                    )
                # Add the design to the config
                min_val, max_val, num_digits = line_data
                design = (float(min_val), float(max_val), int(num_digits))
                design_config.append(design)
        # Update the class attributes based on the imported design config
        self._set_designs(design_config)

    def _determine_table_key(self, num_vars: int) -> int:
        """Determine the key to use for the design table based on number of variables.

        Args:
            num_vars (int): The number of variables in the optimization problem.

        Returns:
            int: The key to use for the design table.

        Raises:
            ValueError: If num_vars is greater than 100.
        """
        if num_vars > 100:
            raise ValueError("NOLHS only supports up to 100 variables at this time.")

        # Keys are the minimum of each range, values are the return keys
        ranges = {
            1: 17,
            8: 33,
            12: 65,
            17: 129,
            23: 257,
            30: 512,
        }

        # Get a sorted list of the keys (the range minimums)
        min_vars = sorted(ranges.keys())

        # Use bisect to find the correct range
        # It finds the insertion point for num_vars in the sorted list
        idx = bisect.bisect_right(min_vars, num_vars) - 1
        return ranges[min_vars[idx]]
