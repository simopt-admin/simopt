"""Module for generating Nearly Orthogonal Latin Hypercube Samples (NOLHS)."""

import bisect
import re
from pathlib import Path

import numpy as np

from simopt.data_farming.data_farming_core import DesignType, Scaler
from simopt.utils import classproperty


class NOLHS(DesignType):
    """Class to generate Nearly Orthogonal Latin Hypercube Samples (NOLHS)."""

    # Store the design table as a class property so it's cached across instances
    _design_table: dict[int, np.ndarray] | None = None

    def __init__(
        self,
        designs: list[tuple[float, float, int]] | Path,
        num_stacks: int = 1,
    ) -> None:
        """Initialize the NOLHS class.

        Args:
            designs (list[tuple[float, float, int]] | Path): A list of tuples where each
                tuple contains (min, max, precision) for a design variable, or a Path to
                a file containing the design config.
            num_stacks (int, optional): The number of stacks to generate. Defaults to 1.
        """
        if isinstance(designs, Path):
            self._import_design_config(designs)
        else:
            self._set_designs(designs)

        self.num_stacks = num_stacks

    # Store the design table as a class property so it's cached across instances
    # Original design table found here:
    # https://gitlab.nps.edu/pjsanche/datafarmingrubyscripts/-/blob/v1.4.1/lib/datafarming/nolh_designs.rb
    @classproperty
    def design_table(cls) -> dict[int, np.ndarray]:
        """The NOLHS design table.

        Returns:
            dict[int, np.ndarray]: A dictionary mapping the number of variables to
                the corresponding NOLHS design matrix.
        """
        if cls._design_table is None:
            current_dir = Path(__file__).parent
            loaded_data = np.load(current_dir / "nolhs_design_table.npz")
            cls._design_table = {int(k): v for k, v in loaded_data.items()}
        return cls._design_table

    @property
    def num_stacks(self) -> int:
        """Number of stacks in the design."""
        return self._num_stacks

    @num_stacks.setter
    def num_stacks(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Number of stacks must be positive.")
        if self._nolhs_size and value > self._nolhs_size:
            plural = "s" if self._design_size > 1 else ""
            raise ValueError(
                "Number of stacks cannot exceed the NOLHS size "
                f"({self._nolhs_size} for {self._design_size} design variable{plural})."
            )
        self._num_stacks = value

    def _set_designs(
        self,
        designs: list[tuple[float, float, int]],
    ) -> None:
        """Set the design configurations.

        Args:
            designs (list[tuple[float, float, int]]): A list of tuples where each tuple
                contains (min, max, precision) for a design variable.
        """
        self._designs = designs
        self._design_size = len(self._designs)
        self._nolhs_size = self._determine_table_key(self._design_size)
        self._scalers = [
            Scaler(
                original_min=1,
                original_max=self._nolhs_size,
                scaled_min=min_val,
                scaled_max=max_val,
                precision=num_digits,
            )
            for min_val, max_val, num_digits in self._designs
        ]

    def generate_design(self) -> list[list[float]]:
        """Generate the scaled NOLHS design as a 2D list.

        Returns:
            list[list[float]]: A 2D list containing the scaled design points.
        """
        # If there are no design variables, return an empty list
        if not self._nolhs_size or self._design_size == 0:
            return []
        # Copy the design to avoid modifying the original
        design: np.ndarray = self.design_table[self._nolhs_size].copy()
        mid_range = self._nolhs_size // 2

        all_scaled_designs = []

        for stack_idx in range(self._num_stacks):
            for i, dp in enumerate(design):
                scaled_dp = [
                    self._scalers[k].scale_value(x)
                    for k, x in enumerate(dp[: self._design_size])
                ]

                # TODO: revisit why this is needed
                if not (stack_idx > 0 and i == mid_range and self._nolhs_size < 512):
                    all_scaled_designs.append(scaled_dp)

                # Rotate the data point for the next iteration
                design[i] = np.roll(dp, -1)

        return all_scaled_designs

    def _import_design_config(
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
            0: 1,  # Special case for 0 variables
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
