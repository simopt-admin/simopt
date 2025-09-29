# noqa: D100
import re
from pathlib import Path

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
    ):
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

    def scale_value(self, value: float) -> float:
        """Scale a value from the original range to the scaled range."""
        if not (self.original_min <= value <= self.original_max):
            raise ValueError(
                f"Value {value} is out of bounds "
                f"({self.original_min}, {self.original_max})"
            )
        new_value = self.scaled_min + (self.scale * (value - self.original_min))
        rounded_val = round(new_value, self.precision)
        # TODO: simplify int return logic
        return int(rounded_val) if rounded_val == int(rounded_val) else rounded_val


class NOLHS:
    def __init__(self) -> None:
        self.designs: list[tuple[float, float, int]] = []
        self.design_size = 0
        self.nolhs_size = 0
        self.scalers: list[Scaler] = []
        self._num_stacks = len(self.designs[0]) if self.designs else 1

    @property
    def num_stacks(self) -> int:
        """Number of stacks in the design."""
        return self._num_stacks

    @num_stacks.setter
    def num_stacks(self, value: int) -> None:
        self._num_stacks = value

    def save_output(
        self,
        output_file: Path,
    ) -> None:
        """Save the NOLHS design to a file."""

        def rotate(dp: list) -> list:
            return dp[1:] + dp[:1]

        lh_max = self.nolhs_size
        design_size = self.design_size
        factor = self.scalers

        design = DESIGN_TABLE[lh_max]

        mid_range = lh_max // 2

        # Remove the output file if it exists
        if output_file.exists():
            output_file.unlink()
        with output_file.open("w") as f:
            for stack_num in range(self._num_stacks):
                for i, dp in enumerate(design):
                    # Slice the data point and scale each element
                    scaled_dp = []
                    for k, x in enumerate(dp[:design_size]):
                        scaled_dp.append(factor[k].scale_value(x))

                    # Join the scaled values with tabs and print conditionally
                    condition = stack_num > 0 and i == mid_range and lh_max < 512
                    if not condition:
                        f.write("\t".join(map(str, scaled_dp)) + "\n")

                    # Rotate the data point for the next iteration
                    design[i] = rotate(dp)
        pass

    def import_design_table_from_file(
        self,
        file_path: Path,
    ) -> None:
        """Import design table from a file.

        Args:
            file_path (Path): The path to the file containing the design table.
        """
        design_table = []
        with file_path.open("r") as f:
            for line in f:
                line_data = re.split(r"\s*[,;:]\s*|\s+", line.strip())
                # Skip empty lines
                if len(line_data) == 0:
                    continue
                # Each line must contain exactly three values
                if len(line_data) != 3:
                    raise ValueError(
                        f"Each line must contain exactly three values: {line.strip()}"
                    )
                # Add the design to the table
                min_val, max_val, num_digits = line_data
                design = (float(min_val), float(max_val), int(num_digits))
                design_table.append(design)
        self.designs = design_table
        self.design_size = len(design_table)
        self.nolhs_size = self._determine_table_key(self.design_size)
        self.scalers = [
            Scaler(
                original_min=1,
                original_max=self.nolhs_size,
                scaled_min=min_val,
                scaled_max=max_val,
                precision=num_digits,
            )
            for min_val, max_val, num_digits in design_table
        ]

    def _determine_table_key(self, num_vars: int) -> int:
        """Determine the key to use for the design table based on the number of variables.

        Args:
            num_vars (int): The number of variables in the optimization problem.

        Returns:
            int: The key to use for the design table.
        """
        if 1 <= num_vars <= 7:
            return 17
        if 8 <= num_vars <= 11:
            return 33
        if 12 <= num_vars <= 16:
            return 65
        if 17 <= num_vars <= 22:
            return 129
        if 23 <= num_vars <= 29:
            return 257
        if 30 <= num_vars <= 100:
            return 512
        raise ValueError("Number of variables must be between 1 and 100.")
