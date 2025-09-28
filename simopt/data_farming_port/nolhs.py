import re
from pathlib import Path

from simopt.data_farming_port.nolhs_designs import DESIGN_TABLE


class Scaler:
    def __init__(self, min, max, decimals, lh_max=17):
        self.min = min
        self.range = (max - min) / (lh_max - 1)
        self.scale_factor = 10**decimals

    def scale(self, value):
        new_value = self.min + self.range * (value - 1)
        if self.scale_factor == 1:
            return round(new_value)
        return round(new_value * self.scale_factor) / self.scale_factor


def save_output(
    lh_max: int,
    num_stacks: int,
    design_size: int,
    factor: list[Scaler],
    output_file: Path,
):
    def rotate(dp: list) -> list:
        return dp[1:] + dp[:1]

    design = DESIGN_TABLE[lh_max]
    num_stacks = num_stacks or len(design[0])

    mid_range = lh_max // 2
    with output_file.open("w") as f:
        for stack_num in range(num_stacks):
            for i, dp in enumerate(design):
                # Slice the data point and scale each element
                scaled_dp = []
                for k, x in enumerate(dp[:design_size]):
                    scaled_dp.append(factor[k].scale(x))

                # Join the scaled values with tabs and print conditionally
                condition = stack_num > 0 and i == mid_range and lh_max < 512
                if not condition:
                    f.write("\t".join(map(str, scaled_dp)) + "\n")

                # Rotate the data point for the next iteration
                design[i] = rotate(dp)


def import_design_table_from_file(file_path: Path) -> list[tuple[float, float, int]]:
    """Import design table from a file.

    Args:
        file_path (Path): The path to the file containing the design table.

    Returns:
        list[tuple[float, float, int]]: The imported design table (min, max, decimals).
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
    return design_table


def determine_table_key(num_vars: int) -> int:
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
