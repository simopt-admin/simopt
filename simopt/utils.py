"""Utility functions for simopt."""

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

T = TypeVar("T", bound=type)
R = TypeVar("R")


class ClassPropertyDescriptor(Generic[T, R]):
    """Descriptor for class properties."""

    def __init__(self, fget: Callable[[type[T]], R]) -> None:
        """Initialize the descriptor.

        Args:
            fget (Callable[[type[T]], R]): The function to get the class property.
        """
        self.fget = fget

    def __get__(self, instance: object | None, owner: type[T]) -> R:
        """Get the class property.

        Args:
            instance (Optional[object]): The instance of the class.
            owner (type[T]): The class itself.

        Returns:
            R: The value of the class property.
        """
        return self.fget(owner)


def classproperty(
    func: Callable[[type[T]], R],
) -> ClassPropertyDescriptor[T, R]:
    """Decorator to define a class property.

    Args:
        func (Callable[[type[T]], R]): The function to be decorated.

    Returns:
        ClassPropertyDescriptor[T, R]: The class property descriptor.
    """
    return ClassPropertyDescriptor(func)


def override(obj: T) -> T:
    """Decorator to mark a method as overridden."""
    obj.__override__ = True
    return obj


def make_nonzero(value: float, name: str, epsilon: float = 1e-15) -> float:
    """Return a non-zero value to avoid division by zero.

    Args:
        value (float): The value to check.
        name (str): The name of the variable.
        epsilon (float, optional): The value to use if the original value is zero.
            Default is 1e-15.

    Returns:
        float: The original value if it's not close to zero, otherwise a non-zero value.
    """
    # Delayed import to avoid lagging when importing the module
    import numpy as np

    # If it's not close to 0, return the original value
    if not np.isclose(value, 0, atol=epsilon):
        return value

    # Delayed import to avoid lagging when importing the module
    import logging
    from math import copysign

    # If it's close to 0, return a non-zero value
    # Use the sign of the original value to help determine the new value
    new_value = copysign(epsilon, value)
    warning_msg = (
        f"{name} is {value}. Setting to {new_value} to avoid division by zero."
    )
    logging.warning(warning_msg)
    return new_value


def resolve_file_path(target: str | Path, directory: str | Path) -> Path:
    """Resolve a file path against a base directory.

    Args:
        target (str | Path): The target file path to resolve.
        directory (str | Path): The base directory to resolve against.

    Returns:
        Path: The resolved file path.

    Raises:
        ValueError: If the target is a directory.
    """
    # If the target is a directory, raise an error
    if Path(target).is_dir():
        raise ValueError(f"Target {target} is a directory, not a file.")
    # If it's already a Path object, resolve it directly
    if isinstance(target, Path):
        return target.resolve()
    # Otherwise, we know it's a string
    # We need to check if it's a fully qualified path or a relative path
    if Path(target).is_absolute():
        return Path(target).resolve()
    # If it's a relative path, resolve it against the directory
    return (Path(directory) / target).resolve()


def print_table(name: str, headers: list[str], data: list[tuple] | dict) -> None:
    """Print a table with headers and data.

    Args:
        name (str): Name of the table.
        headers (list[str]): List of column headers.
        data (list[tuple]): List of rows, each row is a tuple of values.
    """
    # Convert data out of dict (if necessary)
    if isinstance(data, dict):
        data = list(data.items())
    # Calculate the maximum length of each column
    data_widths = [
        max(len(str(item)) for item in col) for col in zip(*data, strict=False)
    ]
    header_widths = [len(header) for header in headers]
    max_widths = [
        max(header_width, col_width)
        for header_width, col_width in zip(header_widths, data_widths, strict=False)
    ]

    # Compute total width of the table
    # There's 3 separator characters between each column
    separator_lengths = 3 * (len(headers) - 1)
    total_width = sum(max_widths) + separator_lengths
    # If table is shorter than name, expand last column
    if total_width < len(name):
        shortfall = len(name) - total_width
        max_widths[-1] += shortfall
        total_width = len(name)

    # Center title in the table
    title_indent_count = (total_width - len(name)) // 2
    title_lead = " " * title_indent_count
    title_follow = " " * (total_width - title_indent_count - len(name))
    title = f"{title_lead}{name}{title_follow}"

    if sys.stdout.isatty():
        # Unicode box-drawing characters
        corner_tl = "┌"
        corner_tr = "┐"
        corner_bl = "└"
        corner_br = "┘"
        tee_left = "├"
        tee_right = "┤"
        dash = "─"
        plus = f"{dash}┼{dash}"
        pipe = "│"

        reset = "\033[0m"
        bg_grey = "\033[48;5;235m"
        bg_black = "\033[48;5;0m"
        fg_white = "\033[38;5;252m"
    else:
        # ASCII fallback
        corner_tl = "+"
        corner_tr = "+"
        corner_bl = "+"
        corner_br = "+"
        tee_left = "+"
        tee_right = "+"
        dash = "-"
        plus = "-+-"
        pipe = "|"

        reset = ""
        bg_grey = ""
        bg_black = ""
        fg_white = ""

    underline_row = dash * (total_width + 2)  # Extend to the tees
    header_row = f" {pipe} ".join(
        f"{header:<{width}}" for header, width in zip(headers, max_widths, strict=False)
    )
    sep_row = plus.join(dash * width for width in max_widths)
    rows = []
    for row in data:
        row_str = f" {pipe} ".join(
            f"{item!s:>{width}}"
            if isinstance(item, (int, float))
            else f"{item!s:<{width}}"
            for item, width in zip(row, max_widths, strict=False)
        )
        rows.append(row_str)

    border_width = total_width + 2  # Extend 2 extra spaces

    # Print the table
    print(f"{bg_black}{fg_white}", end="")  # Override background and foreground colors
    print(f"{corner_tl}{dash * border_width}{corner_tr}")
    print(f"{pipe} {title} {pipe}")
    print(f"{tee_left}{underline_row}{tee_right}")
    print(f"{pipe} {header_row} {pipe}")
    print(f"{pipe} {sep_row} {pipe}")
    for i, row in enumerate(rows):
        row_bg = bg_grey if i % 2 else bg_black
        print(f"{bg_black}{fg_white}{pipe}{row_bg} {row} {bg_black}{pipe}{reset}")
    print(f"{corner_bl}{dash * border_width}{corner_br}")
    print(reset, end="")  # Reset colors


def get_specifications(config_class: type[BaseModel]) -> dict[str, dict]:
    """Get the specifications for a configuration class."""
    spec = {}

    for name, field in config_class.model_fields.items():
        data = {
            "description": field.description,
        }

        datatype = field.annotation

        # `field.default` can be missing when `default_factory` is used
        default = (
            field.default
            if field.default is not PydanticUndefined
            else field.default_factory()
        )

        # Handle data type like list[int]
        if isinstance(datatype, types.GenericAlias):
            datatype = datatype.__origin__
            # NOTE: the GUI only supports lists
            if datatype is tuple:
                datatype = list
                default = list(default)

        data["datatype"] = datatype
        data["default"] = default

        if (
            field.json_schema_extra is not None
            and "isDatafarmable" in field.json_schema_extra
        ):
            data["isDatafarmable"] = field.json_schema_extra["isDatafarmable"]

        name = field.alias or name
        spec[name] = data

    return spec
