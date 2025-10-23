"""Core module for creating data farming designs."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


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
        """Save the design to a file.

        Args:
            output_file (Path): The path to the output file.
        """
        # Generate the design
        design = self.generate_design()

        # Ensure the directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Remove the output file if it exists
        if output_file.exists():
            output_file.unlink()

        with output_file.open("w") as f:
            for row in design:
                f.write("\t".join(map(str, row)) + "\n")
