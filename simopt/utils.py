"""Utility functions for simopt."""

from pathlib import Path
from typing import Callable, Generic, Optional, TypeVar

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

    def __get__(self, instance: Optional[object], owner: type[T]) -> R:
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
