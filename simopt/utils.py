from typing import Optional, TypeVar, Callable, Generic

T = TypeVar("T", bound=type)
R = TypeVar("R")


class ClassPropertyDescriptor(Generic[T, R]):
    def __init__(self, fget: Callable[[type[T]], R]) -> None:
        self.fget = fget

    def __get__(self, instance: Optional[object], owner: type[T]) -> R:
        return self.fget(owner)


def classproperty(
    func: Callable[[type[T]], R],
) -> ClassPropertyDescriptor[T, R]:
    return ClassPropertyDescriptor(func)


def make_nonzero(value: float, name: str, epsilon: float = 1e-15) -> float:
    """Return a non-zero value to avoid division by zero.

    Arguments
    ---------
    value : float
        The value to check.
    name : str
        The name of the variable.
    epsilon : float, optional (default=1e-15)
        The value to use if the original value is zero.

    Returns
    -------
    float
        The original value if it's not close to zero, otherwise a non-zero value.
    """
    # Delayed imports
    import numpy as np

    # If it's not close to 0, return the original value
    if not np.isclose(value, 0, atol=epsilon):
        return value

    # Otherwise, calculate the new value
    import logging

    new_value = epsilon if value == 0 else np.sign(value) * epsilon
    warning_msg = (
        f"{name} is {value}. Setting to {new_value} to avoid division by zero."
    )
    logging.warning(warning_msg)
    return new_value
