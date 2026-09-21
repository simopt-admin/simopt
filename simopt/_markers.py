"""Identity markers for simulations and input models."""

from typing import TypeVar

T = TypeVar("T")


def simulation(function: T) -> T:
    """Mark a simulation without changing its Python behavior."""
    return function


def input_model(cls: type[T]) -> type[T]:
    """Mark an input model without changing its Python behavior."""
    return cls
