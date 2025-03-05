from typing import TypeVar, Callable, Any

T = TypeVar("T", bound=type)


class ClassPropertyDescriptor:
    def __init__(self, fget: Callable[[type[T]], Any]) -> None:
        """A descriptor that allows class-level attribute access like a property."""
        self.fget = fget

    def __get__(self, instance: Any, owner: type[T]) -> Any:  # noqa: ANN401
        """Retrieve the computed class property when accessed on the class."""
        return self.fget(owner)


def classproperty(func: Callable[[type[T]], Any]) -> ClassPropertyDescriptor:
    """Decorator to create a class property using a descriptor."""
    return ClassPropertyDescriptor(func)
