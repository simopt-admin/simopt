"""Internal helpers shared by solvers."""

import types

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


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
    warning_msg = f"{name} is {value}. Setting to {new_value} to avoid division by zero."
    logging.warning(warning_msg)
    return new_value


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
            field.default if field.default is not PydanticUndefined else field.default_factory()
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

        if field.json_schema_extra is not None and "isDatafarmable" in field.json_schema_extra:
            data["isDatafarmable"] = field.json_schema_extra["isDatafarmable"]

        name = field.alias or name
        spec[name] = data

    return spec
