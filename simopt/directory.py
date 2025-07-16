#!/usr/bin/env python
"""Provide dynamically discovered dictionary directories.

Lists solvers, problems, and models found in the SimOpt package.
"""

import importlib
import pkgutil

import simopt.models
import simopt.solvers
from simopt.base import Model, Problem, Solver


def load_models_and_problems() -> (
    tuple[dict[str, type[Model]], dict[str, type[Problem]]]
):
    """Dynamically load models and problems from simopt.models."""
    models = {}
    problems = {}

    for _, modname, _ in pkgutil.iter_modules(
        simopt.models.__path__, simopt.models.__name__ + "."
    ):
        mod = importlib.import_module(modname)
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type):
                if issubclass(attr, Model) and attr is not Model:
                    models[attr.class_name_abbr] = attr
                elif issubclass(attr, Problem) and attr is not Problem:
                    problems[attr.class_name_abbr] = attr

    return models, problems


def load_solvers() -> dict[str, type[Solver]]:
    """Dynamically load solvers from simopt.solvers."""
    solvers = {}

    for _, modname, _ in pkgutil.iter_modules(
        simopt.solvers.__path__, simopt.solvers.__name__ + "."
    ):
        mod = importlib.import_module(modname)
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Solver)
                and attr is not Solver
            ):
                solvers[attr.class_name_abbr] = attr

    return solvers


def generate_unabbreviated_mapping(
    class_dict: dict[str, type], include_compatibility: bool = False
) -> dict[str, type]:
    """Generate dictionary with full names (and compatibility for solvers/problems)."""
    mapping = {}
    for cls in class_dict.values():
        if include_compatibility:
            mapping[f"{cls.class_name} ({cls.compatibility})"] = cls
        else:
            mapping[cls.class_name] = cls
    return mapping


# Load classes dynamically
model_directory, problem_directory = load_models_and_problems()
solver_directory = load_solvers()

# Generate unabbreviated mappings
solver_unabbreviated_directory = generate_unabbreviated_mapping(
    solver_directory, include_compatibility=True
)
problem_unabbreviated_directory = generate_unabbreviated_mapping(
    problem_directory, include_compatibility=True
)
model_unabbreviated_directory = generate_unabbreviated_mapping(model_directory)
