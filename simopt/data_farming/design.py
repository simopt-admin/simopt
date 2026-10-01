"""GUI-independent generation of data-farming designs."""

import copy
import itertools
from typing import Any, Literal

from pydantic import BaseModel

from simopt.data_farming.nolhs import NOLHS
from simopt.directory import model_directory, problem_directory, solver_directory

_DIRECTORIES = {
    "solver": solver_directory,
    "problem": problem_directory,
    "model": model_directory,
}


class NumericRange(BaseModel):
    """Range and rounding precision of a numeric factor."""

    min: float
    max: float
    decimals: int = 0


class DesignSpec(BaseModel):
    """Specification of a data-farming design."""

    kind: Literal["solver", "problem", "model"]
    name: str
    varied: dict[str, NumericRange] = {}
    crossed: dict[str, list[bool]] = {}
    fixed: dict[str, Any] = {}
    design_type: Literal["nolhs"] = "nolhs"
    n_stacks: int = 1


def _get_specifications(spec: DesignSpec) -> dict[str, dict]:
    """Return the factor specifications of the named class (plus model factors for a problem)."""
    directory = _DIRECTORIES[spec.kind]
    if spec.name not in directory:
        raise ValueError(f"{spec.kind.capitalize()} '{spec.name}' not found.")
    cls = directory[spec.name]
    specifications = dict(cls.specifications)
    if spec.kind == "problem":
        specifications.update(cls.model_class.specifications)
    return specifications


def _validate_factors(spec: DesignSpec, specifications: dict[str, dict]) -> None:
    """Raise ValueError for unknown factors or factors given in more than one group."""
    groups = [spec.varied, spec.crossed, spec.fixed]
    for name in itertools.chain.from_iterable(groups):
        if name not in specifications:
            raise ValueError(f"Unknown factor '{name}' for {spec.kind} '{spec.name}'.")
    names = [name for group in groups for name in group]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Factors given in more than one group: {duplicates}.")
    for name in spec.varied:
        factor = specifications[name]
        if factor["datatype"] not in (int, float) or not factor.get("isDatafarmable", True):
            raise ValueError(f"Factor '{name}' cannot be varied.")
    for name in spec.crossed:
        if specifications[name]["datatype"] is not bool:
            raise ValueError(f"Factor '{name}' is not a bool and cannot be crossed.")


def build_design(spec: DesignSpec) -> list[dict[str, Any]]:
    """Build the design points described by `spec`.

    Varied factors form a NOLHS, crossed factors are fully crossed with it, and all
    remaining factors take their fixed value or default.

    Returns:
        list[dict[str, Any]]: One dictionary of factor values per design point.

    Raises:
        ValueError: If the class or a factor is unknown, or a factor is repeated.
    """
    specifications = _get_specifications(spec)
    _validate_factors(spec, specifications)

    # Base rows from the varied factors.
    if spec.varied:
        ranges = [(r.min, r.max, r.decimals) for r in spec.varied.values()]
        nolhs = NOLHS(designs=ranges, num_stacks=spec.n_stacks)
        rows = [
            {
                name: specifications[name]["datatype"](value)
                for name, value in zip(spec.varied, point, strict=True)
            }
            for point in nolhs.generate_design()
        ]
    else:
        rows = [{}]

    # Cross with the bool factors: outer loop over combinations, inner over rows.
    crossed_names = list(spec.crossed)
    combinations = itertools.product(*spec.crossed.values())
    rows = [
        {**row, **dict(zip(crossed_names, combination, strict=True))}
        for combination in combinations
        for row in rows
    ]

    # Fill the remaining factors.
    design = []
    for row in rows:
        point = {}
        for name, factor in specifications.items():
            if name in row:
                point[name] = row[name]
            else:
                point[name] = copy.deepcopy(spec.fixed.get(name, factor.get("default")))
        design.append(point)
    return design
