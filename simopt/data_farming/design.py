"""GUI-independent generation of data-farming designs."""

import copy
import itertools
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

from simopt.data_farming.nolhs import NOLHS
from simopt.directory import model_directory, problem_directory, solver_directory
from simopt.experiment import EXPERIMENT_DIR
from simopt.problem import Problem

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
        problem_cls = problem_directory[spec.name]
        specifications.update(problem_cls.model_class.specifications)
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


def create_design_list_from_table(design_table: DataFrame) -> list[dict[str, Any]]:
    """Create a list of solver or problem objects for each design point.

    Args:
        design_table (DataFrame): DataFrame containing the design table.
            Each row represents a design point, and each column represents a factor.

    Returns:
        list[dict[str, Any]]: List of dictionaries, where each list entry corresponds
        to a design point, and each dictionary contains the name and values for each
        factor in that design point.
    """
    from ast import literal_eval

    # Creates dictonary of table to convert values to proper datatypes.
    dp_dict = design_table.to_dict(orient="list")

    # TODO: this is a hack to get the data type of the factors back.
    design_list = []
    for dp in range(len(design_table)):
        config = {}
        for factor in dp_dict:
            key = str(factor)
            raw_value = str(dp_dict[factor][dp])
            try:
                value = literal_eval(raw_value)
            except ValueError:
                # This exception handles the case where the value is a string.
                value = raw_value
            config[key] = value
        design_list.append(config)

    return design_list


def create_design(
    name: str,
    factor_headers: list[str],
    factor_settings: list[tuple[float, float, int]] | Path,
    fixed_factors: dict | None = None,
    cross_design_factors: dict | None = None,
    design_type: Literal["nolhs"] = "nolhs",
    n_stacks: int = 1,  # TODO: make **variable for other design types?
) -> list[dict[str, Any]]:
    """Creates a design of solver, problem, or model factors.

    Please ensure the indexing of the factor_headers argument matches the indexing of
    the factor_settings argument.

    Args:
        name (str): Name of the solver, problem, or model.
        factor_headers (list[str]): Names of factors that vary in the design.
        factor_settings (list[tuple[float, float, int]] | Path):
            A list of tuples, each of the form (min, max, # decimals)
            or a Path to a .txt file containing those factor settings.
        fixed_factors (dict, optional): Dictionary of fixed factor values that
            override defaults.
        cross_design_factors (dict, optional): Dictionary of lists of cross-design
            factor values. Defaults to None.
        design_type (Literal["nolhs"], optional): Type of design. Defaults to "nolhs".
        n_stacks (int, optional): Number of stacks. Defaults to 1.

    Returns:
        list[dict[str, Any]]: A list of dictionaries, where each dictionary represents
            a design.

    Raises:
        ValueError: If input validation fails.
        Exception: If the design type is unsupported.
    """
    # Default values
    if cross_design_factors is None:
        cross_design_factors = {}
    if fixed_factors is None:
        fixed_factors = {}

    # Create object of the correct type.
    directories = solver_directory | problem_directory | model_directory
    # Make sure we don't accidentally have the same name in multiple directories.
    expected_len = len(solver_directory) + len(problem_directory) + len(model_directory)
    if len(directories) != expected_len:
        error_msg = "Duplicate names found in solver, problem, or model directories."
        raise ValueError(error_msg)
    if name not in directories:
        raise ValueError(f"Name '{name}' not found in any directory.")
    design_object = directories[name]()

    # Make directory to store the current design file.
    df_dir = EXPERIMENT_DIR / "data_farming"
    df_dir.mkdir(parents=True, exist_ok=True)

    # If factor settings is a list of tuples, use that directly.
    if isinstance(factor_settings, list):
        designs = factor_settings
        design_file = df_dir / f"{name}_design.txt"
    # Otherwise, setup the design file output.
    elif isinstance(factor_settings, Path):
        designs = df_dir / f"{factor_settings.name}.txt"
        design_file = df_dir / f"{factor_settings.stem}_design.txt"

    # Only datafarm if there are factors to vary.
    if len(factor_headers) > 0:
        # Select design type.
        if design_type == "nolhs":
            design = NOLHS(designs=designs, num_stacks=n_stacks)
        else:
            error_msg = f"Design type {design_type} not supported."
            raise Exception(error_msg)

        # Generate design table from design object.
        generated_design = design.generate_design()
        if len(generated_design) == 0:
            error_msg = "Error in design generation. No design points generated."
            raise Exception(error_msg)
        design_table = pd.DataFrame(generated_design, columns=pd.Index(factor_headers))

        # Save design to .txt file for backwards compatibility.
        design.save_design(design_file)
    else:
        # Grab one key/value pair from the specifications
        first_item = design_object.specifications
        # Create a DataFrame with the key/value pair
        design_table = pd.DataFrame(first_item, index=pd.Index([0]))

    specifications = design_object.specifications
    # If problem, add model specifications too.
    if isinstance(design_object, Problem):
        specifications = {**specifications, **design_object.model.specifications}

    # Add default values to str dict for unspecified factors.
    fixed_factors_and_headers = set(list(fixed_factors) + factor_headers)
    unspecified_factors = set(specifications) - fixed_factors_and_headers
    for factor in unspecified_factors:
        fixed_factors[factor] = specifications[factor].get("default")

    # Add all the fixed factors to the design table
    for factor in fixed_factors:
        # Use string to ensure correct addition of tuples & list data types.
        design_table[factor] = str(fixed_factors[factor])

    # Add cross design factors to design table.
    if len(cross_design_factors) > 0:
        # Create combination of categorical factor options.
        cross_factor_names = list(cross_design_factors.keys())
        combinations = itertools.product(*(cross_design_factors[opt] for opt in cross_factor_names))

        new_design_table = pd.DataFrame()  # Temp empty value.
        for combination in combinations:
            # Dictionary containing current combination of cross design factor values.
            combination_dict = dict(zip(cross_factor_names, combination, strict=False))
            working_design_table = design_table.copy()

            # Batch add cross design factors to working design table.
            working_design_table = working_design_table.assign(
                **{k: str(v) for k, v in combination_dict.items()}
            )

            # Append working design table to new design table.
            new_design_table = pd.concat(
                [new_design_table, working_design_table], ignore_index=True
            )
        # Update design table to new design table after all combinations added.
        design_table = new_design_table

    design_list = create_design_list_from_table(design_table)

    # Write extra design information to design table.
    design_table.insert(0, "design_num", pd.Series(range(len(design_table))))
    design_table["name"] = design_object.name
    design_table["design_type"] = design_type
    design_table["num_stacks"] = str(n_stacks)
    # Dump the design table to a csv file.
    design_file_csv = design_file.with_suffix(".csv")
    design_table.to_csv(design_file_csv, mode="w", header=True, index=False, sep="\t")

    # Now return the list from earlier
    return design_list
