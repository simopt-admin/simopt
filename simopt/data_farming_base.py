"""Base classes for data-farming experiments and meta-experiments."""

from __future__ import annotations

import ast
import csv
import itertools
import logging
from contextlib import suppress
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Literal

import pandas as pd
from numpy import inf

import simopt.directory as directory
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import Model
from simopt.data_farming.nolhs import NOLHS
from simopt.experiment_base import EXPERIMENT_DIR, ProblemSolver, post_normalize
from simopt.utils import resolve_file_path

DATA_FARMING_DIR = EXPERIMENT_DIR / "data_farming"

# Workaround for AutoAPI
model_directory = directory.model_directory
solver_directory = directory.solver_directory


class DesignType(Enum):
    """Enum for design types."""

    NOLHS = "nolhs"


class DesignPoint:
    """Base class for design points represented as dictionaries of factors."""

    @property
    def model(self) -> Model:
        """Model to simulate."""
        return self._model

    @model.setter
    def model(self, model: Model) -> None:
        self._model = model

    @property
    def model_factors(self) -> dict:
        """Model factor names and values."""
        return self._model_factors

    @model_factors.setter
    def model_factors(self, model_factors: dict) -> None:
        self._model_factors = model_factors

    @property
    def rng_list(self) -> list[MRG32k3a]:
        """RNGs for use when running replications at the solution."""
        return self._rng_list

    @rng_list.setter
    def rng_list(self, rng_list: list[MRG32k3a]) -> None:
        self._rng_list = rng_list

    @property
    def n_reps(self) -> int:
        """Number of replications run at a design point."""
        return self._n_reps

    @n_reps.setter
    def n_reps(self, n_reps: int) -> None:
        self._n_reps = n_reps

    @property
    def responses(self) -> dict:
        """Responses observed from replications."""
        return self._responses

    @responses.setter
    def responses(self, responses: dict) -> None:
        self._responses = responses

    @property
    def gradients(self) -> dict:
        """Gradients of responses (with respect to model factors) from replications."""
        return self._gradients

    @gradients.setter
    def gradients(self, gradients: dict) -> None:
        self._gradients = gradients

    def __init__(self, model: Model) -> None:
        """Initialize a design point with a model object.

        Args:
            model (Model): Model with factors `model_factors`.

        Raises:
            TypeError: If `model` is not an instance of the `Model` class.
        """
        # Type checking
        if not isinstance(model, Model):
            error_msg = "Model must be an instance of the Model class."
            raise TypeError(error_msg)

        super().__init__()
        # Create separate copy of Model object for use at this design point.
        self.model = deepcopy(model)
        self.model_factors = self.model.factors
        self.n_reps = 0
        self.responses = {}
        self.gradients = {}

    def attach_rngs(self, rng_list: list[MRG32k3a], copy: bool = True) -> None:
        """Attach a list of random-number generators to the design point.

        Args:
            rng_list (list[MRG32k3a]): List of random-number generators used to run
                simulation replications.
            copy (bool, optional): Whether to copy the provided `rng_list`.
                Defaults to True.
        """
        self.rng_list = [deepcopy(rng) for rng in rng_list] if copy else rng_list

    def simulate(self, num_macroreps: int = 1) -> None:
        """Simulate macroreplications and update response and gradient data.

        Args:
            num_macroreps (int, optional): Number of macroreplications to run (> 0).
                Defaults to 1.

        Raises:
            ValueError: If `num_macroreps` is not positive.
        """
        # Value checking
        if num_macroreps <= 0:
            error_msg = "Number of macroreplications must be greater than 0."
            raise ValueError(error_msg)

        for _ in range(num_macroreps):
            # Generate a single replication of model, as described by design point.
            self.model.before_replicate(self.rng_list)
            responses, gradients = self.model.replicate()
            # If first replication, set up recording responses and gradients.
            if self.n_reps == 0:
                self.responses = {response_key: [] for response_key in responses}
                self.gradients = {
                    response_key: {
                        factor_key: [] for factor_key in gradients[response_key]
                    }
                    for response_key in responses
                }
            # Append responses and gradients.
            for key in self.responses:
                self.responses[key].append(responses[key])
            for outerkey in self.gradients:
                for innerkey in self.gradients[outerkey]:
                    self.gradients[outerkey][innerkey].append(
                        gradients[outerkey][innerkey]
                    )
            self.n_reps += 1
            # Advance rngs to start of next subsubstream.
            for rng in self.rng_list:
                rng.advance_subsubstream()


class DataFarmingExperiment:
    """Base class for data-farming experiments with a model and factor design."""

    def __init__(
        self,
        model_name: str,
        factor_headers: list[str],
        factor_settings: list[tuple[float, float, int]] | Path | str | None = None,
        design_path: Path | str | None = None,
        model_fixed_factors: dict | None = None,
        design_type: Literal["nolhs"] = "nolhs",
        stacks: int = 1,
    ) -> None:
        """Initializes a data-farming experiment with a model and factor design.

        Either `factor_settings` or `design_path` must be provided.

        Args:
            model_name (str): Name of the model to run.
            factor_headers (list[str]): Ordered list of factor names in the
                settings/design file.
            factor_settings (list[tuple[float, float, int]] | Path | str | None, optional):
                Either a list of tuples (min, max, decimals) specifying factor ranges
                and precision digits, or a Path/str to a .txt file containing
                this information.
            design_path (Path | str | None, optional): Path to the design matrix file.
                Defaults to None.
            model_fixed_factors (dict, optional): Fixed model factor values that are
                not varied.
            design_type (Literal["nolhs"], optional): Design type to use.
                Defaults to "nolhs".
            stacks (int, optional): Number of stacks in the design. Defaults to 1.

        Raises:
            ValueError: If `model_name` is invalid or `design_type` is unsupported.
            FileNotFoundError: If any specified file path does not exist.
        """  # noqa: E501
        if model_fixed_factors is None:
            model_fixed_factors = {}

        # If for some reason the user provides the module name instead of the
        # abbreviated class name, set the proper name.
        if model_name not in model_directory:
            for name, model_class in directory.model_directory.items():
                if model_class.name == model_name:
                    model_name = name
                    break

        # Value checking
        if model_name not in model_directory:
            error_msg = "model_name must be a valid model name."
            raise ValueError(error_msg)
        if design_type not in ["nolhs"]:
            error_msg = "design_type must be a valid design type."
            raise ValueError(error_msg)

        # Initialize model object with fixed factors.
        self.model = model_directory[model_name](fixed_factors=model_fixed_factors)

        # If factor_settings is provided, create design from it.
        if factor_settings is not None:
            if isinstance(factor_settings, (str, Path)):
                # Check the filepath resolves
                factor_settings = resolve_file_path(factor_settings, DATA_FARMING_DIR)
                if not factor_settings.exists():
                    error_msg = f"{factor_settings} is not a valid file path."
                    raise FileNotFoundError(error_msg)
                # Set the path for the design file
                design_path = (DATA_FARMING_DIR / factor_settings).with_name(
                    f"{factor_settings.stem}_design.txt"
                )
            # If factor_settings is given as a list of tuples, no changes needed.
            else:
                design_path = DATA_FARMING_DIR / f"{model_name}_design.txt"
            # Create the design and save to design_path
            design = NOLHS(
                designs=factor_settings,
                num_stacks=stacks,
            )
            design.generate_design()
            design.save_design(design_path)
            logging.info(f"Design saved to {design_path}")
        # If design_path is provided, validate it.
        elif design_path is not None:
            # If a string was provided, attempt to resolve it to a Path.
            if isinstance(design_path, str):
                # 1 - Full Path w/ Extension
                design_path = Path(design_path)
                # 2 - Filename w/ Extension (need to add directory)
                if not design_path.exists():
                    design_path = Path(DATA_FARMING_DIR / design_path)
                # 3 - Filename w/o Extension (need to add directory and extension)
                if not design_path.suffix:
                    design_path = design_path.with_suffix(".csv")
            if not design_path.exists():
                raise FileNotFoundError(
                    f"Path to design ({design_path}) "
                    "was provided but cannot be located."
                )
        # If neither is provided, raise an error.
        else:
            error_msg = "Either factor_settings or design_path must be provided."
            raise ValueError(error_msg)

        design_table = pd.read_csv(
            design_path,
            sep="\t",
            encoding="utf-8",
        )
        # If we don't have factor headers, use the column names from the design table.
        if not factor_headers:
            factor_headers = design_table.columns.tolist()
        # Ensure all the design table columns are numeric where possible.
        for factor in factor_headers:
            # Ignore values that cannot be converted to numeric
            # (e.g., categorical factors).
            with suppress(Exception):
                design_table[factor] = pd.to_numeric(design_table[factor])
                # Convert from numpy to standard Python types.
                design_table[factor] = design_table[factor].apply(lambda x: x.item())
        # Double check the indexing aligns before continuing.
        self.n_design_pts = len(design_table)
        if len(factor_headers) > len(design_table.columns):
            error_msg = (
                f"Number of columns in design table ({len(design_table.columns)}) "
                "must be at least the number of factor headers provided "
                f"({len(factor_headers)})."
            )
            raise ValueError(error_msg)
        # Create all design points.
        self.design = []
        design_pt_factors = {}
        for dp_index in range(self.n_design_pts):
            for idx, factor in enumerate(factor_headers):
                # Skip any extra factors that got read in but are not part of the model.
                if factor not in self.model.factors:
                    logging.debug(f"Factor '{factor}' not in model; skipping.")
                    continue
                # Parse model factors for next design point.
                # If column names are given, then use them to index
                # (columns can be in any order)
                if factor in design_table.columns:
                    factor_val = design_table[factor].iloc[dp_index]
                # Otherwise, use positional indexing (assume columns align with headers)
                else:
                    factor_val = design_table.iloc[dp_index, idx]
                design_pt_factors[factor] = factor_val
            # TODO: investigate if deepcopy is needed for self.model
            # Update model factors according to next design point.
            self.model.factors.update(design_pt_factors)
            # Create new design point and add to design.
            self.design.append(DesignPoint(self.model))

        # Ensure that the data-farming directory exists.
        DATA_FARMING_DIR.mkdir(parents=True, exist_ok=True)

    def run(self, n_reps: int = 10, crn_across_design_pts: bool = True) -> None:
        """Run a fixed number of macroreplications at each design point.

        Args:
            n_reps (int, optional): Number of replications run at each design point.
                Defaults to 10.
            crn_across_design_pts (bool, optional): Whether to use common random numbers
                (CRN) across design points. Defaults to True.

        Raises:
            ValueError: If `n_reps` is not positive.
        """
        # Value checking
        if n_reps <= 0:
            error_msg = "Number of replications must be greater than 0."
            raise ValueError(error_msg)

        # Setup random number generators for model.
        # Use stream 0 for all runs; start with substreams 0, 1, ..., model.n_rngs-1.
        main_rng_list = [
            MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(self.model.n_rngs)
        ]
        # All design points will share the same random number generator objects.
        # Simulate n_reps replications from each design point.
        for design_pt in self.design:
            # Attach random number generators.
            design_pt.attach_rngs(rng_list=main_rng_list, copy=False)
            # Simulate n_reps replications from each design point.
            design_pt.simulate(n_reps)
            # Manage random number streams.
            if crn_across_design_pts:
                # Reset rngs to start of current substream.
                for rng in main_rng_list:
                    rng.reset_substream()
            else:  # If not using CRN...
                # ...advance rngs to starts of next set of substreams.
                for rng in main_rng_list:
                    for _ in range(len(main_rng_list)):
                        rng.advance_substream()

    def print_to_csv(
        self, csv_file_name: Path | str = "raw_results", overwrite: bool = False
    ) -> None:
        """Writes simulated responses for all design points to a CSV file.

        Args:
            csv_file_name (Path | str, optional): Output file name (with or without
                .csv extension). Defaults to "raw_results".
            overwrite (bool, optional): If True, overwrite existing file. Otherwise,
                raises an error if the file already exists. Defaults to False.
        """
        # Resolve the file path
        csv_file_name = resolve_file_path(csv_file_name, DATA_FARMING_DIR)
        # Add CSV extension if not present.
        csv_file_name = csv_file_name.with_suffix(".csv")

        if csv_file_name.exists():
            if overwrite:
                csv_file_name.unlink()
            else:
                error_msg = (
                    f"{csv_file_name} already exists. Set overwrite=True to overwrite."
                )
                raise FileExistsError(error_msg)

        # Write results to csv file.
        with csv_file_name.open(mode="x", newline="") as output_file:
            csv_writer = csv.writer(
                output_file,
                delimiter="\t",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            # Print headers.
            model_factor_names = list(self.model.specifications.keys())
            response_names = list(self.design[0].responses.keys())
            csv_writer.writerow(
                ["DesignPt#", *model_factor_names, "MacroRep#", *response_names]
            )
            for designpt_index in range(self.n_design_pts):
                designpt = self.design[designpt_index]
                # Parse list of model factors.
                model_factor_list = [
                    designpt.model_factors[model_factor_name]
                    for model_factor_name in model_factor_names
                ]
                for mrep in range(designpt.n_reps):
                    # Parse list of responses.
                    response_list = [
                        designpt.responses[response_name][mrep]
                        for response_name in response_names
                    ]
                    print_list = [
                        designpt_index,
                        *model_factor_list,
                        mrep,
                        *response_list,
                    ]
                    csv_writer.writerow(print_list)


# TODO: investigate if this class can be pruned as it is completely unmentioned by
# any other code in the repo.
class DataFarmingMetaExperiment:
    """Base class for data-farming meta-experiments with problem-solver designs."""

    def __init__(
        self,
        solver_name: str | None = None,
        solver_factor_headers: list[str] | None = None,
        n_stacks: int = 1,
        design_type: Literal["nolhs"] = "nolhs",
        solver_factor_settings_file_path: Path | str | None = None,
        design_file_path: Path | str | None = None,
        csv_file_path: Path | str | None = None,
        solver_fixed_factors: dict | None = None,
        cross_design_factors: dict | None = None,
    ) -> None:
        """Initializes a meta-experiment with a solver and a factor design.

        Args:
            solver_name (str, optional): Name of the solver.
            solver_factor_headers (list[str], optional): Ordered list of solver factor
                names from the settings/design file.
            n_stacks (int): Number of stacks in the design. Must be > 0.
            design_type (Literal["nolhs"], optional): Design type to use.
                Defaults to "nolhs".
            solver_factor_settings_file_path (Path | str | None, optional): Path to
                the .txt file defining solver factor ranges and precision.
            design_file_path (Path | str | None, optional): Path to the design
                matrix file.
            csv_file_path (Path | str | None, optional): Path to the CSV file
                containing design data.
            solver_fixed_factors (dict, optional): Solver factors to hold fixed.
            cross_design_factors (dict, optional): Cross-design factor values
                to include.

        Raises:
            ValueError: If solver name is invalid, stacks ≤ 0, or design type is
                unsupported.
            FileNotFoundError: If any given path does not exist.
        """
        # Value checking
        if solver_name is not None and solver_name not in solver_directory:
            error_msg = "solver_name must be a valid solver name."
            raise ValueError(error_msg)
        if n_stacks <= 0:
            error_msg = "Number of stacks must be greater than 0."
            raise ValueError(error_msg)
        if design_type not in ["nolhs"]:
            error_msg = "design_type must be a valid design type."
            raise ValueError(error_msg)

        # Make sure the file_names resolve to valid paths
        if solver_factor_settings_file_path is not None:
            solver_factor_settings_file_path = resolve_file_path(
                solver_factor_settings_file_path, DATA_FARMING_DIR
            )
            if not solver_factor_settings_file_path.exists():
                error_msg = (
                    f"{solver_factor_settings_file_path} is not a valid file path."
                )
                raise FileNotFoundError(error_msg)
        if design_file_path is not None:
            design_file_path = resolve_file_path(design_file_path, DATA_FARMING_DIR)
            if not design_file_path.exists():
                error_msg = f"{design_file_path} is not a valid file path."
                raise FileNotFoundError(error_msg)
        if csv_file_path is not None:
            csv_file_path = resolve_file_path(csv_file_path, DATA_FARMING_DIR)
            if not csv_file_path.exists():
                error_msg = f"{csv_file_path} is not a valid file path."
                raise FileNotFoundError(error_msg)

        if solver_fixed_factors is None:
            solver_fixed_factors = {}
        # if problem_fixed_factors is None:
        #     problem_fixed_factors={}
        # if model_fixed_factors is None:
        #     model_fixed_factors={}
        if cross_design_factors is None:
            cross_design_factors = {}
        if solver_name is not None:
            self.solver_object = solver_directory[
                solver_name
            ]()  # creates solver object
        # TO DO: Extend to allow a design on problem/model factors too.
        # Currently supports designs on solver factors only.
        if design_file_path is None and csv_file_path is None:
            if solver_factor_settings_file_path is None:
                error_msg = "solver_factor_settings_file_name must be provided."
                raise ValueError(error_msg)
            # Create solver factor design from .txt file of factor settings.
            partial_file_path = DATA_FARMING_DIR / solver_factor_settings_file_path
            source_path = partial_file_path.with_suffix(".txt")
            design_path = partial_file_path.with_suffix("_design.txt")
            # Create a design and save to design_path
            design = NOLHS(
                designs=source_path,
                num_stacks=n_stacks,
            )
            design.generate_design()
            design.save_design(design_path)
            # Append design to base file name.
            design_file_path = f"{solver_factor_settings_file_path}_design"

        if csv_file_path is None:
            # Read in design matrix from .txt file. Result is a pandas DataFrame.
            design_path = DATA_FARMING_DIR / f"{design_file_path}.txt"
            design_table = pd.read_csv(
                design_path,
                header=None,
                delimiter="\t",
                encoding="utf-8",
            )

            # Create design csv file from design table
            csv_file_path = DATA_FARMING_DIR / f"{design_file_path}.csv"

            if solver_factor_headers is not None:
                column_names = zip(
                    range(len(solver_factor_headers)),
                    solver_factor_headers,
                    strict=False,
                )
                design_table.rename(columns=dict(column_names), inplace=True)

            # make new dict containing strings of solver factors
            solver_fixed_str = {
                factor: str(value) for factor, value in solver_fixed_factors.items()
            }

            for factor in (
                self.solver_object.specifications
            ):  # add default values to str dict for unspecified factors
                default = self.solver_object.specifications[factor].get("default")
                if (
                    factor not in solver_fixed_str
                    and solver_factor_headers is not None
                    and factor not in solver_factor_headers
                ):
                    logging.debug("default from df base", default)
                    solver_fixed_str[factor] = str(default)

            # all_solver_factor_names = solver_factor_headers + list(
            #     solver_fixed_str.keys()
            # )  # list of all solver factor names

            # # creates list of all solver factors in order of design, cross-design,
            # # then fixed factors
            # all_solver_factor_names = (
            #     solver_factor_headers
            #     + list(cross_design_factors.keys())
            #     + list(solver_fixed_factors.keys())
            # )

            # Add fixed factors to dt
            for factor in solver_fixed_str:
                design_table[factor] = solver_fixed_str[factor]

            # Add cross design factors to design table
            if len(cross_design_factors) != 0:
                # num_cross = 0 # number of times cross design is run

                # create combination of categorical factor options
                cross_factor_names = list(cross_design_factors.keys())
                combinations = itertools.product(
                    *(cross_design_factors[opt] for opt in cross_factor_names)
                )

                new_design_table = pd.DataFrame()  # temp empty value
                for combination in combinations:
                    # dictionary containing current combination of cross design
                    # factor values
                    combination_dict = dict(
                        zip(cross_factor_names, combination, strict=False)
                    )
                    working_design_table = design_table.copy()

                    for factor in combination_dict:
                        str_factor_val = str(combination_dict[factor])
                        working_design_table[factor] = str_factor_val

                    new_design_table = pd.concat(
                        [new_design_table, working_design_table],
                        ignore_index=True,
                    )
                    logging.debug(new_design_table)

                design_table = new_design_table

            # Add design information to table
            design_table.insert(0, "Design #", pd.Series(range(len(design_table))))
            design_table["Solver Name"] = solver_name
            design_table["Design Type"] = design_type
            design_table["Number Stacks"] = str(n_stacks)

            design_table.to_csv(csv_file_path, mode="w", header=True, index=False)

        self.design_table_path = csv_file_path

    # Largely taken from MetaExperiment class in wrapper_base.py.
    def run(
        self,
        problem_name: str,
        problem_fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
        n_macroreps: int = 10,
    ) -> None:
        """Run `n_macroreps` of each problem-solver design point.

        Args:
            problem_name (str): Name of the problem.
            problem_fixed_factors (dict | None, optional): Dictionary of user-specified
                problem factors that will not be varied.
            model_fixed_factors (dict | None, optional): Dictionary of user-specified
                model factors that will not be varied.
            n_macroreps (int): Number of macroreplications for each design point.

        Raises:
            ValueError: If `n_macroreps` is less than or equal to 0.
        """
        # Value checking
        if n_macroreps <= 0:
            error_msg = "Number of macroreplications must be greater than 0."
            raise ValueError(error_msg)

        if problem_fixed_factors is None:
            problem_fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        solver_factors = {}  # holds solver factors for individual dp
        solver_factors_across_design = []  # holds solver factors across all dps
        self.design = []
        num_extra_col = 3  # change if extra columns in design table changes

        # Read design points from csv file
        solver_factors_str = {}  # holds design points as string for all factors
        with self.design_table_path.open() as file:
            reader = csv.reader(file)
            first_row = next(reader)
            # next(reader)
            # next(reader)
            # solver_factor_headers = next(reader)[1:]
            all_solver_factor_names = first_row[1 : -1 * num_extra_col]
            # second_row = next(reader)
            # solver_name = second_row[-1*num_extra_col]

            solver_name = ""
            for row in reader:
                solver_name = row[-1 * num_extra_col]
                dp = row[1 : -1 * num_extra_col]
                dp_index = 0
                self.solver_object = solver_directory[solver_name]()
                for dp_index, factor in enumerate(all_solver_factor_names):
                    solver_factors_str[factor] = dp[dp_index]
                # Convert str to proper data type
                for factor in solver_factors_str:
                    val_str = str(solver_factors_str[factor])
                    solver_factors[factor] = ast.literal_eval(val_str)

                solver_factors_insert = solver_factors.copy()
                solver_factors_across_design.append(solver_factors_insert)

            self.n_design_pts = len(solver_factors_across_design)
            if self.n_design_pts == 0:
                raise ValueError("No design points found in csv file.")
            for i in range(self.n_design_pts):
                # Create design point on problem solver
                file_name = f"{solver_name}_on_{problem_name}_designpt_{i}.pickle"
                file_path = DATA_FARMING_DIR / "outputs" / file_name

                current_solver_factors = solver_factors_across_design[i]
                new_design_pt = ProblemSolver(
                    solver_name=solver_name,
                    problem_name=problem_name,
                    solver_fixed_factors=current_solver_factors,
                    problem_fixed_factors=problem_fixed_factors,
                    model_fixed_factors=model_fixed_factors,
                    file_name_path=file_path,
                )

                self.design.append(new_design_pt)

            # self.n_design_pts = len(self.design)

        for design_pt_index in range(self.n_design_pts):
            # If the problem-solver pair has not been run in this way before,
            # run it now.
            experiment = self.design[design_pt_index]

            if getattr(experiment, "n_macroreps", None) != n_macroreps:
                logging.info(f"Running Design Point {design_pt_index}.")
                experiment.clear_run()
                logging.debug(experiment.solver.name)
                logging.debug(experiment.problem.name)
                experiment.run(n_macroreps)

    def post_replicate(
        self,
        n_postreps: int,
        crn_across_budget: bool = True,
        crn_across_macroreps: bool = False,
    ) -> None:
        """Runs postreplications for each design point on all macroreplications.

        Args:
            n_postreps (int): Number of postreplications per recommended solution.
            crn_across_budget (bool, optional): If True, use CRN across solutions from
                different time budgets. Defaults to True.
            crn_across_macroreps (bool, optional): If True, use CRN across different
                macroreplications. Defaults to False.

        Raises:
            ValueError: If `n_postreps` is not positive.
        """
        # Value checking
        if n_postreps <= 0:
            error_msg = "Number of postreplications must be greater than 0."
            raise ValueError(error_msg)

        for design_pt_index in range(self.n_design_pts):
            experiment = self.design[design_pt_index]
            # If the problem-solver pair has not been post-processed in this way before,
            # post-process it now.
            if (
                getattr(experiment, "n_postreps", None) != n_postreps
                or getattr(experiment, "crn_across_budget", None) != crn_across_budget
                or getattr(experiment, "crn_across_macroreps", None)
                != crn_across_macroreps
            ):
                logging.info(f"Post-processing Design Point {design_pt_index}.")
                experiment.clear_postreplicate()
                experiment.post_replicate(
                    n_postreps,
                    crn_across_budget=crn_across_budget,
                    crn_across_macroreps=crn_across_macroreps,
                )

    def post_normalize(
        self, n_postreps_init_opt: int, crn_across_init_opt: bool = True
    ) -> None:
        """Post-normalizes all problem-solver pairs in the design.

        Args:
            n_postreps_init_opt (int): Number of postreplications at x0 and x*.
            crn_across_init_opt (bool, optional): If True, use CRN across x0 and x*.
                Defaults to True.

        Raises:
            ValueError: If `n_postreps_init_opt` is not positive.
        """
        # Value checking
        if n_postreps_init_opt <= 0:
            error_msg = "Number of postreplications must be greater than 0."
            raise ValueError(error_msg)

        post_normalize(
            experiments=self.design,
            n_postreps_init_opt=n_postreps_init_opt,
            crn_across_init_opt=crn_across_init_opt,
        )

    def report_statistics(
        self,
        solve_tols: list[float] | None = None,
        csv_file_name: Path | str | None = None,
    ) -> None:
        """Calculates and writes macroreplication statistics for each design point.

        Args:
            solve_tols (list[float], optional): Optimality gaps for considering a
                problem solved (values in (0, 1]). Defaults to [0.05, 0.10, 0.20, 0.50].
            csv_file_name (Path | str | None, optional): Name of the output CSV file.
                Defaults to "df_solver_results.csv".
        """
        if solve_tols is None:
            solve_tols = [0.05, 0.10, 0.20, 0.50]
        # Value checking
        if solve_tols is not None and not all(0 < tol <= 1 for tol in solve_tols):
            error_msg = "Relative optimality gap must be in (0,1]."
            raise ValueError(error_msg)
        # Make sure the file name resolves if it isn't None
        if csv_file_name is not None:
            csv_file_name = resolve_file_path(csv_file_name, DATA_FARMING_DIR)
            if not csv_file_name.exists():
                error_msg = f"{csv_file_name} does not exist."
                raise FileNotFoundError(error_msg)
        else:
            csv_file_name = DATA_FARMING_DIR / "df_solver_results.csv"

        # Set the file path
        file_path = csv_file_name.with_suffix(".csv")
        file_path.mkdir(parents=True, exist_ok=True)

        with file_path.open() as output_file:
            csv_writer = csv.writer(
                output_file,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            base_experiment = self.design[0]
            solver_factor_names = list(base_experiment.solver.specifications.keys())
            problem_factor_names = list(base_experiment.problem.specifications.keys())
            model_factor_names = list(
                set(base_experiment.problem.model.specifications.keys())
                - base_experiment.problem.model_decision_factors
            )
            # Concatenate solve time headers.
            solve_time_headers = [
                [f"{solve_tol}-Solve Time", f"{solve_tol}-Solved? (Y/N)"]
                for solve_tol in solve_tols
            ]
            solve_time_headers = list(itertools.chain.from_iterable(solve_time_headers))
            # Print headers.
            csv_writer.writerow(
                [
                    "DesignPt#",
                    *solver_factor_names,
                    *problem_factor_names,
                    *model_factor_names,
                    "MacroRep#",
                    "Final Relative Optimality Gap",
                    "Area Under Progress Curve",
                    *solve_time_headers,
                ]
            )
            # Compute performance metrics.
            for designpt_index in range(self.n_design_pts):
                experiment = self.design[designpt_index]
                # Parse lists of factors.
                solver_factor_list = [
                    experiment.solver.factors[solver_factor_name]
                    for solver_factor_name in solver_factor_names
                ]
                problem_factor_list = [
                    experiment.problem.factors[problem_factor_name]
                    for problem_factor_name in problem_factor_names
                ]
                model_factor_list = [
                    experiment.problem.model.factors[model_factor_name]
                    for model_factor_name in model_factor_names
                ]
                for mrep in range(experiment.n_macroreps):
                    progress_curve = experiment.progress_curves[mrep]
                    # Parse list of statistics.
                    solve_time_values = [
                        [
                            progress_curve.compute_crossing_time(threshold=solve_tol),
                            int(
                                progress_curve.compute_crossing_time(
                                    threshold=solve_tol
                                )
                                < inf
                            ),
                        ]
                        for solve_tol in solve_tols
                    ]
                    solve_time_values = list(
                        itertools.chain.from_iterable(solve_time_values)
                    )
                    statistics_list = [
                        progress_curve.y_vals[-1],
                        progress_curve.compute_area_under_curve(),
                        *solve_time_values,
                    ]
                    print_list = [
                        designpt_index,
                        *solver_factor_list,
                        *problem_factor_list,
                        *model_factor_list,
                        mrep,
                        *statistics_list,
                    ]
                    csv_writer.writerow(print_list)
