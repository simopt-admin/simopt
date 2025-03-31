"""Base classes for data-farming experiments and meta-experiments."""

from __future__ import annotations

import ast
import csv
import itertools
import logging
import os
import subprocess
from copy import deepcopy
from enum import Enum
from typing import Literal

import pandas as pd
from numpy import inf

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import Model
from simopt.directory import model_directory, solver_directory
from simopt.experiment_base import EXPERIMENT_DIR, ProblemSolver, post_normalize

DATA_FARMING_DIR = os.path.join(EXPERIMENT_DIR, "data_farming")


class DesignType(Enum):
    nolhs = "nolhs"


class DesignPoint:
    """Base class for design points represented as dictionaries of factors.

    Attributes
    ----------
    model : ``base.Model``
        Model to simulate.
    model_factors : dict
        Model factor names and values.
    rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
        Rngs for model to use when running replications at the solution.
    n_reps : int
        Number of replications run at a design point.
    responses : dict
        Responses observed from replications.
    gradients : dict [dict]
        Gradients of responses (w.r.t. model factors) observed from replications.

    Parameters
    ----------
    model : ``base.Model``
        Model with factors model_factors.

    """

    def __init__(self, model: Model) -> None:
        """Initialize design point with a model object.

        Parameters
        ----------
        model : ``base.Model``
            Model with factors model_factors.

        Raises
        ------
        TypeError

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

        Parameters
        ----------
        rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
            List of random-number generators used to run simulation replications.
        copy : bool, default=True
            True if rng_list should be copied, otherwise False.

        Raises
        ------
        TypeError

        """
        # Type checking
        if not isinstance(rng_list, list) or not all(
            isinstance(rng, MRG32k3a) for rng in rng_list
        ):
            error_msg = "rng_list must be a list of MRG32k3a objects."
            raise TypeError(error_msg)
        if not isinstance(copy, bool):
            error_msg = "copy must be a boolean."
            raise TypeError(error_msg)

        if copy:
            self.rng_list = [deepcopy(rng) for rng in rng_list]
        else:
            self.rng_list = rng_list

    def simulate(self, num_macroreps: int = 1) -> None:
        """Simulate m replications for the current model factors and append results to the responses and gradients dictionaries.

        Parameters
        ----------
        num_macroreps : int, default=1
            Number of macroreplications to run at the design point; > 0.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(num_macroreps, int):
            error_msg = "num_macroreps must be an integer."
            raise TypeError(error_msg)
        # Value checking
        if num_macroreps <= 0:
            error_msg = "Number of macroreplications must be greater than 0."
            raise ValueError(error_msg)

        for _ in range(num_macroreps):
            # Generate a single replication of model, as described by design point.
            responses, gradients = self.model.replicate(rng_list=self.rng_list)
            # If first replication, set up recording responses and gradients.
            if self.n_reps == 0:
                self.responses = {
                    response_key: [] for response_key in responses
                }
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
    """Base class for data-farming experiments consisting of an model and design of associated factors.

    Attributes
    ----------
    model : ``base.Model``
        Model on which the experiment is run.
    design : list [``data_farming_base.DesignPoint``]
        List of design points forming the design.
    n_design_pts : int
        Number of design points in the design.

    Parameters
    ----------
    model_name : str
        Name of model on which the experiment is run.
    factor_settings_filename : str
        Name of .txt file containing factor ranges and # of digits.
    factor_headers : list [str]
        Ordered list of factor names appearing in factor settings/design file.
    design_filename : str
        Name of .txt file containing design matrix.
    model_fixed_factors : dict
        Non-default values of model factors that will not be varied.

    """

    def __init__(
        self,
        model_name: str,
        factor_settings_filename: str | os.PathLike | None,
        factor_headers: list[str],
        design_filepath: str | os.PathLike | None = None,
        model_fixed_factors: dict | None = None,
        stacks: int = 1,
        design_type: Literal["nolhs"] = "nolhs",
    ) -> None:
        """Initialize data-farming experiment with a model object and design of associated factors.

        Parameters
        ----------
        model_name : str
            Name of model on which the experiment is run.
        factor_settings_filename : str | os.PathLike | None
            Name of .txt file containing factor ranges and # of digits.
        factor_headers : list [str]
            Ordered list of factor names appearing in factor settings/design file.
        design_filepath : str, optional
            Name of .txt file containing design matrix.
        model_fixed_factors : dict, optional
            Non-default values of model factors that will not be varied.
        stacks : int, default=1
            Number of stacks in the design.
        design_type: Literal["nolhs"], default="nolhs"
            Type of design to be used.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(model_name, str):
            error_msg = "model_name must be a string."
            raise TypeError(error_msg)
        if not isinstance(
            factor_settings_filename, (str, os.PathLike, type(None))
        ):
            error_msg = "factor_settings_filename must be a string, path-like object, or None."
            raise TypeError(error_msg)
        if not isinstance(factor_headers, list) or not all(
            isinstance(header, str) for header in factor_headers
        ):
            error_msg = "factor_headers must be a list of strings."
            raise TypeError(error_msg)
        if not isinstance(design_filepath, (str, os.PathLike, type(None))):
            error_msg = (
                "design_filename must be a string, path-like object, or None."
            )
            raise TypeError(error_msg)
        if not isinstance(model_fixed_factors, (dict, type(None))):
            error_msg = "model_fixed_factors must be a dictionary."
            raise TypeError(error_msg)
        if not isinstance(stacks, int):
            error_msg = "stacks must be an integer."
            raise TypeError(error_msg)
        # Value checking
        if model_name not in model_directory:
            error_msg = "model_name must be a valid model name."
            raise ValueError(error_msg)
        if factor_settings_filename is not None and not os.path.exists(
            factor_settings_filename
        ):
            error_msg = f"{factor_settings_filename} is not a valid file path."
            raise ValueError(error_msg)  # Change to FileNotFoundError?
        if design_filepath is not None and not os.path.exists(design_filepath):
            error_msg = f"{design_filepath} is not a valid file path."
            raise ValueError(error_msg)  # Change to FileNotFoundError?
        if stacks <= 0:
            error_msg = "Number of stacks must be greater than 0."
            raise ValueError(error_msg)

        if model_fixed_factors is None:
            model_fixed_factors = {}

        # Initialize model object with fixed factors.
        self.model = model_directory[model_name](
            fixed_factors=model_fixed_factors
        )
        if design_filepath is None:
            assert factor_settings_filename is not None, (
                "factor_settings_filename must be provided if design_filepath is None."
            )
            # Create model factor design from .txt file of factor settings.
            # Hard-coded for a single-stack NOLHS.
            filepath_core = os.path.join(
                DATA_FARMING_DIR, factor_settings_filename
            )
            source_filepath = filepath_core + ".txt"
            design_filepath = filepath_core + "_design.txt"
            command = f"stack_{design_type}.rb -s {stacks} {source_filepath} > {design_filepath}"
            subprocess.run(command)
            # Append design to base filename.
        # Read in design matrix from .txt file. Result is a pandas DataFrame.
        design_table = pd.read_csv(
            design_filepath,
            header=None,
            delimiter="\t",
            encoding="utf-8",
        )
        # Count number of design_points.
        self.n_design_pts = len(design_table)
        # Create all design points.
        self.design = []
        design_pt_factors = {}
        for dp_index in range(self.n_design_pts):
            for factor_idx in range(len(factor_headers)):
                # Parse model factors for next design point.
                design_pt_factors[factor_headers[factor_idx]] = design_table[
                    factor_idx
                ][dp_index]
            # Update model factors according to next design point.
            self.model.factors.update(design_pt_factors)
            # Create new design point and add to design.
            self.design.append(DesignPoint(self.model))

    def run(self, n_reps: int = 10, crn_across_design_pts: bool = True) -> None:
        """Run a fixed number of macroreplications at each design point.

        Parameters
        ----------
        n_reps : int, default=10
            Number of replications run at each design point.
        crn_across_design_pts : bool, default=True
            True if CRN are to be used across design points, otherwise False.

        Raises
        ------
        TypeError
        ValueError

        """
        # Type checking
        if not isinstance(n_reps, int):
            error_msg = "n_reps must be an integer."
            raise TypeError(error_msg)
        if not isinstance(crn_across_design_pts, bool):
            error_msg = "crn_across_design_pts must be a boolean."
            raise TypeError(error_msg)
        # Value checking
        if n_reps <= 0:
            error_msg = "Number of replications must be greater than 0."
            raise ValueError(error_msg)

        # Setup random number generators for model.
        # Use stream 0 for all runs; start with substreams 0, 1, ..., model.n_rngs-1.
        main_rng_list = [
            MRG32k3a(s_ss_sss_index=[0, ss, 0])
            for ss in range(self.model.n_rngs)
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
        self, csv_filename: str | os.PathLike = "raw_results"
    ) -> None:
        """Extract observed responses from simulated design points and publish to .csv output file.

        Parameters
        ----------
        csv_filename : str, default="raw_results"
            Name of .csv file to print output to.

        Raises
        ------
        TypeError

        """
        # Type checking
        if not isinstance(csv_filename, (str, os.PathLike)):
            error_msg = "csv_filename must be a string or path-like object."
            raise TypeError(error_msg)

        # Add CSV extension if not present.
        csv_filename = str(csv_filename)
        if not csv_filename.endswith(".csv"):
            csv_filename += ".csv"
        # Check if there's a directory in the file path.
        if os.path.dirname(csv_filename) == "":
            # If not, add the default directory.
            csv_filename = os.path.join(DATA_FARMING_DIR, csv_filename)
        # Create directory if they do no exist.
        if not os.path.exists(DATA_FARMING_DIR):
            os.makedirs(DATA_FARMING_DIR)

        # Write results to csv file.
        with open(csv_filename, mode="x", newline="") as output_file:
            csv_writer = csv.writer(
                output_file,
                delimiter=",",
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


class DataFarmingMetaExperiment:
    """Base class for data-farming meta experiments consisting of problem-solver pairs and a design of associated factors.

    Attributes
    ----------
    design : list [``experiment_base.ProblemSolver``]
        List of design points forming the design.
    n_design_pts : int
        Number of design points in the design.

    Parameters
    ----------
    solver_name : str
        Name of solver.
    n_stacks : int, default = 1
        Number of stacks in the design.
    design_type : str, default = "nolhs"
        Type of design to use.
    solver_factor_headers : list [str]
        Ordered list of solver factor names appearing in factor settings/design file.
    solver_factor_settings_filename : str | None, default=None
        Name of .txt file containing solver factor ranges and # of digits.
    design_filename : str, default=None
        Name of .txt file containing design matrix.
    csv_filename : str, default=None
        Name of .csv file to print output to.
    solver_fixed_factors : dict, default=None
        Dictionary of user-specified solver factors that will not be varied.
    problem_fixed_factors : dict, default=None
        Dictionary of user-specified problem factors that will not be varied.
    model_fixed_factors : dict, default=None
        Dictionary of user-specified model factors that will not be varied.

    """

    def __init__(
        self,
        solver_name: str | None = None,
        solver_factor_headers: list[str] | None = None,
        n_stacks: int = 1,
        design_type: Literal["nolhs"] = "nolhs",
        solver_factor_settings_filename: str | None = None,
        design_filename: str | None = None,
        csv_filename: str | None = None,
        solver_fixed_factors: dict | None = None,
        cross_design_factors: dict | None = None,
    ) -> None:
        """Initialize data-farming meta experiment with a solver object and design of associated factors.

        Parameters
        ----------
        solver_name : str, optional
            Name of solver.
        solver_factor_headers : list [str], optional
            Ordered list of solver factor names appearing in factor settings/design file.
        n_stacks : int
            Number of stacks in the design.
        design_type : Litreral["nolhs"], default = "nolhs"
            Type of design to be used.
        solver_factor_settings_filename : str, optional
            Name of .txt file containing solver factor ranges and # of digits.
        design_filename : str, optional
            Name of .txt file containing design matrix.
        csv_filename : str, optional
            Name of .csv file containing design matrix.
        solver_fixed_factors : dict, optional
            Dictionary of user-specified solver factors that will not be varied.
        cross_design_factors : dict, optional
            Dictionary of user-specified solver factors that will not be varied.

        Raises
        ------
        ValueError
            If n_stacks is less than or equal to 0.

        """
        # Type checking
        if not isinstance(solver_name, (str, type(None))):
            error_msg = "solver_name must be a string."
            raise TypeError(error_msg)
        if not isinstance(solver_factor_headers, (list, type(None))) or (
            isinstance(solver_factor_headers, list)
            and not all(
                isinstance(header, str) for header in solver_factor_headers
            )
        ):
            error_msg = "solver_factor_headers must be a dictionary."
            raise TypeError(error_msg)
        if not isinstance(n_stacks, int):
            error_msg = "n_stacks must be an integer."
            raise TypeError(error_msg)
        if not isinstance(design_type, str):
            error_msg = "design_type must be a string."
            raise TypeError(error_msg)
        if not isinstance(solver_factor_settings_filename, (str, type(None))):
            error_msg = "solver_factor_settings_filename must be a string."
            raise TypeError(error_msg)
        if not isinstance(design_filename, (str, type(None))):
            error_msg = "design_filename must be a string."
            raise TypeError(error_msg)
        if not isinstance(csv_filename, (str, type(None))):
            error_msg = "csv_filename must be a string."
            raise TypeError(error_msg)
        if not isinstance(solver_fixed_factors, (dict, type(None))):
            error_msg = "solver_fixed_factors must be a dictionary."
            raise TypeError(error_msg)
        if not isinstance(cross_design_factors, (dict, type(None))):
            error_msg = "cross_design_factors must be a dictionary."
            raise TypeError(error_msg)
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
        if solver_factor_settings_filename is not None and not os.path.exists(
            solver_factor_settings_filename
        ):
            error_msg = (
                f"{solver_factor_settings_filename} is not a valid file path."
            )
            raise ValueError(error_msg)  # Change to FileNotFoundError?
        if design_filename is not None and not os.path.exists(design_filename):
            error_msg = f"{design_filename} is not a valid file path."
            raise ValueError(error_msg)  # Change to FileNotFoundError?
        if csv_filename is not None and not os.path.exists(csv_filename):
            error_msg = f"{csv_filename} is not a valid file path."
            raise ValueError(error_msg)  # Change to FileNotFoundError?

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
        if design_filename is None and csv_filename is None:
            if solver_factor_settings_filename is None:
                error_msg = "solver_factor_settings_filename must be provided."
                raise ValueError(error_msg)
            # Create solver factor design from .txt file of factor settings.
            input_file = (
                os.path.join(DATA_FARMING_DIR, solver_factor_settings_filename)
                + ".txt"
            )
            output_file = (
                os.path.join(DATA_FARMING_DIR, solver_factor_settings_filename)
                + "_design.txt"
            )
            command = f"stack_{design_type}.rb -s {n_stacks} {input_file} > {output_file}"
            subprocess.run(command)
            # Append design to base filename.
            design_filename = f"{solver_factor_settings_filename}_design"
        # # Read in design matrix from .txt file. Result is a pandas DataFrame.
        # design_table = pd.read_csv(f"./data_farming_experiments/{design_filename}.txt", header=None, delimiter="\t", encoding="utf-8")

        if csv_filename is None:
            # Read in design matrix from .txt file. Result is a pandas DataFrame.
            design_table = pd.read_csv(
                f"./data_farming_experiments/{design_filename}.txt",
                header=None,
                delimiter="\t",
                encoding="utf-8",
            )

            # Create design csv file from design table

            csv_filename = f"./data_farming_experiments/{design_filename}.csv"

            # self.solver_object = solver_directory[solver_name]() #creates solver object
            if solver_factor_headers is not None:
                column_names = zip(
                    range(len(solver_factor_headers)), solver_factor_headers
                )
                design_table.rename(columns=dict(column_names), inplace=True)

            solver_fixed_str = {}
            for factor in (
                solver_fixed_factors
            ):  # make new dict containing strings of solver factors
                solver_fixed_str[factor] = str(solver_fixed_factors[factor])

            for factor in (
                self.solver_object.specifications
            ):  # add default values to str dict for unspecified factors
                default = self.solver_object.specifications[factor].get(
                    "default"
                )
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

            # all_solver_factor_names = solver_factor_headers + list(cross_design_factors.keys()) + list(solver_fixed_factors.keys()) #creates list of all solver factors in order of design, cross-design, then fixed factors

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
                    combination_dict = dict(
                        zip(cross_factor_names, combination)
                    )  # dictionary containing current combination of cross design factor values
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
            design_table.insert(0, "Design #", range(len(design_table)))
            design_table["Solver Name"] = solver_name
            design_table["Design Type"] = design_type
            design_table["Number Stacks"] = str(n_stacks)

            design_table.to_csv(
                csv_filename, mode="w", header=True, index=False
            )

        self.csv_filename = csv_filename

    # Largely taken from MetaExperiment class in wrapper_base.py.
    def run(
        self,
        problem_name: str,
        problem_fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
        n_macroreps: int = 10,
    ) -> None:
        """Run n_macroreps of each problem-solver design point.

        Parameters
        ----------
        problem_name : str
            Name of problem.
        problem_fixed_factors : dict, default=None
            Dictionary of user-specified problem factors that will not be varied.
        model_fixed_factors : dict, default=None
            Dictionary of user-specified model factors that will not be varied.
        n_macroreps : int
            Number of macroreplications for each design point.

        Raises
        ------
        ValueError
            If n_macroreps is less than or equal to 0.

        """
        # Type checking
        if not isinstance(problem_name, str):
            error_msg = "problem_name must be a string."
            raise TypeError(error_msg)
        if not isinstance(problem_fixed_factors, (dict, type(None))):
            error_msg = "problem_fixed_factors must be a dictionary."
            raise TypeError(error_msg)
        if not isinstance(model_fixed_factors, (dict, type(None))):
            error_msg = "model_fixed_factors must be a dictionary."
            raise TypeError(error_msg)
        if not isinstance(n_macroreps, int):
            error_msg = "n_macroreps must be an integer."
            raise TypeError(error_msg)
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
        with open(self.csv_filename) as file:
            reader = csv.reader(file)
            first_row = next(reader)
            # next(reader)
            # next(reader)
            # solver_factor_headers = next(reader)[1:]
            all_solver_factor_names = first_row[1 : -1 * num_extra_col]
            # second_row = next(reader)
            # solver_name = second_row[-1*num_extra_col]
            row_index = 0

            solver_name = ""
            for row in reader:
                solver_name = row[-1 * num_extra_col]
                dp = row[1 : -1 * num_extra_col]
                dp_index = 0
                self.solver_object = solver_directory[solver_name]()
                for factor in all_solver_factor_names:
                    solver_factors_str[factor] = dp[dp_index]
                    dp_index += 1
                # Convert str to proper data type
                for factor in solver_factors_str:
                    val_str = str(solver_factors_str[factor])
                    solver_factors[factor] = ast.literal_eval(val_str)

                solver_factors_insert = solver_factors.copy()
                solver_factors_across_design.append(solver_factors_insert)

                row_index += 1

            self.n_design_pts = len(solver_factors_across_design)
            if self.n_design_pts == 0:
                raise ValueError("No design points found in csv file.")
            for i in range(self.n_design_pts):
                # Create design point on problem solver

                file_name_path = (
                    "./data_farming_experiments/outputs/"
                    + solver_name
                    + "_on_"
                    + problem_name
                    + "_designpt_"
                    + str(i)
                    + ".pickle"
                )
                current_solver_factors = solver_factors_across_design[i]
                new_design_pt = ProblemSolver(
                    solver_name=solver_name,
                    problem_name=problem_name,
                    solver_fixed_factors=current_solver_factors,
                    problem_fixed_factors=problem_fixed_factors,
                    model_fixed_factors=model_fixed_factors,
                    file_name_path=file_name_path,
                )

                self.design.append(new_design_pt)

            # self.n_design_pts = len(self.design)

        for design_pt_index in range(self.n_design_pts):
            # If the problem-solver pair has not been run in this way before,
            # run it now.
            experiment = self.design[design_pt_index]

            if getattr(experiment, "n_macroreps", None) != n_macroreps:
                logging.info(
                    "Running Design Point " + str(design_pt_index) + "."
                )
                experiment.clear_run()
                logging.debug(experiment.solver.name)
                logging.debug(experiment.problem.name)
                experiment.run(n_macroreps)

    # Largely taken from MetaExperiment class in wrapper_base.py.
    def post_replicate(
        self,
        n_postreps: int,
        crn_across_budget: bool = True,
        crn_across_macroreps: bool = False,
    ) -> None:
        """For each design point, run postreplications at solutions recommended by the solver on each macroreplication.

        Parameters
        ----------
        n_postreps : int
            Number of postreplications to take at each recommended solution.
        crn_across_budget : bool, default=True
            True if CRN are to be used for post-replications at solutions recommended at
            different times, otherwise False.
        crn_across_macroreps : bool, default=False
            True if CRN are to be used for post-replications at solutions recommended on
            different macroreplications, otherwise False.

        """
        # Type checking
        if not isinstance(n_postreps, int):
            error_msg = "n_postreps must be an integer."
            raise TypeError(error_msg)
        if not isinstance(crn_across_budget, bool):
            error_msg = "crn_across_budget must be a boolean."
            raise TypeError(error_msg)
        if not isinstance(crn_across_macroreps, bool):
            error_msg = "crn_across_macroreps must be a boolean."
            raise TypeError(error_msg)
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
                or getattr(experiment, "crn_across_budget", None)
                != crn_across_budget
                or getattr(experiment, "crn_across_macroreps", None)
                != crn_across_macroreps
            ):
                logging.info(
                    "Post-processing Design Point " + str(design_pt_index) + "."
                )
                experiment.clear_postreplicate()
                experiment.post_replicate(
                    n_postreps,
                    crn_across_budget=crn_across_budget,
                    crn_across_macroreps=crn_across_macroreps,
                )

    # Largely taken from MetaExperiment class in wrapper_base.py.
    def post_normalize(
        self, n_postreps_init_opt: int, crn_across_init_opt: bool = True
    ) -> None:
        """Post-normalize problem-solver pairs.

        Parameters
        ----------
        n_postreps_init_opt : int
            Number of postreplications to take at initial x0 and optimal x*.
        crn_across_init_opt : bool, default=True
            True if CRN are to be used for post-replications at solutions x0 and x*, otherwise False.

        """
        # Type checking
        if not isinstance(n_postreps_init_opt, int):
            error_msg = "n_postreps_init_opt must be an integer."
            raise TypeError(error_msg)
        if not isinstance(crn_across_init_opt, bool):
            error_msg = "crn_across_init_opt must be a boolean."
            raise TypeError(error_msg)
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
        csv_filename: str | None = None,
    ) -> None:
        """For each design point, calculate statistics from each macoreplication and print to csv.

        Parameters
        ----------
        solve_tols : list [float], default = [0.05, 0.10, 0.20, 0.50]
            Relative optimality gap(s) definining when a problem is solved; in (0,1].
        csv_filename : str, default="df_solver_results"
            Name of .csv file to print output to.

        """
        # Type checking
        if not isinstance(solve_tols, (list, type(None))) or (
            isinstance(solve_tols, list)
            and not all(isinstance(tol, float) for tol in solve_tols)
        ):
            error_msg = "solve_tols must be a list of floats."
            raise TypeError(error_msg)
        if not isinstance(csv_filename, (str, type(None))):
            error_msg = "csv_filename must be a string."
            raise TypeError(error_msg)
        # Value checking
        if solve_tols is not None and not all(
            0 < tol <= 1 for tol in solve_tols
        ):
            error_msg = "Relative optimality gap must be in (0,1]."
            raise ValueError(error_msg)
        # TODO: Figure out if this is a path or just a name
        if csv_filename is not None and not os.path.exists(csv_filename):
            error_msg = f"{csv_filename} is not a valid file path."
            raise ValueError(error_msg)  # Change to FileNotFoundError?

        if solve_tols is None:
            solve_tols = [0.05, 0.10, 0.20, 0.50]
        if csv_filename is None:
            file_path = os.path.join(DATA_FARMING_DIR, "df_solver_results.csv")
        else:
            file_path = csv_filename + ".csv"

        # Create folder(s) for file if needed
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(file_path) as output_file:
            csv_writer = csv.writer(
                output_file,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            base_experiment = self.design[0]
            solver_factor_names = list(
                base_experiment.solver.specifications.keys()
            )
            problem_factor_names = list(
                base_experiment.problem.specifications.keys()
            )
            model_factor_names = list(
                set(base_experiment.problem.model.specifications.keys())
                - base_experiment.problem.model_decision_factors
            )
            # Concatenate solve time headers.
            solve_time_headers = [
                [f"{solve_tol}-Solve Time", f"{solve_tol}-Solved? (Y/N)"]
                for solve_tol in solve_tols
            ]
            solve_time_headers = list(
                itertools.chain.from_iterable(solve_time_headers)
            )
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
                            progress_curve.compute_crossing_time(
                                threshold=solve_tol
                            ),
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
