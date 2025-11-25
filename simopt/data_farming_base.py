"""Base classes for data-farming experiments and meta-experiments."""

from __future__ import annotations

import csv
import logging
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import pandas as pd

import simopt.directory as directory
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import Model
from simopt.data_farming.nolhs import NOLHS
from simopt.experiment_base import EXPERIMENT_DIR
from simopt.utils import resolve_file_path

DATA_FARMING_DIR = EXPERIMENT_DIR / "data_farming"

# Workaround for AutoAPI
model_directory = directory.model_directory
solver_directory = directory.solver_directory


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
                    for response_key in gradients
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

        def fn(x: Any) -> Any:  # noqa: ANN401
            import ast

            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return x

            return x

        design_table = design_table.map(fn)

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
