"""Multiple problem-solver pairs."""

import importlib
import itertools
import logging
import pickle
from pathlib import Path

import simopt.directory as directory
from simopt.base import Problem, Solver

from .post_normalize import post_normalize
from .single import EXPERIMENT_DIR, ProblemSolver

# Workaround for AutoAPI
model_directory = directory.model_directory
problem_directory = directory.problem_directory
solver_directory = directory.solver_directory


class ProblemsSolvers:
    """Base class for running one or more solver on one or more problem."""

    @property
    def solver_names(self) -> list[str]:
        """List of solver names."""
        return self.__solver_names

    @solver_names.setter
    def solver_names(self, solver_names: list[str]) -> None:
        self.__solver_names = solver_names
        self.__n_solvers = len(solver_names)

    @property
    def n_solvers(self) -> int:
        """Number of solvers."""
        return self.__n_solvers

    @property
    def problem_names(self) -> list[str]:
        """List of problem names."""
        return self.__problem_names

    @problem_names.setter
    def problem_names(self, problem_names: list[str]) -> None:
        self.__problem_names = problem_names
        self.__n_problems = len(problem_names)

    @property
    def n_problems(self) -> int:
        """Number of problems."""
        return self.__n_problems

    @property
    def solvers(self) -> list[Solver]:
        """List of solvers."""
        return self.__solvers

    @solvers.setter
    def solvers(self, solvers: list[Solver]) -> None:
        self.__solvers = solvers

    @property
    def problems(self) -> list[Problem]:
        """List of problems."""
        return self.__problems

    @problems.setter
    def problems(self, problems: list[Problem]) -> None:
        self.__problems = problems

    @property
    def all_solver_fixed_factors(self) -> dict[str, dict]:
        """Fixed solver factors for each solver."""
        return self.__all_solver_fixed_factors

    @all_solver_fixed_factors.setter
    def all_solver_fixed_factors(
        self, all_solver_fixed_factors: dict[str, dict]
    ) -> None:
        self.__all_solver_fixed_factors = all_solver_fixed_factors

    @property
    def all_problem_fixed_factors(self) -> dict[str, dict]:
        """Fixed problem factors for each problem."""
        return self.__all_problem_fixed_factors

    @all_problem_fixed_factors.setter
    def all_problem_fixed_factors(
        self, all_problem_fixed_factors: dict[str, dict]
    ) -> None:
        self.__all_problem_fixed_factors = all_problem_fixed_factors

    @property
    def all_model_fixed_factors(self) -> dict[str, dict]:
        """Fixed model factors for each problem."""
        return self.__all_model_fixed_factors

    @all_model_fixed_factors.setter
    def all_model_fixed_factors(self, all_model_fixed_factors: dict[str, dict]) -> None:
        self.__all_model_fixed_factors = all_model_fixed_factors

    @property
    def experiments(self) -> list[list[ProblemSolver]]:
        """All problem-solver pairs."""
        return self.__experiments

    @experiments.setter
    def experiments(self, experiments: list[list[ProblemSolver]]) -> None:
        self.__experiments = experiments

    @property
    def file_name_path(self) -> Path:
        """Path to the .pickle file for saving the ProblemsSolvers object."""
        return self.__file_name_path

    @file_name_path.setter
    def file_name_path(self, file_name_path: Path) -> None:
        self.__file_name_path = file_name_path

    @property
    def create_pair_pickles(self) -> bool:
        """Whether to create pickle files for each problem-solver pair."""
        return self.__create_pair_pickles

    @create_pair_pickles.setter
    def create_pair_pickles(self, create_pair_pickles: bool) -> None:
        self.__create_pair_pickles = create_pair_pickles

    @property
    def experiment_name(self) -> str:
        """Name of experiment to be appended to the beginning of output files."""
        return self.__experiment_name

    @experiment_name.setter
    def experiment_name(self, experiment_name: str) -> None:
        self.__experiment_name = experiment_name

    # TODO: If loading some ProblemSolver objects from file, check that their factors
    # match those in the overall ProblemsSolvers.
    def __init__(
        self,
        solver_factors: list[dict] | None = None,
        problem_factors: list[dict] | None = None,
        solver_names: list[str] | None = None,
        problem_names: list[str] | None = None,
        solver_renames: list[str] | None = None,
        problem_renames: list[str] | None = None,
        fixed_factors_filename: str | None = None,
        solvers: list[Solver] | None = None,
        problems: list[Problem] | None = None,
        experiments: list[list[ProblemSolver]] | None = None,
        file_name_path: Path | None = None,
        create_pair_pickles: bool = False,
        experiment_name: str | None = None,
    ) -> None:
        """Initialize a ProblemsSolvers object.

        There are three ways to initialize a ProblemsSolvers object:
        1. Provide the names of solvers and problems (for lookup in `directory.py`).
        2. Provide lists of solver and problem objects to pair directly.
        3. Provide a full list of `ProblemSolver` objects (as nested lists).

        Args:
            solver_factors (list[dict] | None): List of solver factor dictionaries,
                one per design point. Requires `solver_names` to match the number
                of entries.
            problem_factors (list[dict] | None): List of problem/model factor
                dictionaries, one per design point. Requires `problem_names` to match
                the number of entries.
            solver_names (list[str] | None): List of solver names to look up.
            problem_names (list[str] | None): List of problem names to look up.
            solver_renames (list[str] | None): User-specified labels for solvers.
            problem_renames (list[str] | None): User-specified labels for problems.
            fixed_factors_filename (str | None): Name of a `.py` file containing
                fixed factor dictionaries.
            solvers (list[Solver] | None): List of `Solver` objects to use directly.
            problems (list[Problem] | None): List of `Problem` objects to use directly.
            experiments (list[list[ProblemSolver]] | None): Explicit problem-solver
                pairings.
            file_name_path (Path | None): Output path for saving the
                `ProblemsSolvers` object.
            create_pair_pickles (bool): Whether to create individual `.pickle` files
                for each problem-solver pair.
            experiment_name (str | None): Optional name to prefix output files.
        """
        # set attributes for pickle create and experiment file names
        self.create_pair_pickles = create_pair_pickles
        if experiment_name is not None:
            self.file_header = f"{experiment_name}_"
            self.experiment_name = experiment_name
        else:
            self.file_header = ""
            self.experiment_name = ""
        # For some reason some of these variables weren't being assigned to the
        # class attributes. TODO: Fix this.

        output_dir = EXPERIMENT_DIR / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        if experiments is not None:  # Method #3
            self.experiments = experiments
            self.solvers = [
                experiments[idx][0].solver for idx in range(len(experiments))
            ]
            self.problems = [experiment.problem for experiment in experiments[0]]
            self.solver_names = [solver.name for solver in self.solvers]
            self.problem_names = [problem.name for problem in self.problems]
            self.solver_set = self.solver_names
            self.problem_set = self.problem_names

        elif solver_factors is not None and problem_factors is not None:
            # Create solvers list.
            solvers = []
            for index, dp in enumerate(solver_factors):
                if solver_names is None:
                    error_msg = "Solver names must be provided."
                    raise ValueError(error_msg)
                # Get corresponding name of solver from names.
                solver_name = solver_names[index]
                # Assign all factor values from current dp to solver object.
                solver = solver_directory[solver_name](fixed_factors=dp)
                solvers.append(solver)

            # Create problems list.
            problems = []
            for index, dp in enumerate(problem_factors):
                if problem_names is None:
                    error_msg = "Problem names must be provided."
                    raise ValueError(error_msg)
                # Get corresponding name of problem from names.
                problem_name = problem_names[index]
                # Will hold problem factor values for current dp.
                fixed_factors = {}
                # Will hold model factor values for current dp.
                model_fixed_factors = {}
                # Create default instances of problem and model to compare factor names.
                default_problem = problem_directory[problem_name]()
                default_model = default_problem.model

                # Set factor values for current dp using problem/model specifications
                # to determine if problem or model factor.
                for factor in dp:
                    if factor in default_problem.specifications:
                        fixed_factors[factor] = dp[factor]
                    if factor in default_model.specifications:
                        model_fixed_factors[factor] = dp[factor]
                # Create instance of problem and append to problems list.
                problem = problem_directory[problem_name](
                    fixed_factors=fixed_factors,
                    model_fixed_factors=model_fixed_factors,
                )
                problems.append(problem)
            # rename problems and solvers if applicable
            if solver_renames is not None:
                self.solver_renames = solver_renames
            else:
                if solver_names is None:
                    error_msg = "Solver names must be provided."
                    raise ValueError(error_msg)
                self.solver_renames = solver_names
            if problem_renames is not None:
                self.problem_renames = problem_renames
            else:
                if problem_names is None:
                    error_msg = "Problem names must be provided."
                    raise ValueError(error_msg)
                self.problem_renames = problem_names
            self.experiments = [
                [
                    ProblemSolver(
                        solver=solver,
                        problem=problem,
                        solver_rename=self.solver_renames[sol_indx],
                        problem_rename=self.problem_renames[prob_indx],
                        create_pickle=self.create_pair_pickles,
                        file_name_path=output_dir
                        / (
                            f"{self.file_header}"
                            f"{self.solver_renames[sol_indx]}_on_{self.problem_renames[prob_indx]}"
                        ),
                    )
                    for prob_indx, problem in enumerate(problems)
                ]
                for sol_indx, solver in enumerate(solvers)
            ]
            self.solvers = solvers
            self.problems = problems
            if solver_names is not None:
                self.solver_names = solver_names
            if problem_names is not None:
                self.problem_names = problem_names

            if solver_names is None:
                self.solver_set = None
            else:
                self.solver_set = set(solver_names)
            if problem_names is None:
                self.problem_set = None
            else:
                self.problem_set = set(problem_names)

        elif solvers is not None and problems is not None:  # Method #2
            self.experiments = [
                [ProblemSolver(solver=solver, problem=problem) for problem in problems]
                for solver in solvers
            ]
            self.solvers = solvers
            self.problems = problems
            self.solver_names = [solver.name for solver in self.solvers]
            self.problem_names = [problem.name for problem in self.problems]
            self.solver_set = self.solver_names
            self.problem_set = self.problem_names

        else:  # Method #1
            if solver_renames is None:
                if solver_names is None:
                    error_msg = "Solver names must be provided."
                    raise ValueError(error_msg)
                self.solver_names = solver_names
            else:
                self.solver_names = solver_renames
            if problem_renames is None:
                if problem_names is None:
                    error_msg = "Problem names must be provided."
                    raise ValueError(error_msg)
                self.problem_names = problem_names
            else:
                self.problem_names = problem_renames

            # Use this for naming file.
            self.solver_set = solver_names
            self.problem_set = problem_names
            # Read in fixed solver/problem/model factors from .py file in the
            # experiments folder.
            # File should contain three dictionaries of dictionaries called
            #   - all_solver_fixed_factors
            #   - all_problem_fixed_factors
            #   - all_model_fixed_factors
            if fixed_factors_filename is None:
                self.all_solver_fixed_factors = {
                    solver_name: {} for solver_name in self.solver_names
                }
                self.all_problem_fixed_factors = {
                    problem_name: {} for problem_name in self.problem_names
                }
                self.all_model_fixed_factors = {
                    problem_name: {} for problem_name in self.problem_names
                }
            else:
                fixed_factors_filename = "experiments.inputs." + fixed_factors_filename
                all_factors = importlib.import_module(fixed_factors_filename)
                self.all_solver_fixed_factors = all_factors.all_solver_fixed_factors
                self.all_problem_fixed_factors = all_factors.all_problem_fixed_factors
                self.all_model_fixed_factors = all_factors.all_model_fixed_factors
            # Create all problem-solver pairs (i.e., instances of ProblemSolver class).
            self.experiments = []
            for solver_idx in range(self.n_solvers):
                solver_name = self.solver_names[solver_idx]
                solver_experiments = []
                for problem_idx in range(self.n_problems):
                    problem_name = self.problem_names[problem_idx]
                    filename = f"{solver_name}_on_{problem_name}.pickle"
                    file_path = output_dir / filename
                    if file_path.exists():
                        with file_path.open("rb") as f:
                            loaded_exp = pickle.load(f)
                            solver_experiments.append(loaded_exp)
                        continue
                        # TODO: Check if the solver/problem/model factors in the file
                        # match those for the ProblemsSolvers.
                    if solver_names is None:
                        error_msg = "Solver names must be provided if no file exists."
                        raise ValueError(error_msg)
                    if problem_names is None:
                        error_msg = "Problem names must be provided if no file exists."
                        raise ValueError(error_msg)
                    # If no file exists, create new ProblemSolver object.
                    logging.debug(
                        f"No experiment file exists for {solver_name} on "
                        f"{problem_name}. "
                        "Creating new experiment."
                    )
                    # Lookup fixed factors for solver, problem, and model.
                    solver_ff = self.all_solver_fixed_factors[solver_name]
                    problem_ff = self.all_problem_fixed_factors[problem_name]
                    model_ff = self.all_model_fixed_factors[problem_name]
                    # Create new ProblemSolver object.
                    next_experiment = ProblemSolver(
                        solver_name=solver_names[solver_idx],
                        problem_name=problem_names[problem_idx],
                        solver_rename=solver_name,
                        problem_rename=problem_name,
                        solver_fixed_factors=solver_ff,
                        problem_fixed_factors=problem_ff,
                        model_fixed_factors=model_ff,
                    )
                    solver_experiments.append(next_experiment)
                self.experiments.append(solver_experiments)
                self.solvers = [
                    self.experiments[idx][0].solver
                    for idx in range(len(self.experiments))
                ]
                self.problems = [
                    experiment.problem for experiment in self.experiments[0]
                ]
        # Initialize file path.
        if file_name_path is None:
            solver_names_string = ""
            problem_names_string = ""
            if self.solver_set is not None:
                solver_names_string = "_".join(self.solver_set)
            if self.problem_set is not None:
                problem_names_string = "_".join(self.problem_set)
            s_on_p = f"{solver_names_string}_on_{problem_names_string}"
            file_name = f"{self.file_header}group_{s_on_p}.pickle"
            self.file_name_path = output_dir / file_name
        else:
            self.file_name_path = file_name_path

            self.solver_set = self.solver_names
            self.problem_set = self.problem_names

    def check_compatibility(self) -> str:
        """Check whether all experiments' solvers and problems are compatible.

        Returns:
            str: Error message in the event any problem and solver are incompatible.
        """
        errors = []
        for solver_idx in range(self.n_solvers):
            for problem_idx in range(self.n_problems):
                new_error_str = self.experiments[solver_idx][
                    problem_idx
                ].check_compatibility()
                if len(new_error_str) > 0:
                    new_msg = (
                        f"For solver {self.solver_names[solver_idx]} "
                        f"and problem {self.problem_names[problem_idx]}... "
                        + new_error_str
                    )
                    errors.append(new_msg)
        return "\n".join(errors)

    def run(self, n_macroreps: int) -> None:
        """Run `n_macroreps` of each solver on each problem.

        Args:
            n_macroreps (int): Number of macroreplications to run per problem-solver
                pair.

        Raises:
            ValueError: If `n_macroreps` is not positive.
        """
        # Value checking
        if n_macroreps <= 0:
            error_msg = "Number of macroreplications must be positive."
            raise ValueError(error_msg)

        for solver_idx in range(self.n_solvers):
            for problem_idx in range(self.n_problems):
                experiment = self.experiments[solver_idx][problem_idx]
                # If the problem-solver pair has not been run in this way before,
                # run it now and save result to .pickle file.
                if not experiment.has_run:
                    # TODO: check if this should be prob on solv instead?
                    s_on_p = f"{experiment.solver.name} on {experiment.problem.name}"
                    logging.debug(
                        f"Running {n_macroreps} macro-replications of {s_on_p}."
                    )
                    experiment.run(n_macroreps)
        # Save ProblemsSolvers object to .pickle file.
        self.record_group_experiment_results()

    def post_replicate(
        self,
        n_postreps: int,
        crn_across_budget: bool = True,
        crn_across_macroreps: bool = False,
    ) -> None:
        """Runs postreplications for each problem-solver pair on all macroreplications.

        Args:
            n_postreps (int): Number of postreplications per recommended solution.
            crn_across_budget (bool, optional): If True, use CRN across solutions from
                different time budgets. Defaults to True.
            crn_across_macroreps (bool, optional): If True, use CRN across solutions
                from different macroreplications. Defaults to False.

        Raises:
            ValueError: If n_postreps is not positive.
        """
        # Value checking
        if n_postreps <= 0:
            error_msg = "Number of postreplications must be positive."
            raise ValueError(error_msg)

        for solver_index in range(self.n_solvers):
            for problem_index in range(self.n_problems):
                experiment = self.experiments[solver_index][problem_index]
                # If the problem-solver pair has not been post-replicated in this
                # way before, post-process it now.
                if not experiment.has_postreplicated:
                    s_on_p = f"{experiment.solver.name} on {experiment.problem.name}"
                    logging.debug(f"Post-processing {s_on_p}.")
                    experiment.post_replicate(
                        n_postreps, crn_across_budget, crn_across_macroreps
                    )
        # Save ProblemsSolvers object to .pickle file.
        self.record_group_experiment_results()

    def post_normalize(
        self, n_postreps_init_opt: int, crn_across_init_opt: bool = True
    ) -> None:
        """Builds objective and progress curves for all experiment collections.

        Args:
            n_postreps_init_opt (int): Number of postreplications at initial (x0) and
                optimal (x*) solutions.
            crn_across_init_opt (bool, optional): If True, use CRN for postreplications
                at x0 and x*. Defaults to True.

        Raises:
            ValueError: If `n_postreps_init_opt` is not positive.
        """
        # Value checking
        if n_postreps_init_opt <= 0:
            error_msg = "Number of postreplications must be positive."
            raise ValueError(error_msg)

        for problem_idx in range(self.n_problems):
            experiments_same_problem = [
                self.experiments[solver_idx][problem_idx]
                for solver_idx in range(self.n_solvers)
            ]
            post_normalize(
                experiments=experiments_same_problem,
                n_postreps_init_opt=n_postreps_init_opt,
                crn_across_init_opt=crn_across_init_opt,
            )
        # Save ProblemsSolvers object to .pickle file.
        self.record_group_experiment_results()

    def check_postreplicate(self) -> bool:
        """Checks whether all experiments have been postreplicated.

        Returns:
            bool: Whether all experiments have been postreplicated
        """
        return all(
            experiment.has_postreplicated
            for row in self.experiments
            for experiment in row
        )

    def check_postnormalize(self) -> bool:
        """Checks whether all experiments have been postnormalized.

        Returns:
            bool: Whether all experiments have been postnormalized.
        """
        return all(
            experiment.has_postnormalized
            for row in self.experiments
            for experiment in row
        )

    def record_group_experiment_results(self) -> None:
        """Saves a ProblemsSolvers object to a .pickle file in the outputs directory."""
        output_dir = EXPERIMENT_DIR / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        with self.file_name_path.open("wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def log_group_experiment_results(self) -> None:
        """Creates a .txt summary of solvers and problems in the ProblemSolvers object.

        The file is saved in the 'logs/' folder next to the experiment's pickle file.
        """
        # Create a new text file in experiment/{date/time of launch}/logs folder
        # with correct name.
        log_dir = self.file_name_path.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        new_filename = self.file_name_path.name.replace(
            ".pickle", "_group_experiment_results.txt"
        )  # Remove .pickle from .txt file name.
        new_path = log_dir / new_filename

        # Create text file.
        with new_path.open("w") as file:
            seperator_len = 100
            # Title text file with experiment information.
            file.write(str(self.file_name_path))
            file.write("\n")
            # Write the name of each problem.
            file.write("-" * seperator_len)
            file.write("\nProblems:\n\n")
            for i in range(self.n_problems):
                file.write(f"{self.problem_names[i]}\n\t")
                # Write model factors for each problem.
                file.write("Model Factors:\n")
                for key, value in self.problems[i].model.factors.items():
                    # Excluding model factors corresponding to decision variables.
                    if key not in self.problems[i].model_decision_factors:
                        file.write(f"\t\t{key}: {value}\n")
                # Write problem factors for each problem.
                file.write("\n\tProblem Factors:\n")
                for key, value in self.problems[i].factors.items():
                    file.write(f"\t\t{key}: {value}\n")
                file.write("\n")
            file.write("-" * seperator_len)
            # Write the name of each Solver.
            file.write("\nSolvers:\n\n")
            # Write solver factors for each solver.
            for solv_idx in range(self.n_solvers):
                file.write(f"{self.solver_names[solv_idx]}\n")
                file.write("\tSolver Factors:\n")
                for key, value in self.solvers[solv_idx].factors.items():
                    file.write(f"\t\t{key}: {value}\n")
                file.write("\n")
            file.write("-" * seperator_len)
            # Write the name of pickle files for each Problem-Solver pair if created.
            if self.create_pair_pickles:
                file.write(
                    "\nThe .pickle files for the associated Problem-Solver pairs are:\n"
                )
                for solver_group in self.experiments:
                    for experiment in solver_group:
                        file_name = experiment.file_name_path.name
                        file.write(f"{file_name}\n")
            # for p in self.problem_names:
            #     for s in self.solver_names:
            #         file.write(f"\t{s}_on_{p}.pickle\n")

    def report_group_statistics(
        self,
        solve_tols: list[float] | None = None,
        csv_filename: str = "df_solver_results",
    ) -> None:
        """Reports statistics for all solvers across all problems.

        Args:
            solve_tols (list[float], optional): Optimality gaps defining when a problem
                is considered solved (values in (0, 1]).
                Defaults to [0.05, 0.10, 0.20, 0.50].
            csv_filename (str, optional): Name of the output CSV file (without '.csv').
                Defaults to "df_solver_results".

        Raises:
            ValueError: If any solve tolerance is not in the range (0, 1].
        """
        # Assign default values
        if solve_tols is None:
            solve_tols = [0.05, 0.10, 0.20, 0.50]
        # Value checking
        if not all(0 < tol <= 1 for tol in solve_tols):
            error_msg = "Solve tols must be in (0,1]."
            raise ValueError(error_msg)
        # TODO: figure out if we should also check for increasing order of solve_tols

        # create dictionary of common solvers and problems
        pair_dict = {}  # used to hold pairs of

        for sublist in self.experiments:
            for obj in sublist:
                solver = type(obj.solver).__name__
                problem = type(obj.problem).__name__
                key = (solver, problem)
                if key not in pair_dict:
                    pair_dict[key] = [obj]
                else:
                    pair_dict[key].append(obj)
        for (solver, problem), pair_list in pair_dict.items():
            csv_filename = f"{self.file_header}{solver}_on_{problem}_results"
            self.report_statistics(
                pair_list=pair_list,
                solve_tols=solve_tols,
                csv_filename=csv_filename,
            )

    def report_statistics(
        self,
        pair_list: list[ProblemSolver],
        solve_tols: list[float] | None = None,
        csv_filename: str = "df_solver_results",
    ) -> None:
        """Calculates statistics from macroreplications and saves results to a CSV file.

        Args:
            pair_list (list[ProblemSolver]): List of ProblemSolver objects.
            solve_tols (list[float], optional): Optimality gaps defining when a problem
                is considered solved (values in (0, 1]).
                Defaults to [0.05, 0.10, 0.20, 0.50].
            csv_filename (str, optional): Name of the CSV file to write results to.
                Defaults to "df_solver_results".

        Raises:
            ValueError: If any solve tolerance is not in the range (0, 1].
        """
        # Local imports
        import csv

        # Assign default values
        if solve_tols is None:
            solve_tols = [0.05, 0.10, 0.20, 0.50]
        # Value checking
        if not all(0 < tol <= 1 for tol in solve_tols):
            error_msg = "Solve tols must be in (0,1]."
            raise ValueError(error_msg)
        # TODO: figure out if we should also check for increasing order of solve_tols

        # Create directory if it does no exist.
        log_dir = EXPERIMENT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        file_path = log_dir / f"{csv_filename}.csv"
        with file_path.open(mode="w", newline="") as output_file:
            csv_writer = csv.writer(
                output_file,
                delimiter="\t",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            base_experiment = pair_list[0]
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
                    "SolverName",
                    *solver_factor_names,
                    "ProblemName",
                    *problem_factor_names,
                    *model_factor_names,
                    "MacroRep#",
                    "Final Relative Optimality Gap",
                    "Area Under Progress Curve",
                    *solve_time_headers,
                    "Initial Solution",
                    "Initial Objective Function Value",
                    "Optimal Solution",
                    "Optimal Objective Function Value",
                ]
            )
            # Compute performance metrics.
            for designpt_index in range(len(pair_list)):
                experiment = pair_list[designpt_index]
                solver_name = experiment.solver.name
                problem_name = experiment.problem.name
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
                                < float("inf")
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
                    init_sol = tuple([round(x, 4) for x in experiment.x0])
                    int_obj = experiment.x0_postreps[mrep]
                    opt_sol = tuple(
                        [round(x, 4) for x in experiment.all_recommended_xs[mrep][-1]]
                    )
                    opt_obj = experiment.all_est_objectives[mrep][-1]
                    solution_list = [init_sol, int_obj, opt_sol, opt_obj]
                    print_list = [
                        designpt_index,
                        solver_name,
                        *solver_factor_list,
                        problem_name,
                        *problem_factor_list,
                        *model_factor_list,
                        mrep,
                        *statistics_list,
                        *solution_list,
                    ]
                    csv_writer.writerow(print_list)
