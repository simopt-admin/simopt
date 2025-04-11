#!/usr/bin/env python
"""Provide base classes for solvers, problems, and models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.utils import classproperty


def _factor_check(self: Solver | Problem | Model, factor_name: str) -> bool:
    # Check if factor is of permissible data type.
    datatype_check = self.check_factor_datatype(factor_name)
    if not datatype_check:
        return False
    # Check if the factor check exists
    if factor_name not in self.check_factor_list:
        # If the factor is a boolean, it's fine
        if self.specifications[factor_name]["datatype"] is bool:
            return True
        # Raise an error since there's an error in the check list
        datatype = self.specifications[factor_name]["datatype"]
        error_msg = f"Missing check for factor {factor_name} of type {datatype}"
        raise ValueError(error_msg)
    # Otherwise, the factor exists in the check list and should be checked
    # This will raise an error if the factor is not permissible
    self.check_factor_list[factor_name]()
    # Return true if we successfully checked the factor
    return True


class ObjectiveType(Enum):
    """Enum class for objective types."""

    SINGLE = 1
    MULTI = 2

    def symbol(self) -> str:
        """Return the symbol of the objective type."""
        symbol_mapping = {ObjectiveType.SINGLE: "S", ObjectiveType.MULTI: "M"}
        return symbol_mapping.get(self, "?")


class ConstraintType(Enum):
    """Enum class for constraint types."""

    UNCONSTRAINED = 1
    BOX = 2
    DETERMINISTIC = 3
    STOCHASTIC = 4

    def symbol(self) -> str:
        """Return the symbol of the constraint type."""
        symbol_mapping = {
            ConstraintType.UNCONSTRAINED: "U",
            ConstraintType.BOX: "B",
            ConstraintType.DETERMINISTIC: "D",
            ConstraintType.STOCHASTIC: "S",
        }
        return symbol_mapping.get(self, "?")


class VariableType(Enum):
    """Enum class for variable types."""

    DISCRETE = 1
    CONTINUOUS = 2
    MIXED = 3

    def symbol(self) -> str:
        """Return the symbol of the variable type."""
        symbol_mapping = {
            VariableType.DISCRETE: "D",
            VariableType.CONTINUOUS: "C",
            VariableType.MIXED: "M",
        }
        return symbol_mapping.get(self, "?")


class Solver(ABC):
    """Base class to implement simulation-optimization solvers.

    This class defines the core structure for simulation-optimization
    solvers in SimOpt. Subclasses must implement the required methods
    for running simulations and updating solutions.

    Args:
        name (str): Name of the solver.
        fixed_factors (dict): Dictionary of user-specified solver factors.
    """

    @classproperty
    def class_name_abbr(cls) -> str:
        """Short name of the solver class."""
        return cls.__name__

    @classproperty
    def class_name(cls) -> str:
        """Long name of the solver class."""
        return cls.__name__.replace("_", " ")

    @classproperty
    def compatibility(cls) -> str:
        """Compatibility of the solver."""
        return (
            f"{cls.objective_type.symbol()}"
            f"{cls.constraint_type.symbol()}"
            f"{cls.variable_type.symbol()}"
            f"{'G' if cls.gradient_needed else 'N'}"
        )

    @property
    def name(self) -> str:
        """Name of solver."""
        return self.__name

    @name.setter
    def name(self, value: str) -> None:
        if len(value) == 0:
            raise ValueError("Name must not be empty.")
        self.__name = value

    @classproperty
    @abstractmethod
    def objective_type(cls) -> ObjectiveType:
        """Description of objective types.

        One of: "single" or "multi".
        """
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def constraint_type(cls) -> ConstraintType:
        """Description of constraint types.

        One of: "unconstrained", "box", "deterministic", or "stochastic".
        """
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def variable_type(cls) -> VariableType:
        """Description of variable types.

        One of: "discrete", "continuous", "mixed".
        """
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def gradient_needed(cls) -> bool:
        """True if gradient of objective function is needed, otherwise False."""
        raise NotImplementedError

    @property
    def factors(self) -> dict:
        """Changeable factors (i.e., parameters) of the solver."""
        return self.__factors

    @factors.setter
    def factors(self, value: dict | None) -> None:
        if value is None:
            value = {}
        self.__factors = value

    @classproperty
    @abstractmethod
    def specifications(
        cls,
    ) -> dict[str, dict]:
        """Details of each factor (for GUI, data validation, and defaults)."""
        raise NotImplementedError

    @property
    def rng_list(self) -> list[MRG32k3a]:
        """List of RNGs used for the solver's internal purposes."""
        return self.__rng_list

    @rng_list.setter
    def rng_list(self, value: list[MRG32k3a]) -> None:
        self.__rng_list = value

    @property
    def solution_progenitor_rngs(self) -> list[MRG32k3a]:
        """List of RNGs used as a baseline for simulating solutions."""
        return self.__solution_progenitor_rngs

    @solution_progenitor_rngs.setter
    def solution_progenitor_rngs(self, value: list[MRG32k3a]) -> None:
        self.__solution_progenitor_rngs = value

    @property
    @abstractmethod
    def check_factor_list(self) -> dict[str, Callable]:
        """Dictionary of functions to check if a factor is permissible."""
        raise NotImplementedError

    def __init__(self, name: str = "", fixed_factors: dict | None = None) -> None:
        """Initialize a solver object.

        Args:
            name (str, optional): Name of the solver. Defaults to an empty string.
            fixed_factors (dict | None, optional): Dictionary of user-specified solver
                factors. Defaults to None.
        """
        self.name = name
        # Add all the fixed factors to the solver
        self.factors = fixed_factors
        all_factors = set(self.specifications.keys())
        present_factors = set(self.factors.keys())
        missing_factors = all_factors - present_factors
        for factor in missing_factors:
            self.factors[factor] = self.specifications[factor]["default"]
        # Run checks
        factor_names = list(self.factors.keys())
        self.run_all_checks(factor_names=factor_names)

    def __eq__(self, other: object) -> bool:
        """Check if two solvers are equivalent.

        Args:
            other (object): Other object to compare to self.

        Returns:
            bool: True if the two objects are equivalent, otherwise False.
        """
        if not isinstance(other, Solver):
            return False
        return type(self) is type(other) and self.factors == other.factors

    def __hash__(self) -> int:
        """Return the hash value of the solver.

        Returns:
            int: Hash value of the solver.
        """
        return hash((self.name, tuple(self.factors.items())))

    def attach_rngs(self, rng_list: list[MRG32k3a]) -> None:
        """Attach a list of random-number generators to the solver.

        Args:
            rng_list (list[``mrg32k3a.mrg32k3a.MRG32k3a``]): List of random-number
                generators used for the solver's internal purposes.
        """
        self.rng_list = rng_list

    @abstractmethod
    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """Run a single macroreplication of a solver on a problem.

        Args:
            problem (Problem): Simulation-optimization problem to solve.

        Returns:
            tuple:
                - list [Solution]: List of solutions recommended throughout the budget.
                - list [int]: List of intermediate budgets when recommended solutions
                    change.
        """
        raise NotImplementedError

    def check_crn_across_solns(self) -> bool:
        """Check solver factor crn_across_solns.

        Returns:
            bool: True if the solver factor is permissible, otherwise False.
        """
        # NOTE: Currently implemented to always return True
        return True

    def check_solver_factor(self, factor_name: str) -> bool:
        """Determine if the setting of a solver factor is permissible.

        Args:
            factor_name (str): Name of factor for dictionary lookup (i.e., key).

        Returns:
            bool: True if the solver factor is permissible, otherwise False.
        """
        return _factor_check(self, factor_name)

    # TODO: Figure out if this should be abstract or not
    # @abstractmethod
    def check_solver_factors(self) -> bool:
        """Determine if the joint settings of solver factors are permissible.

        Returns:
            bool: True if the solver factors are permissible, otherwise False.
        """
        return True

    def check_factor_datatype(self, factor_name: str) -> bool:
        """Determine if a factor's data type matches its specification.

        Args:
            factor_name (str): The name of the factor to check.

        Returns:
            bool: True if factor is of specified data type, otherwise False.
        """
        expected_data_type = self.specifications[factor_name]["datatype"]
        return isinstance(self.factors[factor_name], expected_data_type)

    def run_all_checks(self, factor_names: list[str]) -> bool:
        """Run all checks for the solver factors.

        Args:
            factor_names (list[str]): A list of factor names to check.

        Returns:
            bool: True if all checks are passed, otherwise False.
        """
        is_joint_factors = (
            self.check_solver_factors()
        )  # check all joint factor settings

        if not is_joint_factors:
            error_msg = (
                "There is a joint setting of a solver factor that is not permissible"
            )
            raise ValueError(error_msg)

        # check datatypes for all factors
        for factor in factor_names:
            is_right_type = self.check_factor_datatype(factor)
            if not is_right_type:
                error_msg = f"Solver factor {factor} is not the correct data type."
                raise ValueError(error_msg)

            is_permissible = self.check_solver_factor(factor)
            if not is_permissible:
                error_msg = f"Solver factor {factor} is not permissible."
                raise ValueError(error_msg)

        # Return true if no issues
        return True

    def create_new_solution(self, x: tuple, problem: Problem) -> Solution:
        """Create a new solution object with attached RNGs.

        Args:
            x (tuple): A vector of decision variables.
            problem (Problem): The problem instance associated with the solution.

        Returns:
            Solution: New solution object for the given decision variables and problem.
        """
        # Create new solution with attached rngs.
        new_solution = Solution(x, problem)
        new_solution.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
        # Manipulate progenitor rngs to prepare for next new solution.
        if not self.factors["crn_across_solns"]:  # If CRN are not used ...
            # ...advance each rng to start of the substream
            # substream = current substream + # of model RNGs.
            for rng in self.solution_progenitor_rngs:
                for _ in range(problem.model.n_rngs):
                    rng.advance_substream()
        return new_solution

    def rebase(self, n_reps: int) -> None:
        """Rebase the progenitor rngs to start at a later subsubstream index.

        Args:
            n_reps (int): Substream index to skip to.
        """
        new_rngs = []
        for rng in self.solution_progenitor_rngs:
            stream_index = rng.s_ss_sss_index[0]
            substream_index = rng.s_ss_sss_index[1]
            new_rngs.append(
                MRG32k3a(s_ss_sss_index=[stream_index, substream_index, n_reps])
            )
        self.solution_progenitor_rngs = new_rngs


class Problem(ABC):
    """Base class for simulation-optimization problems.

    Args:
        name (str): Problem name.
        fixed_factors (dict): User-defined factors that affect the problem setup.
        model_fixed_factors (dict): Subset of non-decision factors passed to the model.
        model (Callable[..., Model]): Simulation model that generates replications.
    """

    @classproperty
    def class_name_abbr(cls) -> str:
        """Short name of the solver class."""
        return cls.__name__

    @classproperty
    def class_name(cls) -> str:
        """Long name of the solver class."""
        return cls.__name__.replace("_", " ")

    @classproperty
    def compatibility(cls) -> str:
        """Compatibility of the solver."""
        return (
            "S"
            f"{cls.constraint_type.symbol()}"
            f"{cls.variable_type.symbol()}"
            f"{'G' if cls.gradient_available else 'N'}"
        )

    @property
    def name(self) -> str:
        """Name of the problem."""
        return self.__name

    @name.setter
    def name(self, value: str) -> None:
        if len(value) == 0:
            raise ValueError("Name must not be empty.")
        self.__name = value

    @classproperty
    @abstractmethod
    def dim(cls) -> int:
        """Number of decision variables."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def n_objectives(cls) -> int:
        """Number of objectives."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def n_stochastic_constraints(cls) -> int:
        """Number of stochastic constraints."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def minmax(cls) -> tuple[int]:
        """Indicators of maximization (+1) or minimization (-1) for each objective."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def constraint_type(cls) -> ConstraintType:
        """Description of constraints types.

        One of: "unconstrained", "box", "deterministic", "stochastic".
        """
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def variable_type(cls) -> VariableType:
        """Description of variable types.

        One of: "discrete", "continuous", "mixed".
        """
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def lower_bounds(cls) -> tuple:
        """Lower bound for each decision variable."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def upper_bounds(cls) -> tuple:
        """Upper bound for each decision variable."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def gradient_available(cls) -> bool:
        """Indicates whether the solver provides direct gradient information."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def optimal_value(cls) -> float | None:
        """Optimal objective function value."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def optimal_solution(cls) -> tuple | None:
        """Optimal solution."""
        raise NotImplementedError

    @property
    def model(self) -> Model:
        """Associated simulation model that generates replications."""
        return self.__model

    @model.setter
    def model(self, value: Model) -> None:
        self.__model = value

    @classproperty
    @abstractmethod
    def model_default_factors(cls) -> dict:
        """Default values for overriding model-level default factors."""
        raise NotImplementedError

    @property
    def model_fixed_factors(self) -> dict:
        """Combination of overriden model-level factors and defaults."""
        return self.__model_fixed_factors

    @model_fixed_factors.setter
    def model_fixed_factors(self, value: dict | None) -> None:
        if value is None:
            value = {}
        self.__model_fixed_factors = value

    @classproperty
    @abstractmethod
    def model_decision_factors(cls) -> set[str]:
        """Set of keys for factors that are decision variables."""
        raise NotImplementedError

    @property
    def rng_list(self) -> list[MRG32k3a]:
        """List of RNGs used to generate a random initial solution/problem instance."""
        return self.__rng_list

    @rng_list.setter
    def rng_list(self, value: list[MRG32k3a]) -> None:
        self.__rng_list = value

    @property
    def factors(self) -> dict:
        """Changeable factors of the problem."""
        return self.__factors

    @factors.setter
    def factors(self, value: dict | None) -> None:
        if value is None:
            value = {}
        self.__factors = value

    @classproperty
    @abstractmethod
    def specifications(cls) -> dict:
        """Details of each factor (for GUI, data validation, and defaults)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def check_factor_list(self) -> dict:
        """Dictionary of functions to check if a factor is permissible."""
        raise NotImplementedError

    def __init__(
        self,
        name: str = "",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
        model: Callable[..., Model] | None = None,
    ) -> None:
        """Initialize a problem object.

        Args:
            name (str): Name of the problem.
            fixed_factors (dict | None): Dictionary of user-specified problem factors.
            model_fixed_factors (dict | None): Subset of user-specified non-decision
                factors passed to the model.
            model (Callable[..., Model] | None): Simulation model that generates
                replications.
        """
        if model is None:
            raise ValueError("Model must be specified.")
        # Assign the name of the problem
        self.name = name

        # Add all the fixed factors to the problem
        self.factors = fixed_factors
        all_factors = set(self.specifications.keys())
        present_factors = set(self.factors.keys())
        missing_factors = all_factors - present_factors
        for factor in missing_factors:
            self.factors[factor] = self.specifications[factor]["default"]

        # Add all the fixed factors to the model
        self.model_fixed_factors = model_fixed_factors
        all_model_factors = set(self.model_default_factors.keys())
        present_model_factors = set(self.model_fixed_factors.keys())
        missing_model_factors = all_model_factors - present_model_factors
        for factor in missing_model_factors:
            self.model_fixed_factors[factor] = self.model_default_factors[factor]

        # Set the model
        self.model = model(self.model_fixed_factors)

        keys = list(self.factors.keys())
        self.run_all_checks(factor_names=keys)

    def __eq__(self, other: object) -> bool:
        """Check if two problems are equivalent.

        Args:
            other (object): Other object to compare to self.

        Returns:
            bool: True if the two problems are equivalent, otherwise False.
        """
        if not isinstance(other, Problem):
            return False
        if type(self) is type(other) and self.factors == other.factors:
            # Check if non-decision-variable factors of models are the same.
            non_decision_factors = (
                set(self.model.factors.keys()) - self.model_decision_factors
            )
            for factor in non_decision_factors:
                if self.model.factors[factor] != other.model.factors[factor]:
                    return False
            return True
        return False

    def __hash__(self) -> int:
        """Return the hash value of the solver.

        Returns:
            int: Hash value of the solver.
        """
        non_decision_factors = (
            set(self.model.factors.keys()) - self.model_decision_factors
        )
        return hash(
            (
                self.name,
                tuple(self.factors.items()),
                tuple([(key, self.model.factors[key]) for key in non_decision_factors]),
            )
        )

    def check_initial_solution(self) -> bool:
        """Check if initial solution is feasible and of correct dimension.

        Returns:
            bool: True if initial solution is feasible and of correct dimension;
                False otherwise.
        """
        return True

    def check_budget(self) -> bool:
        """Check if budget is strictly positive.

        Returns:
            bool: True if budget is strictly positive, otherwise False.
        """
        return self.factors["budget"] > 0

    def check_problem_factor(self, factor_name: str) -> bool:
        """Determine if the setting of a problem factor is permissible.

        Args:
            factor_name (str): The name of the factor to check

        Returns:
            bool: True if the factor setting is permissible; False otherwise.
        """
        return _factor_check(self, factor_name)

    # NOTE: This was originally supposed to be an abstract method, but only
    # SPSA actually implements it. It's currently not clear if this
    # method should be implemented in other Problems as well.
    # @abstractmethod
    def check_problem_factors(self) -> bool:
        """Determine if the joint settings of problem factors are permissible.

        Returns:
            bool: True if problem factors are permissible; False otherwise.
        """
        return True

    def check_factor_datatype(self, factor_name: str) -> bool:
        """Determine if a factor's data type matches its specification.

        Args:
            factor_name (str): The name of the factor to check.

        Returns:
            bool: True if factor is of specified data type, otherwise False.
        """
        return isinstance(
            self.factors[factor_name],
            self.specifications[factor_name]["datatype"],
        )

    def run_all_checks(self, factor_names: list[str]) -> bool:
        """Run all checks for the problem factors.

        Args:
            factor_names (list[str]): A list of factor names to check.

        Returns:
            bool: True if all checks are passed, otherwise False.
        """
        is_joint_factors = (
            self.check_problem_factors()
        )  # check all joint factor settings
        if not is_joint_factors:
            error_msg = (
                "There is a joint setting of a problem factor that is not permissible"
            )
            raise ValueError(error_msg)

        is_initial_sol = self.check_initial_solution()
        if not is_initial_sol:
            error_msg = (
                "The initial solution is not feasible and/or not correct dimension"
            )
            raise ValueError(error_msg)

        # TODO: investigate why this is not working
        # is_budget = self.check_budget()
        if isinstance(self.factors["budget"], int) and self.factors["budget"] <= 0:
            error_msg = "The budget is not positive."
            raise ValueError(error_msg)

        # check datatypes for all factors
        for factor in factor_names:
            is_permissible = self.check_problem_factor(factor)
            is_right_type = self.check_factor_datatype(factor)

            if not is_right_type:
                error_msg = f"Problem factor {factor} is not a permissible data type."
                raise ValueError(error_msg)

            if not is_permissible:
                error_msg = f"Problem factor {factor} is not permissible."
                raise ValueError(error_msg)

        return True

    def attach_rngs(self, rng_list: list[MRG32k3a]) -> None:
        """Attach a list of random-number generators to the problem.

        Args:
            rng_list (list[``mrg32k3a.mrg32k3a.MRG32k3a``]): List of random-number
                generators used to generate a random initial solution or a random
                problem instance.
        """
        self.rng_list = rng_list

    @abstractmethod
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """Convert a vector of variables to a dictionary with factor keys.

        Args:
            vector (tuple): A vector of values associated with decision variables.

        Returns:
            dict: Dictionary with factor keys and associated values.
        """
        raise NotImplementedError

    @abstractmethod
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """Convert a dictionary with factor keys to a vector of variables.

        Args:
            factor_dict (dict): Dictionary with factor keys and associated values.

        Returns:
            tuple: Vector of values associated with decision variables.
        """
        raise NotImplementedError

    def factor_dict_to_vector_gradients(self, factor_dict: dict) -> tuple:
        """Convert a dictionary with factor keys to a gradient vector.

        A subclass of ``base.Problem`` can have its own custom
        ``factor_dict_to_vector_gradients`` method if the objective is deterministic.

        Args:
            factor_dict (dict): A dictionary with factor keys and associated values.

        Returns:
            tuple: Vector of partial derivatives associated with decision variables.
        """
        return self.factor_dict_to_vector(factor_dict)

    @abstractmethod
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        """Convert a dictionary with response keys to a vector of objectives.

        Args:
            response_dict (dict): A dictionary containing response keys and their
                associated values.

        Returns:
            tuple: Vector of objectives.
        """
        raise NotImplementedError

    def response_dict_to_objectives_gradients(self, _response_dict: dict) -> tuple:
        """Convert a dictionary with response keys to a vector of gradients.

        Can be overridden by subclasses if the objective is deterministic.

        Args:
            response_dict (dict): A dictionary containing response keys and their
                associated values.

        Returns:
            tuple: Vector of gradients.
        """
        return self.response_dict_to_objectives(_response_dict)

    def response_dict_to_stoch_constraints(self, _response_dict: dict) -> tuple:
        """Convert a response dictionary to a vector of stochastic constraint values.

        Each returned value represents the left-hand side of a constraint of the form
        E[Y] ≤ 0.

        Args:
            response_dict (dict): A dictionary containing response keys and their
                associated values.

        Returns:
            tuple: A tuple representing the left-hand sides of the stochastic
                constraints.
        """
        return ()

    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        """Compute deterministic components of objectives for a solution `x`.

        Args:
            x (tuple): A vector of decision variables.

        Returns:
            tuple:
                - tuple: The deterministic components of the objective values.
                - tuple: The gradients of those deterministic components.
        """
        det_objectives = (0,) * self.n_objectives
        det_objectives_gradients = tuple(
            [(0,) * self.dim for _ in range(self.n_objectives)]
        )
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self) -> tuple[tuple, tuple]:
        """Compute deterministic components of stochastic constraints.

        Returns:
            tuple:
                - tuple: The deterministic components of the stochastic constraints.
                - tuple: The gradients of those deterministic components.
        """
        det_stoch_constraints = (0,) * self.n_stochastic_constraints
        det_stoch_constraints_gradients = tuple(
            [(0,) * self.dim for _ in range(self.n_stochastic_constraints)]
        )
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, _x: tuple) -> bool:
        """Check if a solution `x` satisfies the problem's deterministic constraints.

        Args:
            x (tuple): A vector of decision variables.

        Returns:
            bool: True if the solution satisfies all deterministic constraints;
                False otherwise.
        """
        # Check box constraints.
        return all(
            lb <= x_i <= ub
            for x_i, lb, ub in zip(_x, self.lower_bounds, self.upper_bounds)
        )

    @abstractmethod
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """Generate a random solution for starting or restarting solvers.

        Args:
            rand_sol_rng (MRG32k3a): Random number generator used to sample the
                solution.

        Returns:
            tuple: A tuple representing a randomly generated vector of decision
                variables.
        """
        raise NotImplementedError

    def simulate(self, solution: Solution, num_macroreps: int = 1) -> None:
        """Simulate `m` i.i.d. replications at solution `x`.

        Args:
            solution (Solution): Solution to evaluate.
            num_macroreps (int, optional): Number of macroreplications to simulate at
                `x`. Defaults to 1.
        """
        # NOTE:
        # Gradients of objective function and stochastic constraint LHSs are temporarily
        # commented out. Under development.

        # Value checking
        if num_macroreps <= 0:
            error_msg = "Number of replications must be at least 1."
            raise ValueError(error_msg)
        # Pad numpy arrays if necessary.
        if solution.n_reps + num_macroreps > solution.storage_size:
            solution.pad_storage(num_macroreps)
        # Set the decision factors of the model.
        self.model.factors.update(solution.decision_factors)
        for _ in range(num_macroreps):
            # Generate one replication at x.
            responses, gradients = self.model.replicate(solution.rng_list)
            # Convert gradient subdictionaries to vectors mapping to decision variables.
            vector_gradients = {}
            if self.gradient_available:
                vector_gradients = {
                    keys: self.factor_dict_to_vector_gradients(gradient_dict)
                    for (keys, gradient_dict) in gradients.items()
                }
                # vector_gradients = {
                #   keys: self.factor_dict_to_vector(gradient_dict)
                #   for (keys, gradient_dict) in gradients.items()
                # }
            # Convert responses and gradients to objectives and gradients and add
            # to those of deterministic components of objectives.
            solution.objectives[solution.n_reps] = [
                sum(pairs)
                for pairs in zip(
                    self.response_dict_to_objectives(responses),
                    solution.det_objectives,
                )
            ]
            if self.gradient_available:
                solution.objectives_gradients[solution.n_reps] = [
                    [sum(pairs) for pairs in zip(stoch_obj, det_obj)]
                    for stoch_obj, det_obj in zip(
                        self.response_dict_to_objectives_gradients(vector_gradients),
                        solution.det_objectives_gradients,
                    )
                ]
                # solution.objectives_gradients[solution.n_reps] = [
                #     [sum(pairs) for pairs in zip(stoch_obj, det_obj)]
                #     for stoch_obj, det_obj in zip(
                #         self.response_dict_to_objectives(vector_gradients),
                #         solution.det_objectives_gradients,
                #     )
                # ]
            if (
                self.n_stochastic_constraints > 0
                and solution.stoch_constraints is not None
            ):
                # Convert responses and gradients to stochastic constraints and
                # gradients and addto those of deterministic components of stochastic
                # constraints.
                solution.stoch_constraints[solution.n_reps] = [
                    sum(pairs)
                    for pairs in zip(
                        self.response_dict_to_stoch_constraints(responses),
                        solution.det_stoch_constraints,
                    )
                ]
                # solution.stoch_constraints_gradients[solution.n_reps] = [
                #     [sum(pairs) for pairs in zip(stoch_stoch_cons, det_stoch_cons)]
                #     for stoch_stoch_cons, det_stoch_cons in zip(
                #         self.response_dict_to_stoch_constraints(vector_gradients),
                #         solution.det_stoch_constraints_gradients,
                #     )
                # ]
            # Increment counter.
            solution.n_reps += 1
            # Advance rngs to start of next subsubstream.
            for rng in solution.rng_list:
                rng.advance_subsubstream()

    def simulate_up_to(self, solutions: list[Solution], n_reps: int) -> None:
        """Simulate a list of solutions up to a given number of replications.

        Args:
            solutions (list[Solution]): List of Solution objects to simulate.
            n_reps (int): Common number of replications to simulate each solution up to.

        Raises:
            TypeError: If `solutions` is not a list of Solution objects or if `n_reps`
                is not an integer.
            ValueError: If `n_reps` is less than or equal to 0.
        """
        # Type checking
        if not isinstance(solutions, list) or not all(
            isinstance(solution, Solution) for solution in solutions
        ):
            error_msg = "Input solutions must be a list of Solution objects."
            raise TypeError(error_msg)
        if not isinstance(n_reps, int):
            error_msg = "Number of replications must be an integer."
            raise TypeError(error_msg)
        # Value checking
        if n_reps <= 0:
            error_msg = "Number of replications must be at least 1."
            raise ValueError(error_msg)

        for solution in solutions:
            # If more replications needed, take them.
            if solution.n_reps < n_reps:
                n_reps_to_take = n_reps - solution.n_reps
                self.simulate(solution=solution, num_macroreps=n_reps_to_take)


class Model(ABC):
    """Base class for simulation models used in simulation-optimization problems.

    Each model defines the simulation logic behind a given problem instance.
    """

    @classproperty
    def class_name_abbr(cls) -> str:
        """Short name of the model class."""
        return cls.__name__.capitalize()

    @classproperty
    def class_name(cls) -> str:
        """Long name of the model class."""
        # Insert spaces before capital letters
        import re

        return re.sub(r"(?<!^)(?=[A-Z])", " ", cls.__name__)

    @classproperty
    def name(cls) -> str:
        """Name of model."""
        return cls.__name__.replace("_", " ")

    @classproperty
    @abstractmethod
    def n_rngs(cls) -> int:
        """Number of random-number generators used to run a simulation replication."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def n_responses(cls) -> int:
        """Number of responses (performance measures)."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def specifications(cls) -> dict[str, dict]:
        """Details of each factor (for GUI, data validation, and defaults)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def check_factor_list(self) -> dict[str, Callable]:
        """Switch case for checking factor simulatability."""
        raise NotImplementedError

    @property
    def factors(self) -> dict:
        """Changeable factors of the simulation model."""
        return self.__factors

    @factors.setter
    def factors(self, value: dict | None) -> None:
        if value is None:
            value = {}
        self.__factors = value

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize a model object.

        Args:
            fixed_factors (dict | None, optional): Dictionary of user-specified model
                factors.
        """
        # Add all the fixed factors to the model
        self.factors = fixed_factors
        all_factors = set(self.specifications.keys())
        present_factors = set(self.factors.keys())
        missing_factors = all_factors - present_factors
        for key in missing_factors:
            self.factors[key] = self.specifications[key]["default"]

        factor_names = list(self.factors.keys())
        self.run_all_checks(factor_names=factor_names)

    def __eq__(self, other: object) -> bool:
        """Check if two models are equivalent.

        Args:
            other (object): Other object to compare to self.

        Returns:
            bool: True if the two models are equivalent, otherwise False.
        """
        if not isinstance(other, Model):
            return False
        return type(self) is type(other) and self.factors == other.factors

    def __hash__(self) -> int:
        """Return the hash value of the model.

        Returns:
            int: Hash value of the model.
        """
        return hash((self.name, tuple(self.factors.items())))

    def check_simulatable_factor(self, factor_name: str) -> bool:
        """Determine if a simulation replication can be run with the given factor.

        Args:
            factor_name (str): Name of factor for dictionary lookup (i.e., key).

        Returns:
            bool: True if the model specified by factors is simulatable;
                False otherwise.
        """
        return _factor_check(self, factor_name)

    def check_simulatable_factors(self) -> bool:
        """Determine whether a simulation can be run with the current factor settings.

        Each subclass of `Model` can override this method to implement custom logic.
        If not overridden, this base implementation returns True.

        Returns:
            bool: True if the model configuration is considered simulatable;
                False otherwise.
        """
        return True

    def check_factor_datatype(self, factor_name: str) -> bool:
        """Determine if a factor's data type matches its specification.

        Args:
            factor_name (str): The name of the factor to check.

        Returns:
            bool: True if factor is of specified data type, otherwise False.
        """
        datatype = self.specifications[factor_name]["datatype"]
        if datatype is float:
            datatype = (int, float)
        return isinstance(self.factors[factor_name], datatype)

    def run_all_checks(self, factor_names: list[str]) -> bool:
        """Run all checks for the model factors.

        Args:
            factor_names (list[str]): A list of factor names to check.

        Returns:
            bool: True if all checks are passed, otherwise False.

        Raises:
            ValueError: If any of the checks fail.
        """
        is_joint_factors = (
            self.check_simulatable_factors()
        )  # check all joint factor settings

        if not is_joint_factors:
            error_msg = (
                "There is a joint setting of a model factor that is not permissible"
            )
            raise ValueError(error_msg)

        # check datatypes for all factors
        for factor in factor_names:
            is_right_type = self.check_factor_datatype(factor)
            if not is_right_type:
                error_msg = f"Model factor {factor} is not a permissible data type."
                raise ValueError(error_msg)

            is_permissible = self.check_simulatable_factor(factor)
            if not is_permissible:
                error_msg = f"Model factor {factor} is not permissible."
                raise ValueError(error_msg)

        return True

    @abstractmethod
    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[``mrg32k3a.mrg32k3a.MRG32k3a``]): List of random-number
                generators used to generate a random replication.

        Returns:
            tuple:
                - dict: Performance measures of interest.
                - dict: Gradient estimates for each response.
        """
        raise NotImplementedError


class Solution:
    """Base class for solutions in simulation-optimization problems.

    Solutions are represented by a vector of decision variables and a dictionary of
    associated decision factors.
    """

    @property
    def x(self) -> tuple:
        """Vector of decision variables."""
        return self.__x

    @x.setter
    def x(self, value: tuple) -> None:
        self.__x = value
        self.__dim = len(value)

    @property
    def dim(self) -> int:
        """Number of decision variables describing `x`."""
        return self.__dim

    @property
    def decision_factors(self) -> dict:
        """Decision factor names and values."""
        return self.__decision_factors

    @decision_factors.setter
    def decision_factors(self, value: dict) -> None:
        self.__decision_factors = value

    @property
    def rng_list(self) -> list[MRG32k3a]:
        """RNGs for model to use when running replications at the solution."""
        return self.__rng_list

    @rng_list.setter
    def rng_list(self, value: list[MRG32k3a]) -> None:
        self.__rng_list = value

    @property
    def n_reps(self) -> int:
        """Number of replications run at the solution."""
        return self.__n_reps

    @n_reps.setter
    def n_reps(self, value: int) -> None:
        self.__n_reps = value

    @property
    def det_objectives(self) -> tuple:
        """Deterministic components added to objectives."""
        return self.__det_objectives

    @det_objectives.setter
    def det_objectives(self, value: tuple) -> None:
        self.__det_objectives = value

    @property
    def det_objectives_gradients(self) -> tuple[tuple]:
        """Gradients of deterministic components added to objectives.

        # objectives x dimension.
        """
        return self.__det_objectives_gradients

    @det_objectives_gradients.setter
    def det_objectives_gradients(self, value: tuple[tuple]) -> None:
        self.__det_objectives_gradients = value

    @property
    def det_stoch_constraints(self) -> tuple:
        """Deterministic components added to LHS of stochastic constraints."""
        return self.__det_stoch_constraints

    @det_stoch_constraints.setter
    def det_stoch_constraints(self, value: tuple) -> None:
        self.__det_stoch_constraints = value

    @property
    def det_stoch_constraints_gradients(self) -> tuple[tuple]:
        """Gradients of deterministic components added to LHS stochastic constraints.

        # stochastic constraints x dimension.
        """
        return self.__det_stoch_constraints_gradients

    @det_stoch_constraints_gradients.setter
    def det_stoch_constraints_gradients(self, value: tuple[tuple]) -> None:
        self.__det_stoch_constraints_gradients = value

    @property
    def storage_size(self) -> int:
        """Max number of replications that can be recorded in current storage."""
        return self.__storage_size

    @storage_size.setter
    def storage_size(self, value: int) -> None:
        self.__storage_size = value

    @property
    def objectives(self) -> np.ndarray:
        """Objective(s) estimates from each replication.

        # replications x # objectives.
        """
        return self.__objectives

    @objectives.setter
    def objectives(self, value: np.ndarray) -> None:
        self.__objectives = value

    @property
    def objectives_gradients(self) -> np.ndarray:
        """Gradient estimates of objective(s) from each replication.

        # replications x # objectives x dimension.
        """
        return self.__objectives_gradients

    @objectives_gradients.setter
    def objectives_gradients(self, value: np.ndarray) -> None:
        self.__objectives_gradients = value

    @property
    def stochastic_constraints(self) -> np.ndarray:
        """Stochastic constraint estimates from each replication.

        # replications x # stochastic constraints.
        """
        return self.__stochastic_constraints

    @stochastic_constraints.setter
    def stochastic_constraints(self, value: np.ndarray) -> None:
        self.__stochastic_constraints = value

    @property
    def stochastic_constraints_gradients(self) -> np.ndarray:
        """Gradient estimates of stochastic constraints from each replication.

        # replications x # stochastic constraints x dimension.
        """
        return self.__stochastic_constraints_gradients

    @stochastic_constraints_gradients.setter
    def stochastic_constraints_gradients(self, value: np.ndarray) -> None:
        self.__stochastic_constraints_gradients = value

    @property
    def objectives_mean(self) -> np.ndarray:
        """Mean of objectives."""
        mean = np.mean(self.objectives[: self.n_reps], axis=0)
        return np.round(mean, 15)

    @property
    def objectives_var(self) -> np.ndarray:
        """Variance of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        var = np.var(self.objectives[: self.n_reps], axis=0, ddof=1)
        return np.round(var, 15)

    @property
    def objectives_stderr(self) -> np.ndarray:
        """Standard error of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        std = np.std(self.objectives[: self.n_reps], axis=0, ddof=1)
        sqrt = np.sqrt(self.n_reps)
        return np.round(std / sqrt, 15)

    @property
    def objectives_cov(self) -> np.ndarray:
        """Covariance of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        cov = np.cov(self.objectives[: self.n_reps], rowvar=False, ddof=1)
        return np.round(cov, 15)

    @property
    def objectives_gradients_mean(self) -> np.ndarray:
        """Mean of gradients of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        mean = np.mean(self.objectives_gradients[: self.n_reps], axis=0)
        return np.round(mean, 15)

    @property
    def objectives_gradients_var(self) -> np.ndarray:
        """Variance of gradients of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        var = np.var(self.objectives_gradients[: self.n_reps], axis=0, ddof=1)
        return np.round(var, 15)

    @property
    def objectives_gradients_stderr(self) -> np.ndarray:
        """Standard error of gradients of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        std = np.std(self.objectives_gradients[: self.n_reps], axis=0, ddof=1)
        sqrt = np.sqrt(self.n_reps)
        return np.round(std / sqrt, 15)

    @property
    def objectives_gradients_cov(self) -> np.ndarray:
        """Covariance of gradients of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        return np.array(
            [
                np.cov(
                    self.objectives_gradients[: self.n_reps, obj],
                    rowvar=False,
                    ddof=1,
                )
                for obj in range(len(self.det_objectives))
            ]
        )

    @property
    def stoch_constraints_mean(self) -> np.ndarray:
        """Mean of stochastic constraints."""
        if self.stoch_constraints is None:
            return np.array([])
        mean = np.mean(self.stoch_constraints[: self.n_reps], axis=0)
        return np.round(mean, 15)

    @property
    def stoch_constraints_var(self) -> np.ndarray:
        """Variance of stochastic constraints."""
        if self.stoch_constraints is None:
            return np.array([])
        var = np.var(self.stoch_constraints[: self.n_reps], axis=0, ddof=1)
        return np.round(var, 15)

    @property
    def stoch_constraints_stderr(self) -> np.ndarray:
        """Standard error of stochastic constraints."""
        if self.stoch_constraints is None:
            return np.array([])
        std = np.std(self.stoch_constraints[: self.n_reps], axis=0, ddof=1)
        sqrt = np.sqrt(self.n_reps)
        return np.round(std / sqrt, 15)

    @property
    def stoch_constraints_cov(self) -> np.ndarray:
        """Covariance of stochastic constraints."""
        if self.stoch_constraints is None:
            return np.array([])
        cov = np.cov(self.stoch_constraints[: self.n_reps], rowvar=False, ddof=1)
        return np.round(cov, 15)

    # TODO: implement these properties
    # self.stoch_constraints_gradients_mean = np.mean(
    #     self.stoch_constraints_gradients[: self.n_reps], axis=0
    # )
    # self.stoch_constraints_gradients_var = np.var(
    #     self.stoch_constraints_gradients[: self.n_reps], axis=0, ddof=1
    # )
    # self.stoch_constraints_gradients_stderr = np.std(
    #     self.stoch_constraints_gradients[: self.n_reps], axis=0, ddof=1
    # ) / np.sqrt(self.n_reps)
    # self.stoch_constraints_gradients_cov = np.array(
    #     [
    #         np.cov(
    #             self.stoch_constraints_gradients[: self.n_reps, stcon],
    #             rowvar=False,
    #             ddof=1,
    #         )
    #         for stcon in range(len(self.det_stoch_constraints))
    #     ]
    # )

    def __init__(self, x: tuple, problem: Problem) -> None:
        """Initialize a solution object.

        Args:
            x (tuple): Vector of decision variables.
            problem (Problem): Problem to which `x` is a solution.
        """
        super().__init__()
        self.x = x
        self.decision_factors = problem.vector_to_factor_dict(x)
        self.n_reps = 0
        self.det_objectives, self.det_objectives_gradients = (
            problem.deterministic_objectives_and_gradients(self.x)
        )
        self.det_stoch_constraints, self.det_stoch_constraints_gradients = (
            problem.deterministic_stochastic_constraints_and_gradients()
        )
        # Initialize numpy arrays to store up to 100 replications.
        init_size = 100
        self.storage_size = init_size
        # Raw data.
        self.objectives = np.zeros((init_size, problem.n_objectives))
        self.objectives_gradients = np.zeros(
            (init_size, problem.n_objectives, problem.dim)
        )
        if problem.n_stochastic_constraints > 0:
            self.stoch_constraints = np.zeros(
                (init_size, problem.n_stochastic_constraints)
            )
            self.stoch_constraints_gradients = np.zeros(
                (init_size, problem.n_stochastic_constraints, problem.dim)
            )
        else:
            self.stoch_constraints = None
            self.stoch_constraints_gradients = None
        # Summary statistics
        # self.objectives_mean = np.full((problem.n_objectives), np.nan)
        # self.objectives_var = np.full((problem.n_objectives), np.nan)
        # self.objectives_stderr = np.full((problem.n_objectives), np.nan)
        # self.objectives_cov = np.full(
        #     (problem.n_objectives, problem.n_objectives), np.nan
        # )
        # self.objectives_gradients_mean = np.full(
        #     (problem.n_objectives, problem.dim), np.nan
        # )
        # self.objectives_gradients_var = np.full(
        #     (problem.n_objectives, problem.dim), np.nan
        # )
        # self.objectives_gradients_stderr = np.full(
        #     (problem.n_objectives, problem.dim), np.nan
        # )
        # self.objectives_gradients_cov = np.full(
        #     (problem.n_objectives, problem.dim, problem.dim), np.nan
        # )
        # self.stoch_constraints_mean = np.full(
        #     (problem.n_stochastic_constraints), np.nan
        # )
        # self.stoch_constraints_var = np.full(
        #     (problem.n_stochastic_constraints), np.nan
        # )
        # self.stoch_constraints_stderr = np.full(
        #     (problem.n_stochastic_constraints), np.nan
        # )
        # self.stoch_constraints_cov = np.full(
        #     (problem.n_stochastic_constraints, problem.n_stochastic_constraints),
        #     np.nan
        # )
        # self.stoch_constraints_gradients_mean = np.full(
        #     (problem.n_stochastic_constraints, problem.dim), np.nan
        # )
        # self.stoch_constraints_gradients_var = np.full(
        #     (problem.n_stochastic_constraints, problem.dim), np.nan
        # )
        # self.stoch_constraints_gradients_stderr = np.full(
        #     (problem.n_stochastic_constraints, problem.dim), np.nan
        # )
        # self.stoch_constraints_gradients_cov = np.full(
        #     (problem.n_stochastic_constraints, problem.dim, problem.dim), np.nan
        # )

    def attach_rngs(self, rng_list: list[MRG32k3a], copy: bool = True) -> None:
        """Attach a list of random-number generators to the solution.

        Args:
            rng_list (list[MRG32k3a]): List of RNGs used to run simulation replications.
            copy (bool, optional): If True (default), copies the RNGs before attaching
                them. If False, attaches the original RNG objects directly.
        """
        if copy:
            self.rng_list = [deepcopy(rng) for rng in rng_list]
        else:
            self.rng_list = rng_list

    def pad_storage(self, num_macroreps: int) -> None:
        """Append zeros to numpy arrays for summary statistics.

        Args:
            num_macroreps (int): Number of replications to simulate.
        """
        # Size of data storage.
        n_objectives = len(self.det_objectives)
        base_pad_size = 100
        # Default is to append space for 100 more replications.
        # If more space needed, append in multiples of 100.
        pad_size = int(np.ceil(num_macroreps / base_pad_size)) * base_pad_size
        self.storage_size += pad_size
        self.objectives = np.concatenate(
            (self.objectives, np.zeros((pad_size, n_objectives)))
        )
        self.objectives_gradients = np.concatenate(
            (
                self.objectives_gradients,
                np.zeros((pad_size, n_objectives, self.dim)),
            )
        )
        if self.stoch_constraints is not None:
            n_stochastic_constraints = len(self.det_stoch_constraints)
            self.stoch_constraints = np.concatenate(
                (
                    self.stoch_constraints,
                    np.zeros((pad_size, n_stochastic_constraints)),
                )
            )
            if self.stoch_constraints_gradients is not None:
                self.stoch_constraints_gradients = np.concatenate(
                    (
                        self.stoch_constraints_gradients,
                        np.zeros((pad_size, n_stochastic_constraints, self.dim)),
                    )
                )
            else:
                self.stoch_constraints_gradients = np.zeros(
                    (pad_size, n_stochastic_constraints, self.dim)
                )
