#!/usr/bin/env python
"""Provide base classes for solvers, problems, and models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Callable

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a


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
        else:
            # Raise an error since there's an error in the check list
            error_msg = f"Missing check for factor {factor_name} of type {self.specifications[factor_name]['datatype']}"
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


class ConstraintType(Enum):
    """Enum class for constraint types."""

    UNCONSTRAINED = 1
    BOX = 2
    DETERMINISTIC = 3
    STOCHASTIC = 4


class VariableType(Enum):
    """Enum class for variable types."""

    DISCRETE = 1
    CONTINUOUS = 2
    MIXED = 3


class Solver(ABC):
    """Base class to implement simulation-optimization solvers.

    Attributes
    ----------
    name : str
        Name of solver.
    objective_type : str
        Description of objective types: "single" or "multi".
    constraint_type : str
        Description of constraints types: "unconstrained", "box", "deterministic", "stochastic".
    variable_type : str
        Description of variable types: "discrete", "continuous", "mixed".
    gradient_needed : bool
        True if gradient of objective function is needed, otherwise False.
    factors : dict[str, int | float | bool]
        Changeable factors (i.e., parameters) of the solver.
    specifications : dict[str, dict[str, str | type | int | float | bool]]
        Details of each factor (for GUI, data validation, and defaults).
    rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
        List of RNGs used for the solver's internal purposes.
    solution_progenitor_rngs : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
        List of RNGs used as a baseline for simulating solutions.

    Parameters
    ----------
    fixed_factors : dict
        Dictionary of user-specified solver factors.

    """

    @property
    def name(self) -> str:
        """Name of solver."""
        return self.__name

    @name.setter
    def name(self, value: str) -> None:
        self.__name = value

    @property
    @abstractmethod
    def objective_type(self) -> ObjectiveType:
        """Description of objective types: "single" or "multi"."""
        raise NotImplementedError

    @property
    @abstractmethod
    def constraint_type(self) -> ConstraintType:
        """Description of constraints types: "unconstrained", "box", "deterministic", "stochastic"."""
        raise NotImplementedError

    @property
    @abstractmethod
    def variable_type(self) -> VariableType:
        """Description of variable types: "discrete", "continuous", "mixed"."""
        raise NotImplementedError

    @property
    @abstractmethod
    def gradient_needed(self) -> bool:
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

    @property
    @abstractmethod
    def specifications(
        self,
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

    def __init__(
        self, name: str = "", fixed_factors: dict | None = None
    ) -> None:
        """Initialize a solver object.

        Parameters
        ----------
        fixed_factors : dict
            Dictionary of user-specified solver factors.

        """
        assert len(name) > 0, "Name must be specified."
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

        Parameters
        ----------
        other : object
            Other object to compare to self.

        Returns
        -------
        bool
            True if the two objects are equivalent, otherwise False.

        """
        if not isinstance(other, Solver):
            return False
        return type(self) is type(other) and self.factors == other.factors

    def __hash__(self) -> int:
        """Return the hash value of the solver.

        Returns
        -------
        int
            Hash value of the solver.

        """
        return hash((self.name, tuple(self.factors.items())))

    def attach_rngs(self, rng_list: list[MRG32k3a]) -> None:
        """Attach a list of random-number generators to the solver.

        Parameters
        ----------
        rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
            List of random-number generators used for the solver's internal purposes.

        """
        self.rng_list = rng_list

    @abstractmethod
    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """Run a single macroreplication of a solver on a problem.

        Parameters
        ----------
        problem : ``base.Problem``
            Simulation-optimization problem to solve.

        Returns
        -------
        list [``Solution``]
            List of solutions recommended throughout the budget.
        list [int]
            List of intermediate budgets when recommended solutions changes.

        """
        raise NotImplementedError

    def check_crn_across_solns(self) -> bool:
        """Check solver factor crn_across_solns.

        Notes
        -----
        Currently implemented to always return True.

        Returns
        -------
        bool
            True if the solver factor is permissible, otherwise False.

        """
        return True

    def check_solver_factor(self, factor_name: str) -> bool:
        """Determine if the setting of a solver factor is permissible.

        Parameters
        ----------
        factor_name : str
            Name of factor for dictionary lookup (i.e., key).

        Returns
        -------
        bool
            True if the solver factor is permissible, otherwise False.

        """
        return _factor_check(self, factor_name)

    # TODO: Figure out if this should be abstract or not
    # @abstractmethod
    def check_solver_factors(self) -> bool:
        """Determine if the joint settings of solver factors are permissible.

        Returns
        -------
        is_simulatable : bool
            True if the solver factors are permissible, otherwise False.

        """
        return True
        raise NotImplementedError

    def check_factor_datatype(self, factor_name: str) -> bool:
        """Determine if a factor's data type matches its specification.

        Parameters
        ----------
        factor_name : str
            String corresponding to name of factor to check.

        Returns
        -------
        bool
            True if factor is of specified data type, otherwise False.

        """
        expected_data_type = self.specifications[factor_name]["datatype"]
        return isinstance(self.factors[factor_name], expected_data_type)

    def run_all_checks(self, factor_names: list[str]) -> bool:
        """Run all checks for the solver factors.

        Parameters
        ----------
        factor_names : list
            list of str names of factors to check.

        Returns
        -------
        bool
            defines if all checks came back as true.

        """
        is_joint_factors = (
            self.check_solver_factors()
        )  # check all joint factor settings

        if not is_joint_factors:
            error_msg = "There is a joint setting of a solver factor that is not permissible"
            raise ValueError(error_msg)

        # check datatypes for all factors
        for factor in factor_names:
            is_right_type = self.check_factor_datatype(factor)
            if not is_right_type:
                error_msg = (
                    f"Solver factor {factor} is not the correct data type."
                )
                raise ValueError(error_msg)

            is_permissible = self.check_solver_factor(factor)
            if not is_permissible:
                error_msg = f"Solver factor {factor} is not permissible."
                raise ValueError(error_msg)

        # Return true if no issues
        return True

    def create_new_solution(self, x: tuple, problem: Problem) -> Solution:
        """Create a new solution object with attached RNGs primed to simulate replications.

        Parameters
        ----------
        x : tuple
            Vector of decision variables.
        problem : ``base.Problem``
            Problem being solved by the solvers.

        Returns
        -------
        ``base.Solution``
            New solution.

        """
        # Create new solution with attached rngs.
        new_solution = Solution(x, problem)
        new_solution.attach_rngs(
            rng_list=self.solution_progenitor_rngs, copy=True
        )
        # Manipulate progenitor rngs to prepare for next new solution.
        if not self.factors["crn_across_solns"]:  # If CRN are not used ...
            # ...advance each rng to start of the substream = current substream + # of model RNGs.
            for rng in self.solution_progenitor_rngs:
                for _ in range(problem.model.n_rngs):
                    rng.advance_substream()
        return new_solution

    def rebase(self, n_reps: int) -> None:
        """Rebase the progenitor rngs to start at a later subsubstream index.

        Parameters
        ----------
        n_reps : int
            Substream index to skip to.

        """
        new_rngs = []
        for rng in self.solution_progenitor_rngs:
            stream_index = rng.s_ss_sss_index[0]
            substream_index = rng.s_ss_sss_index[1]
            new_rngs.append(
                MRG32k3a(s_ss_sss_index=[stream_index, substream_index, n_reps])
            )
        self.solution_progenitor_rngs = new_rngs

    def get_extended_name(self) -> str:
        """Get the extended name of the solver.

        Returns
        -------
        str
            Extended name of the solver.

        """
        # Single (S)
        # Multiple (M)
        objective = ""
        # Unconstrained (U)
        # Box (B)
        # Deterministic (D)
        # Stochastic (S)
        constraint = ""
        # Discrete (D)
        # Continuous (C)
        # Mixed (M)
        variable = ""
        # Available (G)
        # Not Available (N)
        direct_gradient_observations = ""
        # Set the string values
        if self.objective_type == "single":
            objective = "S"
        elif self.objective_type == "multi":
            objective = "M"
        if self.constraint_type == "unconstrained":
            constraint = "U"
        elif self.constraint_type == "box":
            constraint = "B"
        elif self.constraint_type == "deterministic":
            constraint = "D"
        elif self.constraint_type == "stochastic":
            constraint = "S"
        if self.variable_type == "discrete":
            variable = "D"
        elif self.variable_type == "continuous":
            variable = "C"
        elif self.variable_type == "mixed":
            variable = "M"
        if self.gradient_needed:
            direct_gradient_observations = "G"
        else:
            direct_gradient_observations = "N"

        return f"{self.name} ({objective}{constraint}{variable}{direct_gradient_observations})"


class Problem(ABC):
    """Base class to implement simulation-optimization problems.

    Attributes
    ----------
    name : str
        Name of problem.
    dim : int
        Number of decision variables.
    n_objectives : int
        Number of objectives.
    n_stochastic_constraints : int
        Number of stochastic constraints.
    minmax : tuple [int]
        Indicators of maximization (+1) or minimization (-1) for each objective.
    constraint_type : str
        Description of constraints types: "unconstrained", "box", "deterministic", "stochastic".
    variable_type : str
        Description of variable types: "discrete", "continuous", "mixed".
    lower_bounds : tuple
        Lower bound for each decision variable.
    upper_bounds : tuple
        Upper bound for each decision variable.
    gradient_available : bool
        True if direct gradient of objective function is available, otherwise False.
    optimal_value : float
        Optimal objective function value.
    optimal_solution : tuple
        Optimal solution.
    model : ``base.Model``
        Associated simulation model that generates replications.
    model_default_factors : dict
        Default values for overriding model-level default factors.
    model_fixed_factors : dict
        Combination of overriden model-level factors and defaults.
    model_decision_factors : set [str]
        Set of keys for factors that are decision variables.
    rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
        List of RNGs used to generate a random initial solution
        or a random problem instance.
    factors : dict
        Changeable factors of the problem:
            initial_solution : tuple
                Default initial solution from which solvers start.
            budget : int
                Max number of replications (fn evals) for a solver to take.
    specifications : dict
        Details of each factor (for GUI, data validation, and defaults).

    Parameters
    ----------
    fixed_factors : dict
        Dictionary of user-specified problem factors.
    model_fixed_factors : dict
        Subset of user-specified non-decision factors to pass through to the model.

    """

    @property
    def name(self) -> str:
        """Name of the problem."""
        return self.__name

    @name.setter
    def name(self, value: str) -> None:
        self.__name = value

    @property
    @abstractmethod
    def dim(self) -> int:
        """Number of decision variables."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_objectives(self) -> int:
        """Number of objectives."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_stochastic_constraints(self) -> int:
        """Number of stochastic constraints."""
        raise NotImplementedError

    @property
    @abstractmethod
    def minmax(self) -> tuple[int]:
        """Indicators of maximization (+1) or minimization (-1) for each objective."""
        raise NotImplementedError

    @property
    @abstractmethod
    def constraint_type(self) -> ConstraintType:
        """Description of constraints types: "unconstrained", "box", "deterministic", "stochastic"."""
        raise NotImplementedError

    @property
    @abstractmethod
    def variable_type(self) -> VariableType:
        """Description of variable types: "discrete", "continuous", "mixed"."""
        raise NotImplementedError

    @property
    @abstractmethod
    def lower_bounds(self) -> tuple:
        """Lower bound for each decision variable."""
        raise NotImplementedError

    @property
    @abstractmethod
    def upper_bounds(self) -> tuple:
        """Upper bound for each decision variable."""
        raise NotImplementedError

    @property
    @abstractmethod
    def gradient_available(self) -> bool:
        """True if direct gradient of objective function is available, otherwise False."""
        raise NotImplementedError

    @property
    @abstractmethod
    def optimal_value(self) -> float | None:
        """Optimal objective function value."""
        raise NotImplementedError

    @property
    @abstractmethod
    def optimal_solution(self) -> tuple | None:
        """Optimal solution."""
        raise NotImplementedError

    @property
    def model(self) -> Model:
        """Associated simulation model that generates replications."""
        return self.__model

    @model.setter
    def model(self, value: Model) -> None:
        self.__model = value

    @property
    @abstractmethod
    def model_default_factors(self) -> dict:
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

    @property
    @abstractmethod
    def model_decision_factors(self) -> set[str]:
        """Set of keys for factors that are decision variables."""
        raise NotImplementedError

    @property
    def rng_list(self) -> list[MRG32k3a]:
        """List of RNGs used to generate a random initial solution or a random problem instance."""
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

    @property
    @abstractmethod
    def specifications(self) -> dict:
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

        Parameters
        ----------
        fixed_factors : dict
            Dictionary of user-specified problem factors.
        model_fixed_factors : dict
            Subset of user-specified non-decision factors to pass through to the model.

        """
        assert len(name) > 0, "Name must be specified."
        assert model is not None, "Model must be specified."

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
            self.model_fixed_factors[factor] = self.model_default_factors[
                factor
            ]

        # Set the model
        self.model = model(self.model_fixed_factors)

        keys = list(self.factors.keys())
        self.run_all_checks(factor_names=keys)

    def __eq__(self, other: object) -> bool:
        """Check if two problems are equivalent.

        Parameters
        ----------
        other : object
            Other ``base.Problem`` objects to compare to self.

        Returns
        -------
        bool
            True if the two problems are equivalent, otherwise False.

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
        else:
            return False

    def __hash__(self) -> int:
        """Return the hash value of the solver.

        Returns
        -------
        int
            Hash value of the solver.

        """
        non_decision_factors = (
            set(self.model.factors.keys()) - self.model_decision_factors
        )
        return hash(
            (
                self.name,
                tuple(self.factors.items()),
                tuple(
                    [
                        (key, self.model.factors[key])
                        for key in non_decision_factors
                    ]
                ),
            )
        )

    def check_initial_solution(self) -> bool:
        """Check if initial solution is feasible and of correct dimension.

        Returns
        -------
        bool
            True if initial solution is feasible and of correct dimension, otherwise False.

        """
        # return len(
        #     self.factors["initial_solution"]
        # ) == self.dim and self.check_deterministic_constraints(
        #     decision_variables=self.factors["initial_solution"]
        # )
        return True

    def check_budget(self) -> bool:
        """Check if budget is strictly positive.

        Returns
        -------
        bool
            True if budget is strictly positive, otherwise False.

        """
        is_positive = self.factors["budget"] > 0
        return is_positive

    def check_problem_factor(self, factor_name: str) -> bool:
        """Determine if the setting of a problem factor is permissible.

        Parameters
        ----------
        factor_name : str
            Name of factor for dictionary lookup (i.e., key).

        Returns
        -------
        is_permissible : bool
            True if problem factor is permissible, otherwise False.

        """
        return _factor_check(self, factor_name)

    # NOTE: This was originally supposed to be an abstract method, but only
    # SPSA actually implements it. It's currently not clear if this
    # method should be implemented in other Problems as well.
    # @abstractmethod
    def check_problem_factors(self) -> bool:
        """Determine if the joint settings of problem factors are permissible.

        Returns
        -------
        is_simulatable : bool
            True if problem factors are permissible, otherwise False.

        """
        return True
        raise NotImplementedError

    def check_factor_datatype(self, factor_name: str) -> bool:
        """Determine if a factor's data type matches its specification.

        Parameters
        ----------
        factor_name : str
            String corresponding to name of factor to check.

        Returns
        -------
        is_right_type : bool
            True if factor is of specified data type, otherwise False.

        """
        return isinstance(
            self.factors[factor_name],
            self.specifications[factor_name]["datatype"],
        )

    def run_all_checks(self, factor_names: list[str]) -> bool:
        """Run all checks for the problem factors.

        Parameters
        ----------
        factor_names : list[str]
            list of str names of factors to check.

        Returns
        -------
        bool
            defines if all checks came back as true.

        """
        is_joint_factors = (
            self.check_problem_factors()
        )  # check all joint factor settings
        if not is_joint_factors:
            error_msg = "There is a joint setting of a problem factor that is not permissible"
            raise ValueError(error_msg)

        is_initial_sol = self.check_initial_solution()
        if not is_initial_sol:
            error_msg = "The initial solution is not feasible and/or not correct dimension"
            raise ValueError(error_msg)

        # TODO: investigate why this is not working
        # is_budget = self.check_budget()
        if (
            isinstance(self.factors["budget"], int)
            and self.factors["budget"] <= 0
        ):
            error_msg = "The budget is not positive."
            raise ValueError(error_msg)

        # check datatypes for all factors
        for factor in factor_names:
            is_permissible = self.check_problem_factor(factor)
            is_right_type = self.check_factor_datatype(factor)

            if not is_right_type:
                error_msg = (
                    f"Problem factor {factor} is not a permissible data type."
                )
                raise ValueError(error_msg)

            if not is_permissible:
                error_msg = f"Problem factor {factor} is not permissible."
                raise ValueError(error_msg)

        return True

    def attach_rngs(self, rng_list: list[MRG32k3a]) -> None:
        """Attach a list of random-number generators to the problem.

        Parameters
        ----------
        rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
            List of random-number generators used to generate a random initial solution
            or a random problem instance.

        """
        self.rng_list = rng_list

    @abstractmethod
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """Convert a vector of variables to a dictionary with factor keys.

        Parameters
        ----------
        vector : tuple
            Vector of values associated with decision variables.

        Returns
        -------
        dict
            Dictionary with factor keys and associated values.

        """
        raise NotImplementedError

    @abstractmethod
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """Convert a dictionary with factor keys to a vector of variables.

        Parameters
        ----------
        factor_dict : dict
            Dictionary with factor keys and associated values.

        Returns
        -------
        tuple
            Vector of values associated with decision variables.

        """
        raise NotImplementedError

    def factor_dict_to_vector_gradients(self, factor_dict: dict) -> tuple:
        """Convert a dictionary with factor keys to a gradient vector.

        Notes
        -----
        A subclass of ``base.Problem`` can have its own custom ``factor_dict_to_vector_gradients`` method if the objective is deterministic.

        Parameters
        ----------
        factor_dict : dict
            Dictionary with factor keys and associated values.

        Returns
        -------
        tuple
            Vector of partial derivatives associated with decision variables.

        """
        return self.factor_dict_to_vector(factor_dict)

    @abstractmethod
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        """Convert a dictionary with response keys to a vector of objectives.

        Parameters
        ----------
        response_dict : dict
            Dictionary with response keys and associated values.

        Returns
        -------
        tuple
            Vector of objectives.

        """
        raise NotImplementedError

    def response_dict_to_objectives_gradients(
        self, response_dict: dict
    ) -> tuple:
        """Convert a dictionary with response keys to a vector of gradients.

        Notes
        -----
        Can be overridden by subclasses if the objective is deterministic.

        Parameters
        ----------
        response_dict : dict
            Dictionary with response keys and associated values.

        Returns
        -------
        tuple
            Vector of gradients.

        """
        return self.response_dict_to_objectives(response_dict)

    @abstractmethod
    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """Convert a dictionary with response keys to a vector of left-hand sides of stochastic constraints: E[Y] <= 0.

        Parameters
        ----------
        response_dict : dict
            Dictionary with response keys and associated values.

        Returns
        -------
        tuple
            Vector of LHSs of stochastic constraints.

        """
        raise NotImplementedError

    def deterministic_objectives_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
        """Compute deterministic components of objectives for a solution `x`.

        Parameters
        ----------
        x : tuple
            Vector of decision variables.

        Returns
        -------
        det_objectives : tuple
            Vector of deterministic components of objectives.
        det_objectives_gradients : tuple
            Vector of gradients of deterministic components of objectives.

        """
        det_objectives = (0,) * self.n_objectives
        det_objectives_gradients = tuple(
            [(0,) * self.dim for _ in range(self.n_objectives)]
        )
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
        """Compute deterministic components of stochastic constraints for a solution `x`.

        Parameters
        ----------
        x : tuple
            Vector of decision variables.

        Returns
        -------
        det_stoch_constraints : tuple
            Vector of deterministic components of stochastic constraints.
        det_stoch_constraints_gradients : tuple
            Vector of gradients of deterministic components of stochastic constraints.

        """
        det_stoch_constraints = (0,) * self.n_stochastic_constraints
        det_stoch_constraints_gradients = tuple(
            [(0,) * self.dim for _ in range(self.n_stochastic_constraints)]
        )
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Check if a solution `x` satisfies the problem's deterministic constraints.

        Parameters
        ----------
        x : tuple
            Vector of decision variables.

        Returns
        -------
        bool
            True if solution `x` satisfies the deterministic constraints, otherwise False.

        """
        # Check box constraints.
        return bool(
            np.prod(
                [
                    self.lower_bounds[idx] <= x[idx] <= self.upper_bounds[idx]
                    for idx in range(len(x))
                ]
            )
        )

    @abstractmethod
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """Generate a random solution for starting or restarting solvers.

        Parameters
        ----------
        rand_sol_rng : ``mrg32k3a.mrg32k3a.MRG32k3a``
            Random-number generator used to sample a new random solution.

        Returns
        -------
        tuple
            vector of decision variables

        """
        raise NotImplementedError

    def simulate(self, solution: Solution, num_macroreps: int = 1) -> None:
        """Simulate `m` i.i.d. replications at solution `x`.

        Notes
        -----
        Gradients of objective function and stochastic constraint LHSs are temporarily commented out. Under development.

        Parameters
        ----------
        solution : ``base.Solution``
            Solution to evalaute.
        num_macroreps : int, default=1
            Number of replications to simulate at `x`.

        """
        # Type checking
        if not isinstance(solution, Solution):
            error_msg = "Input solution must be of type Solution."
            raise TypeError(error_msg)
        if not isinstance(num_macroreps, int):
            error_msg = "Number of replications must be an integer."
            raise TypeError(error_msg)
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
                # vector_gradients = {keys: self.factor_dict_to_vector(gradient_dict) for (keys, gradient_dict) in gradients.items()}
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
                # print(self.response_dict_to_objectives_gradients(vector_gradients))
                # print(solution.det_objectives_gradients)
                # TODO: Ensure that this never happens
                if "vector_gradients" not in locals():
                    raise ValueError("vector_gradients not defined")
                else:
                    solution.objectives_gradients[solution.n_reps] = [
                        [sum(pairs) for pairs in zip(stoch_obj, det_obj)]
                        for stoch_obj, det_obj in zip(
                            self.response_dict_to_objectives_gradients(
                                vector_gradients
                            ),
                            solution.det_objectives_gradients,
                        )
                    ]
                    # solution.objectives_gradients[solution.n_reps] = [[sum(pairs) for pairs in zip(stoch_obj, det_obj)] for stoch_obj, det_obj in zip(self.response_dict_to_objectives(vector_gradients), solution.det_objectives_gradients)]
            if (
                self.n_stochastic_constraints > 0
                and solution.stoch_constraints is not None
            ):
                # Convert responses and gradients to stochastic constraints and gradients and add
                # to those of deterministic components of stochastic constraints.
                solution.stoch_constraints[solution.n_reps] = [
                    sum(pairs)
                    for pairs in zip(
                        self.response_dict_to_stoch_constraints(responses),
                        solution.det_stoch_constraints,
                    )
                ]
                # solution.stoch_constraints_gradients[solution.n_reps] = [[sum(pairs) for pairs in zip(stoch_stoch_cons, det_stoch_cons)] for stoch_stoch_cons, det_stoch_cons in zip(self.response_dict_to_stoch_constraints(vector_gradients), solution.det_stoch_constraints_gradients)]
            # Increment counter.
            solution.n_reps += 1
            # Advance rngs to start of next subsubstream.
            for rng in solution.rng_list:
                rng.advance_subsubstream()
        # Update summary statistics.
        solution.recompute_summary_statistics()

    def simulate_up_to(self, solutions: list[Solution], n_reps: int) -> None:
        """Simulate a list of solutions up to a given number of replications.

        Parameters
        ----------
        solutions : list [``base.Solution``]
            A list of ``base.Solution`` objects.
        n_reps : int
            Common number of replications to simulate each solution up to.

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
    """Base class to implement simulation models (models) featured in simulation-optimization problems.

    Attributes
    ----------
    name : str
        Name of model.
    n_rngs : int
        Number of random-number generators used to run a simulation replication.
    n_responses : int
        Number of responses (performance measures).
    factors : dict
        Changeable factors of the simulation model.
    specifications : dict
        Details of each factor (for GUI, data validation, and defaults).
    check_factor_list : dict
        Switch case for checking factor simulatability.

    Parameters
    ----------
    fixed_factors : dict
        Dictionary of user-specified model factors.

    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_rngs(self) -> int:
        """Number of random-number generators used to run a simulation replication."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_responses(self) -> int:
        """Number of responses (performance measures)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def specifications(self) -> dict[str, dict]:
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

        Parameters
        ----------
        fixed_factors : dict
            Dictionary of user-specified model factors.

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

        Parameters
        ----------
        other : object
            Other object to compare to self.

        Returns
        -------
        bool
            True if the two models are equivalent, otherwise False.

        """
        if not isinstance(other, Model):
            return False
        return type(self) is type(other) and self.factors == other.factors

    def __hash__(self) -> int:
        """Return the hash value of the model.

        Returns
        -------
        int
            Hash value of the model.

        """
        return hash((self.name, tuple(self.factors.items())))

    def check_simulatable_factor(self, factor_name: str) -> bool:
        """Determine if a simulation replication can be run with the given factor.

        Parameters
        ----------
        factor_name : str
            Name of factor for dictionary lookup (i.e., key).

        Returns
        -------
        bool
            True if model specified by factors is simulatable, otherwise False.

        """
        return _factor_check(self, factor_name)

    def check_simulatable_factors(self) -> bool:
        """Determine if a simulation replication can be run with the given factors.

        Notes
        -----
        Each subclass of ``base.Model`` has its own custom ``check_simulatable_factors`` method.
        If the model does not override this method, it will return True.

        Returns
        -------
        bool
            True if model specified by factors is simulatable, otherwise False.

        """
        return True

    def check_factor_datatype(self, factor_name: str) -> bool:
        """Determine if a factor's data type matches its specification.

        Parameters
        ----------
        factor_name : str
            String corresponding to name of factor to check.

        Returns
        -------
        bool
            True if factor is of specified data type, otherwise False.

        """
        datatype = self.specifications[factor_name]["datatype"]
        if datatype is float:
            datatype = (int, float)
        is_right_type = isinstance(self.factors[factor_name], datatype)
        return is_right_type

    def run_all_checks(self, factor_names: list[str]) -> bool:
        """Run all checks for the model factors.

        Parameters
        ----------
        factor_names : list
            list of str names of factors to check.

        Returns
        -------
        check_all : bool
            defines if all checks came back as true.

        """
        is_joint_factors = (
            self.check_simulatable_factors()
        )  # check all joint factor settings

        if not is_joint_factors:
            error_msg = "There is a joint setting of a model factor that is not permissible"
            raise ValueError(error_msg)

        # check datatypes for all factors
        for factor in factor_names:
            is_right_type = self.check_factor_datatype(factor)
            if not is_right_type:
                error_msg = (
                    f"Model factor {factor} is not a permissible data type."
                )
                raise ValueError(error_msg)

            is_permissible = self.check_simulatable_factor(factor)
            if not is_permissible:
                error_msg = f"Model factor {factor} is not permissible."
                raise ValueError(error_msg)

        return True

    @abstractmethod
    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Parameters
        ----------
        rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
            RNGs for model to use when simulating a replication.

        Returns
        -------
        responses : dict
            Performance measures of interest.
        gradients : dict [dict]
            Gradient estimate for each response.

        """
        raise NotImplementedError


class Solution:
    """Base class for solutions represented as vectors of decision variables and dictionaries of decision factors.

    Attributes
    ----------
    x : tuple
        Vector of decision variables.
    dim : int
        Number of decision variables describing `x`.
    decision_factors : dict
        Decision factor names and values.
    rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
        RNGs for model to use when running replications at the solution.
    n_reps : int
        Number of replications run at the solution.
    det_objectives : tuple
        Deterministic components added to objectives.
    det_objectives_gradients : tuple [tuple]
        Gradients of deterministic components added to objectives;
        # objectives x dimension.
    det_stoch_constraints : tuple
        Deterministic components added to LHS of stochastic constraints.
    det_stoch_constraints_gradients : tuple [tuple]
        Gradients of deterministics components added to LHS stochastic constraints;
        # stochastic constraints x dimension.
    storage_size : int
        Max number of replications that can be recorded in current storage.
    objectives : numpy array
        Objective(s) estimates from each replication;
        # replications x # objectives.
    objectives_gradients : numpy array
        Gradient estimates of objective(s) from each replication;
        # replications x # objectives x dimension.
    stochastic_constraints : numpy array
        Stochastic constraint estimates from each replication;
        # replications x # stochastic constraints.
    stochastic_constraints_gradients : numpy array
        Gradient estimates of stochastic constraints from each replication;
        # replications x # stochastic constraints x dimension.

    Parameters
    ----------
    x : tuple
        Vector of decision variables.
    problem : ``base.Problem``
        Problem to which `x` is a solution.

    """

    @property
    def x(self) -> tuple:
        """Vector of decision variables."""
        return self.__x

    @x.setter
    def x(self, value: tuple) -> None:
        self.__x = value

    @property
    def dim(self) -> int:
        """Number of decision variables describing `x`."""
        return len(self.__x)

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
        """Gradients of deterministic components added to objectives; # objectives x dimension."""
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
        """Gradients of deterministic components added to LHS stochastic constraints; # stochastic constraints x dimension."""
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
        """Objective(s) estimates from each replication; # replications x # objectives."""
        return self.__objectives

    @objectives.setter
    def objectives(self, value: np.ndarray) -> None:
        self.__objectives = value

    @property
    def objectives_gradients(self) -> np.ndarray:
        """Gradient estimates of objective(s) from each replication; # replications x # objectives x dimension."""
        return self.__objectives_gradients

    @objectives_gradients.setter
    def objectives_gradients(self, value: np.ndarray) -> None:
        self.__objectives_gradients = value

    @property
    def stochastic_constraints(self) -> np.ndarray:
        """Stochastic constraint estimates from each replication; # replications x # stochastic constraints."""
        return self.__stochastic_constraints

    @stochastic_constraints.setter
    def stochastic_constraints(self, value: np.ndarray) -> None:
        self.__stochastic_constraints = value

    @property
    def stochastic_constraints_gradients(self) -> np.ndarray:
        """Gradient estimates of stochastic constraints from each replication; # replications x # stochastic constraints x dimension."""
        return self.__stochastic_constraints_gradients

    @stochastic_constraints_gradients.setter
    def stochastic_constraints_gradients(self, value: np.ndarray) -> None:
        self.__stochastic_constraints_gradients = value

    def __init__(self, x: tuple, problem: Problem) -> None:
        """Initialize a solution object.

        Parameters
        ----------
        x : tuple
            Vector of decision variables.
        problem : ``base.Problem``
            Problem to which `x` is a solution.

        """
        super().__init__()
        self.x = x
        self.decision_factors = problem.vector_to_factor_dict(x)
        self.n_reps = 0
        self.det_objectives, self.det_objectives_gradients = (
            problem.deterministic_objectives_and_gradients(self.x)
        )
        self.det_stoch_constraints, self.det_stoch_constraints_gradients = (
            problem.deterministic_stochastic_constraints_and_gradients(self.x)
        )
        init_size = (
            100  # Initialize numpy arrays to store up to 100 replications.
        )
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
        # self.objectives_cov = np.full((problem.n_objectives, problem.n_objectives), np.nan)
        # self.objectives_gradients_mean = np.full((problem.n_objectives, problem.dim), np.nan)
        # self.objectives_gradients_var = np.full((problem.n_objectives, problem.dim), np.nan)
        # self.objectives_gradients_stderr = np.full((problem.n_objectives, problem.dim), np.nan)
        # self.objectives_gradients_cov = np.full((problem.n_objectives, problem.dim, problem.dim), np.nan)
        # self.stoch_constraints_mean = np.full((problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_var = np.full((problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_stderr = np.full((problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_cov = np.full((problem.n_stochastic_constraints, problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_gradients_mean = np.full((problem.n_stochastic_constraints, problem.dim), np.nan)
        # self.stoch_constraints_gradients_var = np.full((problem.n_stochastic_constraints, problem.dim), np.nan)
        # self.stoch_constraints_gradients_stderr = np.full((problem.n_stochastic_constraints, problem.dim), np.nan)
        # self.stoch_constraints_gradients_cov = np.full((problem.n_stochastic_constraints, problem.dim, problem.dim), np.nan)

    def attach_rngs(self, rng_list: list[MRG32k3a], copy: bool = True) -> None:
        """Attach a list of random-number generators to the solution.

        Parameters
        ----------
        rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
            List of random-number generators used to run simulation replications.
        copy : bool, default=True
            True if we want to copy the ``mrg32k3a.mrg32k3a.MRG32k3a`` objects, otherwise False.

        """
        if copy:
            self.rng_list = [deepcopy(rng) for rng in rng_list]
        else:
            self.rng_list = rng_list

    def pad_storage(self, num_macroreps: int) -> None:
        """Append zeros to numpy arrays for summary statistics.

        Parameters
        ----------
        num_macroreps : int
            Number of replications to simulate.

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
                        np.zeros(
                            (pad_size, n_stochastic_constraints, self.dim)
                        ),
                    )
                )
            else:
                self.stoch_constraints_gradients = np.zeros(
                    (pad_size, n_stochastic_constraints, self.dim)
                )

    def recompute_summary_statistics(self) -> None:
        """Recompute summary statistics of the solution.

        Notes
        -----
        Statistics for gradients of objectives and stochastic constraint LHSs are temporarily commented out. Under development.

        """
        self.objectives_mean = np.mean(self.objectives[: self.n_reps], axis=0)
        if self.n_reps > 1:
            self.objectives_var = np.var(
                self.objectives[: self.n_reps], axis=0, ddof=1
            )
            self.objectives_stderr = np.std(
                self.objectives[: self.n_reps], axis=0, ddof=1
            ) / np.sqrt(self.n_reps)
            self.objectives_cov = np.cov(
                self.objectives[: self.n_reps], rowvar=False, ddof=1
            )
        self.objectives_gradients_mean = np.mean(
            self.objectives_gradients[: self.n_reps], axis=0
        )
        if self.n_reps > 1:
            self.objectives_gradients_var = np.var(
                self.objectives_gradients[: self.n_reps], axis=0, ddof=1
            )
            self.objectives_gradients_stderr = np.std(
                self.objectives_gradients[: self.n_reps], axis=0, ddof=1
            ) / np.sqrt(self.n_reps)
            self.objectives_gradients_cov = np.array(
                [
                    np.cov(
                        self.objectives_gradients[: self.n_reps, obj],
                        rowvar=False,
                        ddof=1,
                    )
                    for obj in range(len(self.det_objectives))
                ]
            )
        if self.stoch_constraints is not None:
            self.stoch_constraints_mean = np.mean(
                self.stoch_constraints[: self.n_reps], axis=0
            )
            self.stoch_constraints_var = np.var(
                self.stoch_constraints[: self.n_reps], axis=0, ddof=1
            )
            self.stoch_constraints_stderr = np.std(
                self.stoch_constraints[: self.n_reps], axis=0, ddof=1
            ) / np.sqrt(self.n_reps)
            self.stoch_constraints_cov = np.cov(
                self.stoch_constraints[: self.n_reps], rowvar=False, ddof=1
            )
            # self.stoch_constraints_gradients_mean = np.mean(self.stoch_constraints_gradients[:self.n_reps], axis=0)
            # self.stoch_constraints_gradients_var = np.var(self.stoch_constraints_gradients[:self.n_reps], axis=0, ddof=1)
            # self.stoch_constraints_gradients_stderr = np.std(self.stoch_constraints_gradients[:self.n_reps], axis=0, ddof=1) / np.sqrt(self.n_reps)
            # self.stoch_constraints_gradients_cov = np.array([np.cov(self.stoch_constraints_gradients[:self.n_reps, stcon], rowvar=False, ddof=1) for stcon in range(len(self.det_stoch_constraints))])
