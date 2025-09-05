"""Base classes for simulation optimization problems and models."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, ClassVar

import numpy as np
from boltons.typeutils import classproperty
from pydantic import BaseModel

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.model import Model
from simopt.problem_types import ConstraintType, VariableType


class Problem(ABC):
    """Base class for simulation-optimization problems.

    Args:
        name (str): Problem name.
        fixed_factors (dict): User-defined factors that affect the problem setup.
        model_fixed_factors (dict): Subset of non-decision factors passed to the model.
        model (Callable[..., Model]): Simulation model that generates replications.
    """

    config_class: ClassVar[type[BaseModel]]
    model_class: ClassVar[type[Model]]

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
        raise RuntimeError("deprecated")

    @model_fixed_factors.setter
    def model_fixed_factors(self, value: dict | None) -> None:
        raise RuntimeError("deprecated")

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
        return self.config.model_dump(by_alias=True)

    @factors.setter
    def factors(self, value: dict | None) -> None:
        raise RuntimeError("factors are read-only")

    @classproperty
    @abstractmethod
    def specifications(cls) -> dict:
        """Details of each factor (for GUI, data validation, and defaults)."""
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
        # Assign the name of the problem
        self.name = name or self.class_name_abbr

        # Add all the fixed factors to the problem
        fixed_factors = fixed_factors or {}
        self.config = self.config_class(**fixed_factors)

        model_factors = {}
        model_config = self.model_class.config_class()
        model_factors.update(model_config.model_dump(by_alias=True))
        model_factors.update(self.model_default_factors)
        model_fixed_factors = model_fixed_factors or {}
        model_factors.update(model_fixed_factors)

        # Set the model
        self.model = self.model_class(model_factors)

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

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: B027
        """Hook executed before each simulation replication.

        Subclasses can override this to perform per-replication setup such as
        using the same RNG for different input models.

        Args:
            rng_list (list[MRG32k3a]): RNGs used for this replication.
        """
        pass

    def simulate(self, solution: "Solution", num_macroreps: int = 1) -> None:
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
            self.model.before_replicate(solution.rng_list)
            self.before_replicate(self.model, solution.rng_list)

            responses, gradients = self.model.replicate()
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

    def simulate_up_to(self, solutions: list["Solution"], n_reps: int) -> None:
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
