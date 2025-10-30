"""Base classes for simulation optimization problems and models."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from typing import ClassVar

import numpy as np
from boltons.typeutils import classproperty
from pydantic import BaseModel

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.model import Model
from simopt.problem_types import ConstraintType, VariableType
from simopt.utils import get_specifications


def _to_grad_array(gradients: float | Iterable[float]) -> np.ndarray:
    rv = np.array(gradients)
    if rv.ndim == 0:
        rv = rv.reshape((1,))
    rv[np.isnan(rv)] = 0.0
    return rv


class Objective:
    """Represents an objective function value with optional gradients.

    Combines stochastic and deterministic components of an objective function,
    along with their respective gradients if available.
    """

    def __init__(
        self,
        stochastic: float,
        stochastic_gradients: float | Iterable[float] | None = None,
        deterministic: float = 0.0,
        deterministic_gradients: float | Iterable[float] | None = None,
    ) -> None:
        """Initialize an Objective with stochastic and deterministic components.

        Args:
            stochastic: The stochastic component of the objective value.
            stochastic_gradients: Gradients of the stochastic component.
            deterministic: The deterministic component of the objective value.
            deterministic_gradients: Gradients of the deterministic component.
        """
        self.stochastic = stochastic
        self.stochastic_gradients = stochastic_gradients
        self.deterministic = deterministic
        self.deterministic_gradients = deterministic_gradients

    def value(self) -> float:
        """Return the total objective value (stochastic + deterministic)."""
        return self.stochastic + self.deterministic

    def grad(self) -> np.ndarray | None:
        """Return the combined gradients of both components, or None if unavailable."""
        match self.stochastic_gradients, self.deterministic_gradients:
            case None, None:
                return None
            case None, _:
                return _to_grad_array(self.deterministic_gradients)  # pyrefly: ignore
            case _, None:
                return _to_grad_array(self.stochastic_gradients)  # pyrefly: ignore
            case _, _:
                stochastic_gradients = _to_grad_array(
                    self.stochastic_gradients  # pyrefly: ignore
                )
                deterministic_gradients = _to_grad_array(
                    self.deterministic_gradients  # pyrefly: ignore
                )
                return stochastic_gradients + deterministic_gradients


class StochasticConstraint:
    """Represents a stochastic constraint with optional deterministic component."""

    def __init__(
        self,
        stochastic: float,
        stochastic_gradients: float | Iterable[float] | None = None,
        deterministic: float = 0.0,
        deterministic_gradients: float | Iterable[float] | None = None,
    ) -> None:
        """Initialize a StochasticConstraint.

        Args:
            stochastic: The stochastic component of the constraint.
            stochastic_gradients: Gradients of the stochastic component.
            deterministic: The deterministic component of the constraint.
            deterministic_gradients: Gradients of the deterministic component.
        """
        self.stochastic = stochastic
        self.stochastic_gradients = stochastic_gradients
        self.deterministic = deterministic
        self.deterministic_gradients = deterministic_gradients

    def value(self) -> float:
        """Return the total constraint value (stochastic + deterministic if present)."""
        return self.stochastic + self.deterministic

    def grad(self) -> np.ndarray | None:
        """Return the combined gradients of both components, or None if unavailable."""
        match self.stochastic_gradients, self.deterministic_gradients:
            case None, None:
                return None
            case None, _:
                return _to_grad_array(self.deterministic_gradients)  # pyrefly: ignore
            case _, None:
                return _to_grad_array(self.stochastic_gradients)  # pyrefly: ignore
            case _, _:
                stochastic_gradients = _to_grad_array(
                    self.stochastic_gradients  # pyrefly: ignore
                )
                deterministic_gradients = _to_grad_array(
                    self.deterministic_gradients  # pyrefly: ignore
                )
                return stochastic_gradients + deterministic_gradients


class RepResult:
    """Container for results from a single simulation replication."""

    def __init__(
        self,
        objectives: list[Objective],
        stochastic_constraints: list[StochasticConstraint] | None = None,
    ) -> None:
        """Initialize a RepResult with objectives and optional constraints.

        Args:
            objectives: List of objective function results from the replication.
            stochastic_constraints: Optional list of stochastic constraint results.
        """
        self.objectives = objectives
        self.stochastic_constraints = stochastic_constraints


class Problem(ABC):
    """Base class for simulation-optimization problems.

    Args:
        name (str): Problem name.
        fixed_factors (dict): User-defined factors that affect the problem setup.
        model_fixed_factors (dict): Subset of non-decision factors passed to the model.
    """

    class_name_abbr: ClassVar[str]
    """Short name of the problem class."""

    class_name: ClassVar[str]
    """Long name of the problem class."""

    config_class: ClassVar[type[BaseModel]]
    """Configuration class for problem."""

    model_class: ClassVar[type[Model]]
    """Simulation model class for problem."""

    constraint_type: ClassVar[ConstraintType]
    """Description of constraints types."""

    variable_type: ClassVar[VariableType]
    """Description of variable types."""

    gradient_available: ClassVar[bool]
    """Indicates whether the solver provides direct gradient information."""

    n_objectives: ClassVar[int]
    """Number of objectives."""

    minmax: ClassVar[tuple[int, ...]]
    """Indicators of maximization (+1) or minimization (-1) for each objective."""

    n_stochastic_constraints: ClassVar[int]
    """Number of stochastic constraints."""

    model_default_factors: ClassVar[dict]
    """Default values for overriding model-level default factors."""

    model_decision_factors: ClassVar[set[str]]
    """Set of keys for factors that are decision variables."""

    def __init__(
        self,
        name: str = "",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize a problem object.

        Args:
            name (str): Name of the problem.
            fixed_factors (dict | None): Dictionary of user-specified problem factors.
            model_fixed_factors (dict | None): Subset of user-specified non-decision
                factors passed to the model.
        """
        # Assign the name of the problem
        self.name = name or self.class_name_abbr

        # Add all the fixed factors to the problem
        fixed_factors = fixed_factors or {}
        self.config = self.config_class(**fixed_factors)
        self._factors = self.config.model_dump(by_alias=True)

        model_factors = {}
        model_config = self.model_class.config_class()
        model_factors.update(model_config.model_dump(by_alias=True))
        model_factors.update(self.model_default_factors)
        model_fixed_factors = model_fixed_factors or {}
        model_factors.update(model_fixed_factors)

        # Set the model
        self.model = self.model_class(model_factors)

        self.rng_list: list[MRG32k3a] = []
        self.before_replicate_override = None

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

    @property
    def optimal_value(self) -> float | None:
        """Optimal objective function value (if known)."""
        return None

    @property
    def optimal_solution(self) -> tuple | None:
        """Optimal solution if known; defaults to None."""
        return None

    @property
    @abstractmethod
    def dim(self) -> int:
        """Number of decision variables."""
        raise NotImplementedError

    @property
    @abstractmethod
    def lower_bounds(self) -> tuple[float, ...]:
        """Lower bound for each decision variable."""
        raise NotImplementedError

    @property
    @abstractmethod
    def upper_bounds(self) -> tuple[float, ...]:
        """Upper bound for each decision variable."""
        raise NotImplementedError

    @classproperty
    def compatibility(cls) -> str:  # noqa: N805
        """Compatibility of the solver."""
        return (
            "S"
            f"{cls.constraint_type.symbol()}"
            f"{cls.variable_type.symbol()}"
            f"{'G' if cls.gradient_available else 'N'}"
        )

    @classproperty
    def specifications(cls) -> dict[str, dict]:  # noqa: N805
        """Details of each factor (for GUI, data validation, and defaults)."""
        return get_specifications(cls.config_class)

    @property
    def factors(self) -> dict:
        """Changeable factors of the problem."""
        return self._factors

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

    def check_deterministic_constraints(self, x: tuple, /) -> bool:
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
            for x_i, lb, ub in zip(
                x, self.lower_bounds, self.upper_bounds, strict=False
            )
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

    @abstractmethod
    def replicate(self, x: tuple, /) -> RepResult:
        """Replicate the problem for a given solution.

        Args:
            x (tuple): The solution to evaluate.
        """
        raise NotImplementedError

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

        # Set the decision factors of the model.
        self.model.factors.update(solution.decision_factors)
        for _ in range(num_macroreps):
            # Generate one replication at x.
            self.model.before_replicate(solution.rng_list)
            self.before_replicate(solution.rng_list)
            if self.before_replicate_override is not None:
                self.before_replicate_override(self.model, solution.rng_list)

            result = self.replicate(solution.x)
            solution.add_replicate_result(result)

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

    @property
    def stoch_constraints_gradients(self) -> np.ndarray:
        """Gradients of stochastic constraints."""
        if not self._stoch_constraints_gradients:
            return np.array([])
        return np.array(self._stoch_constraints_gradients)

    @property
    def stoch_constraints_gradients_mean(self) -> np.ndarray:
        """Mean of gradients of stochastic constraints."""
        if not self._stoch_constraints_gradients:
            return np.array([])
        mean = np.mean(self.stoch_constraints_gradients[: self.n_reps], axis=0)
        return np.round(mean, 15)

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
        # self.n_reps = 0

        self._objectives = []
        self._objectives_gradients = []
        self._stoch_constraints = []
        self._stoch_constraints_gradients = []
        self._objectives_array: np.ndarray | None = None
        self._objectives_gradients_array: np.ndarray | None = None
        self._stoch_constraints_array: np.ndarray | None = None

    @property
    def n_reps(self) -> int:
        """Number of replications."""
        return len(self._objectives)

    @property
    def objectives(self) -> np.ndarray:
        """Objectives."""
        if self._objectives_array is None:
            self._objectives_array = np.array(self._objectives)
        return self._objectives_array

    @property
    def objectives_gradients(self) -> np.ndarray:
        """Objectives gradients."""
        if self._objectives_gradients_array is None:
            self._objectives_gradients_array = np.array(self._objectives_gradients)
        return self._objectives_gradients_array

    @property
    def stoch_constraints(self) -> np.ndarray:
        """Stochastic constraints."""
        if self._stoch_constraints_array is None:
            self._stoch_constraints_array = np.array(self._stoch_constraints)
        return self._stoch_constraints_array

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

    def add_replicate_result(self, result: RepResult) -> None:
        """Add a replicate result to the solution.

        Args:
            result (RepResult): The replicate result to add.
        """
        objectives = result.objectives
        stochastic_constraints = result.stochastic_constraints

        # Convert responses and gradients to objectives and gradients and add
        # to those of deterministic components of objectives.
        self._objectives.append(
            np.array([objective.value() for objective in objectives])
        )
        self._objectives_array = None

        gradients = []
        for objective in objectives:
            grad = objective.grad()
            if grad is None:
                grad = np.zeros(self.dim)
            gradients.append(grad)
        self._objectives_gradients.append(np.array(gradients))
        self._objectives_gradients_array = None

        # Convert responses and gradients to stochastic constraints and
        # gradients and addto those of deterministic components of stochastic
        # constraints.
        if stochastic_constraints is not None:
            self._stoch_constraints.append(
                np.array([constraint.value() for constraint in stochastic_constraints])
            )
            self._stoch_constraints_array = None

            gradients = []
            for constraint in stochastic_constraints:
                grad = constraint.grad()
                if grad is None:
                    grad = np.zeros(self.dim)
                gradients.append(grad)
            gradients = np.array(gradients)
            self._stoch_constraints_gradients.append(gradients)
            self._stoch_constraints_gradients_array = None
