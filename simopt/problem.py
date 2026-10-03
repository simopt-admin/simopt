"""Base classes for simulation optimization problems and models."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from boltons.typeutils import classproperty
from pydantic import BaseModel

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt import dsl
from simopt.model import Model
from simopt.utils import get_specifications
from simopt_solvers.problem_types import ConstraintType, VariableType
from simopt_solvers.solution import Solution


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

        self._optimization_problem: dsl.Model | None = None
        self.rng_list: list[MRG32k3a] = []

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
            non_decision_factors = set(self.model.factors.keys()) - self.model_decision_factors
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
        non_decision_factors = set(self.model.factors.keys()) - self.model_decision_factors
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
    def dim(self) -> int:
        """Number of decision variables."""
        return self.optimization_problem.dim

    @property
    def lower_bounds(self) -> tuple[int | float, ...]:
        """Lower bound for each decision variable."""
        return self.optimization_problem.lower_bounds()

    @property
    def upper_bounds(self) -> tuple[int | float, ...]:
        """Upper bound for each decision variable."""
        return self.optimization_problem.upper_bounds()

    @property
    def optimization_problem(self) -> dsl.Model:
        """Optimization problem expressed with the SimOpt DSL."""
        if self._optimization_problem is None:
            self._optimization_problem = self.build()
            self._optimization_problem.set_random_solution(self.get_random_solution)
        return self._optimization_problem

    def build(self) -> dsl.Model:
        """Build an optimization problem with the SimOpt DSL."""
        raise NotImplementedError(f"{type(self).__name__} does not build a SimOpt DSL problem")

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

    def add_simulation(
        self,
        target: dsl.Model,
        decisions: Mapping[str, dsl.Variable | dsl.VectorVariable],
        *,
        name: str = "",
    ) -> dsl.Simulation:
        """Add this problem's simulation model to the DSL model.

        The problem supplies the replication adapter, RNG count, and gradient
        availability so problem implementations only need to declare the
        simulation's decisions.

        Args:
            target: DSL model to which the simulation is added.
            decisions: Decision variables passed to the simulation.
            name: Name of the simulation.

        Returns:
            The simulation added to ``target``.
        """

        def run(
            decisions: dict[str, float | tuple[float, ...]], rngs: list[MRG32k3a]
        ) -> tuple[dict, dict]:
            decision_vector = []
            for decision in decisions.values():
                if isinstance(decision, tuple):
                    decision_vector.extend(decision)
                else:
                    decision_vector.append(decision)
            decision_vector = tuple(decision_vector)
            decision_factors = self.vector_to_factor_dict(decision_vector)
            factors = self.model.config.model_copy(update=decision_factors)
            return self.model.replicate(factors, rngs)

        return target.add_simulation(
            name=name,
            run=run,
            decisions=decisions,
            n_rngs=self.model.n_rngs,
            gradients=self.gradient_available,
        )

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
            for x_i, lb, ub in zip(x, self.lower_bounds, self.upper_bounds, strict=False)
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

    def simulate(self, solution: "Solution", num_macroreps: int = 1) -> None:
        """Simulate `m` i.i.d. replications at solution `x`.

        Args:
            solution (Solution): Solution to evaluate.
            num_macroreps (int, optional): Number of macroreplications to simulate at
                `x`. Defaults to 1.
        """
        solution.simulate(self.optimization_problem, num_macroreps)
