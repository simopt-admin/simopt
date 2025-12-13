"""Base classes for simulation optimization solvers."""

import contextlib
from abc import ABC, abstractmethod
from typing import Annotated, ClassVar

import pandas as pd
from boltons.typeutils import classproperty
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.problem import Problem, Solution
from simopt.problem_types import ConstraintType, ObjectiveType, VariableType
from simopt.utils import get_specifications


class BudgetExhaustedError(Exception):
    """Raised when a solver exceeds its allotted simulation budget.

    This exception is thrown by :class:`Budget` when a call to
    :meth:`Budget.request` asks for more replications than remain in the
    available budget. It is caught in :meth:`Solver.run` to stop the
    macroreplication cleanly once the budget is exhausted.
    """


class Budget:
    """Tracks and enforces a solver's replication budget.

    A ``Budget`` instance is attached to each solver run and measures the number of
    simulation replications consumed. Solvers should call :meth:`request` before
    taking replications. If the request would exceed ``total``, a
    :class:`BudgetExhaustedException` is raised. This provides a consistent way for
    solvers to terminate exactly at the specified budget.

    Args:
        total (int): Total number of replications available for the run.
    """

    def __init__(self, total: int) -> None:
        """Initialize object with the total number of replications available."""
        self.total = total
        self._used = 0

    def request(self, amount: int) -> None:
        """Consume ``amount`` replications from the budget.

        Typical usage is to call ``request(r)`` immediately before taking ``r``
        replications at the current solution.

        Args:
            amount (int): Number of replications to consume.

        Raises:
            BudgetExhaustedException: If ``amount`` would cause usage to exceed
                :attr:`total`.
        """
        if self._used + amount > self.total:
            raise BudgetExhaustedError()
        self._used += amount

    @property
    def used(self) -> int:
        """Number of replications consumed so far."""
        return self._used

    @property
    def remaining(self) -> int:
        """Number of replications still available (``total - used``)."""
        return self.total - self._used


class SolverConfig(BaseModel):
    """Base class for solver configuration."""

    crn_across_solns: Annotated[
        bool, Field(default=True, description="use CRN across solutions?")
    ]


class Solver(ABC):
    """Base class to implement simulation-optimization solvers.

    This class defines the core structure for simulation-optimization
    solvers in SimOpt. Subclasses must implement the required methods
    for running simulations and updating solutions.

    Args:
        name (str): Name of the solver.
        fixed_factors (dict): Dictionary of user-specified solver factors.
    """

    class_name_abbr: ClassVar[str]
    """Short name of the solver class."""

    class_name: ClassVar[str]
    """Long name of the solver class."""

    config_class: ClassVar[type[SolverConfig]]
    """Configuration class for the solver."""

    objective_type: ClassVar[ObjectiveType]
    """Description of objective types."""

    constraint_type: ClassVar[ConstraintType]
    """Description of constraint types."""

    variable_type: ClassVar[VariableType]
    """Description of variable types."""

    gradient_needed: ClassVar[bool]
    """True if gradient of objective function is needed, otherwise False."""

    def __init__(self, name: str = "", fixed_factors: dict | None = None) -> None:
        """Initialize a solver object.

        Args:
            name (str, optional): Name of the solver. Defaults to an empty string.
            fixed_factors (dict | None, optional): Dictionary of user-specified solver
                factors. Defaults to None.
        """
        self.name = name or self.name

        fixed_factors = fixed_factors or {}
        self.config = self.config_class(**fixed_factors)

        self.rng_list: list[MRG32k3a] = []
        self.solution_progenitor_rngs: list[MRG32k3a] = []

        self.recommended_solns = []
        self.intermediate_budgets = []

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

    @classproperty
    def compatibility(cls) -> str:  # noqa: N805
        """Compatibility of the solver."""
        return (
            f"{cls.objective_type.symbol()}"
            f"{cls.constraint_type.symbol()}"
            f"{cls.variable_type.symbol()}"
            f"{'G' if cls.gradient_needed else 'N'}"
        )

    @classproperty
    def specifications(cls) -> dict[str, dict]:  # noqa: N805
        """Details of each factor (for GUI, data validation, and defaults)."""
        return get_specifications(cls.config_class)

    @property
    def factors(self) -> dict:
        """Changeable factors (i.e., parameters) of the solver."""
        return self.config.model_dump(by_alias=True)

    def attach_rngs(self, rng_list: list[MRG32k3a]) -> None:
        """Attach a list of random-number generators to the solver.

        Args:
            rng_list (list[``mrg32k3a.mrg32k3a.MRG32k3a``]): List of random-number
                generators used for the solver's internal purposes.
        """
        self.rng_list = rng_list

    @abstractmethod
    def solve(self, problem: Problem) -> None:
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

    def run(self, problem: Problem) -> pd.DataFrame:
        """Run the solver on a problem.

        Args:
            problem (Problem): The problem to solve.

        Returns:
            tuple[list[Solution], list[int]]: A tuple containing a list of solutions
            and a list of intermediate budgets.
        """
        self.budget = Budget(problem.factors["budget"])
        with contextlib.suppress(BudgetExhaustedError):
            self.solve(problem)

        recommended_solns = self.recommended_solns
        intermediate_budgets = self.intermediate_budgets
        self.recommended_solns = []
        self.intermediate_budgets = []

        df = pd.DataFrame(
            {
                "step": range(len(recommended_solns)),
                "budget": intermediate_budgets,
                "solution": recommended_solns,
            }
        )
        df["solution"] = df["solution"].apply(lambda solution: solution.x)

        return df

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
        if not self.config.crn_across_solns:  # If CRN are not used ...
            # ...advance each rng to start of the substream
            # substream = current substream + # of model RNGs.
            for rng in self.solution_progenitor_rngs:
                for _ in range(problem.model.n_rngs):
                    rng.advance_substream()
        return new_solution
