"""Base classes for simulation optimization solvers."""

import contextlib
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
import pandas as pd
from boltons.typeutils import classproperty
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt import dsl
from simopt.problem import Solution
from simopt.problem_types import ConstraintType, ObjectiveType, VariableType
from simopt.utils import get_specifications


class BudgetExhaustedError(Exception):
    """Raised when a solver exceeds its allotted simulation budget.

    This exception is thrown by :class:`Budget` when a call to
    :meth:`Budget.request` asks for more replications than remain in the
    available budget. It is caught in solver runs to stop the macroreplication
    cleanly once the budget is exhausted.
    """


class Budget:
    """Tracks and enforces a solver's replication budget.

    A ``Budget`` instance is attached to each solver run and measures the number of
    simulation replications consumed. Solvers should call :meth:`request` before
    taking replications. If the request would exceed ``total``, a
    :class:`BudgetExhaustedError` is raised. This provides a consistent way for
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
            BudgetExhaustedError: If ``amount`` would cause usage to exceed
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


class Context:
    """Owns runtime services needed while executing one solver macroreplication."""

    def __init__(
        self,
        model: dsl.Model,
        total_budget: int,
        solution_progenitor_rngs: list[MRG32k3a],
        solver_rngs: list[MRG32k3a],
        crn_across_solns: bool,
    ) -> None:
        """Initialize the runtime context."""
        self.model = model
        self.budget = Budget(total_budget)
        self.solution_progenitor_rngs = solution_progenitor_rngs
        self.solver_rngs = solver_rngs
        self.crn_across_solns = crn_across_solns
        self.recommended_solns: list[Solution] = []
        self.intermediate_budgets: list[int] = []

    def rng(self, index: int) -> MRG32k3a:
        """Return the solver RNG at ``index`` for the solver's internal randomness."""
        return self.solver_rngs[index]

    def evaluate(self, x: Solution | tuple, n_reps: int = 1) -> Solution:
        """Evaluate a solution by simulating it and returning the updated object."""
        if not isinstance(x, Solution):
            # Create new solution with attached rngs.
            x = Solution(x=x, rng_list=[deepcopy(rng) for rng in self.solution_progenitor_rngs])
            # Manipulate progenitor rngs to prepare for next new solution.
            if not self.crn_across_solns:
                # ...advance each rng to start of the substream
                # substream = current substream + # of model RNGs.
                for rng in self.solution_progenitor_rngs:
                    for _ in range(self.model.n_rngs):
                        rng.advance_substream()
        start_reps = x.n_reps
        self.budget.request(n_reps)
        x.simulate(self.model, n_reps)
        if self.model.objective_sense == "maximize":
            factor = np.array([-1])
            for i in range(start_reps, x.n_reps):
                x._objectives[i] = factor * x._objectives[i]
                x._objectives_gradients[i] = factor[:, np.newaxis] * x._objectives_gradients[i]
            x._objectives_array = None
            x._objectives_gradients_array = None
        return x

    def log(self, x: tuple | Solution) -> None:
        """Record a recommended solution and the budget where it was observed."""
        if isinstance(x, tuple):
            x = Solution(x, [])
        self.recommended_solns.append(x)
        self.intermediate_budgets.append(self.budget.used)

    def history(self) -> pd.DataFrame:
        """Build a solver-history dataframe from recorded recommendations."""
        df = pd.DataFrame(
            {
                "step": range(len(self.recommended_solns)),
                "budget": self.intermediate_budgets,
                "solution": self.recommended_solns,
            }
        )
        df["solution"] = df["solution"].apply(lambda solution: solution.x)
        return df


class SolverConfig(BaseModel):
    """Base class for solver configuration."""

    crn_across_solns: Annotated[bool, Field(default=True, description="use CRN across solutions?")]


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

    @abstractmethod
    def solve(self, problem: dsl.Model, ctx: Context) -> None:
        """Run a single macroreplication of a solver on a problem.

        Args:
            problem (dsl.Model): Simulation-optimization problem to solve.
            ctx (Context): Runtime services for the current run.
        """
        raise NotImplementedError


def _make_rngs(n_rngs: int, seed: int) -> tuple[list[MRG32k3a], list[MRG32k3a]]:
    """Create the simulation and solver RNGs for run ``seed``."""
    # Stream 0: reserved for taking post-replications
    # Stream 1: reserved for bootstrapping
    # Streams 3, 4, ...: one per run (seed), i.e. per macroreplication
    #     Substreams 0, ..., n_rngs - 1: simulation RNGs (solution progenitors)
    #     Substreams n_rngs, n_rngs + 1, n_rngs + 2: solver RNGs (ctx.rng(0..2))

    # Create RNGs for simulation
    simulation_rngs = [MRG32k3a(s_ss_sss_index=[seed + 3, i, 0]) for i in range(n_rngs)]

    # Create RNGs for the solver
    solver_rngs = [MRG32k3a(s_ss_sss_index=[seed + 3, n_rngs + i, 0]) for i in range(3)]

    return simulation_rngs, solver_rngs


@dataclass
class SolveResult:
    """Outcome of running a solver on a model.

    Attributes:
        x: Final recommended solution, or None if the solver recommended nothing
            before the budget ran out.
        history: Recommended solutions with the budget used when each was
            recommended (columns ``step``, ``budget``, ``solution``).
        budget_used: Number of replications consumed.
    """

    x: tuple | None
    history: pd.DataFrame
    budget_used: int


def solve(model: dsl.Model, solver: Solver, budget: int, seed: int = 0) -> SolveResult:
    """Run ``solver`` on ``model`` until ``budget`` replications are used.

    Args:
        model (dsl.Model): Simulation-optimization model to solve.
        solver (Solver): Solver to run.
        budget (int): Maximum number of simulation replications.
        seed (int, optional): Run index selecting independent random-number streams.
            The same seed reproduces the same run; ``seed=k`` matches
            macroreplication ``k`` of a SimOpt experiment. Defaults to 0.

    Returns:
        SolveResult: Final recommendation and recommendation history.
    """
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    simulation_rngs, solver_rngs = _make_rngs(model.n_rngs, seed)
    ctx = Context(
        model=model,
        total_budget=budget,
        solution_progenitor_rngs=simulation_rngs,
        solver_rngs=solver_rngs,
        crn_across_solns=solver.config.crn_across_solns,
    )
    with contextlib.suppress(BudgetExhaustedError):
        solver.solve(model, ctx)

    history = ctx.history()
    x = history["solution"].iloc[-1] if len(history) else None
    return SolveResult(x=x, history=history, budget_used=ctx.budget.used)
