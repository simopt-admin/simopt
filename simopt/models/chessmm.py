"""Simulate matching of chess players on an online platform."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt import dsl
from simopt.base import (
    ConstraintType,
    Model,
    Problem,
    VariableType,
)
from simopt.input_models import Exp
from simopt.simulations.chessmm import (
    MAX_ALLOWABLE_DIFF,
    ChessMatchmakingConfig,
    EloInputModel,
    replicate,
)
from simopt.utils import override


class ChessAvgDifferenceConfig(BaseModel):
    """Configuration model for Chess Average Difference Problem.

    A problem configuration that minimizes the average difference in Elo ratings
    between matched chess players while maintaining wait time constraints.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(MAX_ALLOWABLE_DIFF,),
            description="initial solution",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=1000,
            description="max # of replications for a solver to take",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    upper_time: Annotated[
        float,
        Field(
            default=5.0,
            description="upper bound on wait time",
            gt=0,
        ),
    ]


class ChessMatchmaking(Model[ChessMatchmakingConfig]):
    """Matchmaking model following an Elo distribution.

    A model that simulates a matchmaking problem with a Elo (truncated normal)
    distribution of players and Poisson arrivals and returns the average difference
    between matched players.
    """

    class_name_abbr: ClassVar[str] = "CHESS"
    class_name: ClassVar[str] = "Chess Matchmaking"
    config_class: ClassVar[type[ChessMatchmakingConfig]] = ChessMatchmakingConfig
    n_rngs: ClassVar[int] = 2
    n_responses: ClassVar[int] = 2

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the ChessMatchmaking model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.elo_model = EloInputModel()
        self.arrival_model = Exp()

    def replicate(self, factors: ChessMatchmakingConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): List of random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict[str, dict]]: A tuple containing:
                - dict: Performance measures of interest, including:
                    - "avg_diff": Average Elo difference between all pairs.
                    - "avg_wait_time": Average waiting time.
                - dict[str, dict]: Gradient estimates for each response.
        """
        avg_diff, avg_wait_time = replicate(factors, rngs, self.elo_model, self.arrival_model)
        # Compose responses and gradients.
        responses = {
            "avg_diff": avg_diff,
            "avg_wait_time": avg_wait_time,
        }
        return responses, {}


class ChessAvgDifference(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "CHESS-1"
    class_name: ClassVar[str] = "Min Avg Difference for Chess Matchmaking"
    config_class: ClassVar[type[BaseModel]] = ChessAvgDifferenceConfig
    model_class: ClassVar[type[Model]] = ChessMatchmaking
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 1
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.STOCHASTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"allowable_diff"}

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        allowable_diff = problem.add_continuous_variable(
            lb=0.0, ub=2400.0, initial=self.factors["initial_solution"][0]
        )
        simulation = self.add_simulation(problem, {"allowable_diff": allowable_diff})
        problem.minimize(dsl.mean(simulation.metric("avg_diff")))
        problem.add_stochastic_constraint(
            simulation.metric("avg_wait_time") <= self.factors["upper_time"]
        )
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"allowable_diff": vector[0]}

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(x_val > 0 for x_val in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        val = rand_sol_rng.normalvariate(150, 50)
        return (min(max(0.0, float(val)), 2400.0),)
