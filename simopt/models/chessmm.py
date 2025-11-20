"""Simulate matching of chess players on an online platform."""

from __future__ import annotations

from random import Random
from typing import Annotated, ClassVar, Final

import numpy as np
from pydantic import BaseModel, Field
from scipy import special

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    StochasticConstraint,
    VariableType,
)
from simopt.input_models import InputModel, Poisson

MEAN_ELO: Final[int] = 1200
MAX_ALLOWABLE_DIFF: Final[int] = 150


class ChessMatchmakingConfig(BaseModel):
    """Configuration model for Chess Matchmaking simulation.

    A model that simulates a matchmaking problem with a Elo (truncated normal)
    distribution of players and Poisson arrivals and returns the average difference
    between matched players.
    """

    elo_mean: Annotated[
        float,
        Field(
            default=MEAN_ELO,
            description="mean of normal distribution for Elo rating",
            gt=0,
        ),
    ]
    elo_sd: Annotated[
        float,
        Field(
            default=round(MEAN_ELO / (np.sqrt(2) * special.erfcinv(1 / 50)), 1),
            description="standard deviation of normal distribution for Elo rating",
            gt=0,
        ),
    ]
    poisson_rate: Annotated[
        float,
        Field(
            default=1.0,
            description="rate of Poisson process for player arrivals",
            gt=0,
        ),
    ]
    num_players: Annotated[
        int,
        Field(
            default=1000,
            description="number of players",
            gt=0,
        ),
    ]
    allowable_diff: Annotated[
        float,
        Field(
            default=MAX_ALLOWABLE_DIFF,
            description="maximum allowable difference between Elo ratings",
            gt=0,
        ),
    ]


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


class EloInputModel(InputModel):
    """Input model for player Elo ratings."""

    rng: Random | None = None

    def random(
        self, mean: float, std: float, min_rating: float, max_rating: float
    ) -> float:
        """Draw a truncated normal rating within [min_rating, max_rating]."""
        assert self.rng is not None
        while True:
            rating = self.rng.normalvariate(mean, std)
            if min_rating <= rating <= max_rating:
                return rating


class ChessMatchmaking(Model):
    """Matchmaking model following an Elo distribution.

    A model that simulates a matchmaking problem with a Elo (truncated normal)
    distribution of players and Poisson arrivals and returns the average difference
    between matched players.
    """

    class_name_abbr: ClassVar[str] = "CHESS"
    class_name: ClassVar[str] = "Chess Matchmaking"
    config_class: ClassVar[type[BaseModel]] = ChessMatchmakingConfig
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
        self.arrival_model = Poisson()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.elo_model.set_rng(rng_list[0])
        self.arrival_model.set_rng(rng_list[1])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): List of random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict[str, dict]]: A tuple containing:
                - dict: Performance measures of interest, including:
                    - "avg_diff": Average Elo difference between all pairs.
                    - "avg_wait_time": Average waiting time.
                - dict[str, dict]: Gradient estimates for each response.
        """
        # Constants
        num_players = self.factors["num_players"]
        num_players_range = range(num_players)
        elo_mean = self.factors["elo_mean"]
        elo_sd = self.factors["elo_sd"]
        elo_min, elo_max = 0, 2400
        allowable_diff = self.factors["allowable_diff"]
        poisson_rate = self.factors["poisson_rate"]

        # Generate Elo ratings (normal distribution).
        player_ratings = [
            self.elo_model.random(elo_mean, elo_sd, elo_min, elo_max)
            for _ in num_players_range
        ]

        # Generate interarrival times (Poisson distribution).
        interarrival_times = [
            self.arrival_model.random(poisson_rate) for _ in num_players_range
        ]

        # Initialize statistics.
        # Incoming players are initialized with a wait time of 0.
        wait_times = np.zeros(num_players)
        waiting_players = []
        total_diff = 0  # TODO: make this do something
        elo_diffs = []

        # Simulate arrival and matching and players.
        for interarrival_time, player_rating in zip(
            interarrival_times, player_ratings, strict=False
        ):
            # Try to match the player
            for i, waiting_rating in enumerate(waiting_players):
                diff = abs(player_rating - waiting_rating)
                if diff <= allowable_diff:
                    total_diff += diff
                    elo_diffs.append(diff)
                    waiting_players.pop(i)
                    break
                wait_times[i] += interarrival_time
            # If break did not execute, then the player was not matched.
            else:
                waiting_players.append(player_rating)

        # If there weren't any matches, the elo_diffs list will be empty.
        avg_diff = np.mean(elo_diffs) if elo_diffs else np.nan
        # Compose responses and gradients.
        responses = {
            "avg_diff": avg_diff,
            "avg_wait_time": np.mean(wait_times),
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

    @property
    def dim(self) -> int:  # noqa: D102
        return 1

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,)

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (2400,)

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"allowable_diff": vector[0]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (factor_dict["allowable_diff"],)

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        return RepResult(
            objectives=[Objective(stochastic=responses["avg_diff"])],
            stochastic_constraints=[
                StochasticConstraint(
                    stochastic=responses["avg_wait_time"],
                    deterministic=-1 * self.factors["upper_time"],
                )
            ],
        )

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return all(x_val > 0 for x_val in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        val = rand_sol_rng.normalvariate(150, 50)
        return (min(max(0.0, float(val)), 2400.0),)
