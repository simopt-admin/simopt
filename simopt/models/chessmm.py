"""Simulate matching of chess players on an online platform."""

from __future__ import annotations

from typing import Annotated, ClassVar, Final

import numpy as np
from pydantic import BaseModel, Field
from scipy import special

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import InputModel, Poisson
from simopt.utils import override

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

    def set_rng(self, rng: random.Random) -> None:  # noqa: D102
        self.rng = rng

    def unset_rng(self) -> None:  # noqa: D102
        self.rng = None

    def random(
        self, mean: float, std: float, min_rating: float, max_rating: float
    ) -> float:
        """Draw a truncated normal rating within [min_rating, max_rating]."""
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

    config_class: ClassVar[type[BaseModel]] = ChessMatchmakingConfig
    class_name_abbr: str = "CHESS"
    class_name: str = "Chess Matchmaking"
    n_rngs: int = 2
    n_responses: int = 2

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

    @override
    def check_simulatable_factors(self) -> bool:
        # No factors need cross-checked
        return True

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
        for interarrival_time, player_rating in zip(interarrival_times, player_ratings):
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
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class ChessAvgDifference(Problem):
    """Base class to implement simulation-optimization problems."""

    config_class: ClassVar[type[BaseModel]] = ChessAvgDifferenceConfig
    model_class: ClassVar[type[Model]] = ChessMatchmaking
    class_name_abbr: str = "CHESS-1"
    class_name: str = "Min Avg Difference for Chess Matchmaking"
    n_objectives: int = 1
    n_stochastic_constraints: int = 1
    minmax: tuple[int] = (-1,)
    constraint_type: ConstraintType = ConstraintType.STOCHASTIC
    variable_type: VariableType = VariableType.CONTINUOUS
    gradient_available: bool = False
    optimal_value: float | None = None
    optimal_solution: tuple | None = None
    model_default_factors: dict = {}
    model_decision_factors: set[str] = {"allowable_diff"}
    dim: int = 1
    lower_bounds: tuple = (0,)
    upper_bounds: tuple = (2400,)

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"allowable_diff": vector[0]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return (factor_dict["allowable_diff"],)

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["avg_diff"],)

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
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
        return (response_dict["avg_wait_time"],)

    def deterministic_stochastic_constraints_and_gradients(self) -> tuple[tuple, tuple]:
        """Compute deterministic components of stochastic constraints.

        Returns:
            tuple:
                - tuple: The deterministic components of the stochastic constraints.
                - tuple: The gradients of those deterministic components.
        """
        det_stoch_constraints = (-1 * self.factors["upper_time"],)
        det_stoch_constraints_gradients = ((0,),)
        return det_stoch_constraints, det_stoch_constraints_gradients

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (0,)
        det_objectives_gradients = ()
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(x_val > 0 for x_val in x)

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return (min(max(0, rand_sol_rng.normalvariate(150, 50)), 2400),)
