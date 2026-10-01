"""Standalone Python simulation for chess matchmaking."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from random import Random
from typing import Annotated, Final, cast

import numpy as np
import simpy
from pydantic import BaseModel, Field
from scipy import special

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import input_model, simulation
from simopt.input_models import InputModel

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


@input_model
class EloInputModel(InputModel):
    """Input model for player Elo ratings."""

    def random(
        self, rng: Random, mean: float, std: float, min_rating: float, max_rating: float
    ) -> float:
        """Draw a truncated normal rating within [min_rating, max_rating]."""
        while True:
            rating = rng.normalvariate(mean, std)
            if min_rating <= rating <= max_rating:
                return rating


@dataclass
class State:
    """Rating difference accumulated by the player arrival process."""

    total_diff: float = 0  # TODO: make this do something


@simulation
def replicate(
    factors: ChessMatchmakingConfig,
    rngs: list[MRG32k3a],
    elo_model: InputModel,
    arrival_model: InputModel,
) -> tuple[float, float]:
    """Return average matched rating difference and player wait time for one replication."""
    # Constants
    num_players = factors.num_players
    num_players_range = range(num_players)
    elo_mean = factors.elo_mean
    elo_sd = factors.elo_sd
    elo_min, elo_max = 0, 2400
    allowable_diff = factors.allowable_diff
    poisson_rate = factors.poisson_rate

    # Initialize statistics.
    # Incoming players are initialized with a wait time of 0.
    wait_times = np.zeros(num_players)
    env = simpy.Environment()
    waiting_players = simpy.FilterStore(env, capacity=num_players)
    state = State()
    elo_diffs = []

    def player_arrivals() -> Generator[simpy.Event, object, None]:
        for player_idx in num_players_range:
            # Generate the player's Elo rating and interarrival time.
            player_rating = elo_model.random(rngs[0], elo_mean, elo_sd, elo_min, elo_max)
            interarrival_time = arrival_model.random(rngs[1], poisson_rate)
            yield env.timeout(interarrival_time)

            # Try to match the player
            for waiting_player in waiting_players.items:
                waiting_rating = waiting_player[0]
                diff = abs(player_rating - waiting_rating)
                if diff <= allowable_diff:
                    state.total_diff += diff
                    elo_diffs.append(diff)
                    matched_player = cast(
                        tuple[float, int, float],
                        (
                            yield waiting_players.get(
                                lambda player, incoming_rating=player_rating: (
                                    abs(incoming_rating - player[0]) <= allowable_diff
                                )
                            )
                        ),
                    )
                    wait_times[matched_player[1]] = env.now - matched_player[2]
                    break
            # If break did not execute, then the player was not matched.
            else:
                yield waiting_players.put((player_rating, player_idx, env.now))

    env.process(player_arrivals())
    env.run()

    # Players still in the pool have waited through the end of the replication.
    for _, player_idx, arrival_time in waiting_players.items:
        wait_times[player_idx] = env.now - arrival_time

    # If there weren't any matches, the elo_diffs list will be empty.
    avg_diff = np.mean(elo_diffs) if elo_diffs else np.nan
    return float(avg_diff), float(np.mean(wait_times))
