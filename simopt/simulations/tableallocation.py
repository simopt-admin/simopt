"""Standalone Python simulation for restaurant table allocation."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Annotated, Self

import numpy as np
import simpy
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel


class TableAllocationConfig(BaseModel):
    """Configuration for the Table Allocation model."""

    n_hours: Annotated[
        float,
        Field(
            default=5.0,
            description="number of hours to simulate",
            gt=0,
        ),
    ]
    capacity: Annotated[
        int,
        Field(
            default=80,
            description="maximum capacity of restaurant",
            gt=0,
        ),
    ]
    table_cap: Annotated[
        list[int],
        Field(
            default=[2, 4, 6, 8],
            description="seating capacity of each type of table",
        ),
    ]
    lambda_: Annotated[
        list[float],
        Field(
            default=[3, 6, 3, 3, 2, 4 / 3, 6 / 5, 1],
            description="average number of arrivals per hour",
            alias="lambda",
        ),
    ]
    service_time_means: Annotated[
        list[float],
        Field(
            default=[20, 25, 30, 35, 40, 45, 50, 60],
            description="mean service time (in minutes)",
        ),
    ]
    table_revenue: Annotated[
        list[float],
        Field(
            default=[15, 30, 45, 60, 75, 90, 105, 120],
            description="revenue earned for each group size",
        ),
    ]
    num_tables: Annotated[
        list[int],
        Field(
            default=[10, 5, 4, 2],
            description="number of tables of each capacity",
        ),
    ]

    def _check_table_cap(self) -> None:
        if any(x <= 0 for x in self.table_cap):
            raise ValueError("All elements in table_cap must be greater than 0.")

    def _check_lambda(self) -> None:
        if any(lam < 0 for lam in self.lambda_):
            raise ValueError("Each element in lambda must be non-negative.")

    def _check_service_time_means(self) -> None:
        if any(x <= 0 for x in self.service_time_means):
            raise ValueError("Each element in service_time_means must be positive.")

    def _check_table_revenue(self) -> None:
        if any(x < 0 for x in self.table_revenue):
            raise ValueError("Each element in table_revenue must be non-negative.")

    def _check_num_tables(self) -> None:
        if any(x < 0 for x in self.num_tables):
            raise ValueError("Each element in num_tables must be greater than or equal to 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_table_cap()
        self._check_lambda()
        self._check_service_time_means()
        self._check_table_revenue()
        self._check_num_tables()

        if len(self.num_tables) != len(self.table_cap):
            raise ValueError("The length of num_tables must be equal to the length of table_cap.")
        if len(self.lambda_) != max(self.table_cap):
            raise ValueError("The length of lamda must be equal to the maximum value in table_cap.")
        if len(self.lambda_) != len(self.service_time_means):
            raise ValueError(
                "The length of lambda must be equal to the length of service_time_means."
            )
        if len(self.service_time_means) != len(self.table_revenue):
            raise ValueError(
                "The length of service_time_means must be equal to the length of table_revenue."
            )
        return self


@dataclass
class State:
    """Revenue shared by group processes within one replication."""

    total_rev: float = 0


@simulation
def replicate(
    factors: TableAllocationConfig,
    rngs: list[MRG32k3a],
    arrival_time_model: InputModel,
    arrival_number_model: InputModel,
    group_size_model: InputModel,
    service_time_model: InputModel,
) -> tuple[float, float]:
    """Return total revenue and the fraction of arriving groups seated in one replication."""
    num_tables = factors.num_tables
    # TODO: figure out how floats are getting into the num_tables list
    num_tables = [int(n) for n in num_tables]
    n_hours = factors.n_hours
    f_lambda = factors.lambda_
    table_cap = factors.table_cap
    max_table_cap = max(table_cap)
    service_time_means = factors.service_time_means
    table_revenue = factors.table_revenue
    # Track total revenue.
    state = State()
    # Generate total number of arrivals in the period
    n_arrivals = arrival_number_model.random(rngs[1], round(n_hours * sum(f_lambda)))
    # Generate arrival times in minutes
    arrival_times = 60 * np.sort(
        [arrival_time_model.random(rngs[0], 0, n_hours) for _ in range(n_arrivals)]
    )
    # Track seating rate
    found = np.zeros(n_arrivals)
    # Precompute options for group sizes.
    group_size_options = list(range(1, max_table_cap + 1))
    env = simpy.Environment()
    tables = [
        [simpy.Resource(env, capacity=1) for _ in range(num_tables[k])]
        for k in range(len(num_tables))
    ]

    def group(n: int) -> Generator[simpy.Event, object, None]:
        yield env.timeout(arrival_times[n])

        # Determine group size.
        group_size = group_size_model.random(
            rngs[2], population=group_size_options, weights=f_lambda
        )

        # Find smallest table size to start search.
        table_size_idx = 0
        while table_cap[table_size_idx] < group_size:
            table_size_idx += 1

        # Find smallest available table.
        result = next(
            (
                (k, j)
                for k in range(table_size_idx, len(num_tables))
                for j, table in enumerate(tables[k])
                if table.count == 0
            ),
            None,
        )
        # If no table is available, move on to next group.
        if result is None:
            return
        k, j = result
        with tables[k][j].request() as request:
            yield request
            # Mark group as seated.
            found[n] = 1
            # Sample service time.
            service_time = service_time_model.random(
                rngs[3], 1 / service_time_means[group_size - 1]
            )
            # Update revenue.
            state.total_rev += table_revenue[group_size - 1]
            # Hold the table for the full sampled service duration.
            yield env.timeout(service_time)

    # Pass through all arrivals of groups to the restaurants.
    for n in range(n_arrivals):
        env.process(group(n))
    env.run()
    # Calculate responses from simulation data.
    return state.total_rev, sum(found) / len(found)
