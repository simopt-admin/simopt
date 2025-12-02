"""Simulate multiple periods of arrival and seating at a restaurant."""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Sequence
from typing import Annotated, ClassVar, Self, cast

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    VariableType,
)
from simopt.input_models import Exp, Poisson, Uniform, WeightedChoice


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
            raise ValueError(
                "Each element in num_tables must be greater than or equal to 0."
            )

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_table_cap()
        self._check_lambda()
        self._check_service_time_means()
        self._check_table_revenue()
        self._check_num_tables()

        if len(self.num_tables) != len(self.table_cap):
            raise ValueError(
                "The length of num_tables must be equal to the length of table_cap."
            )
        if len(self.lambda_) != max(self.table_cap):
            raise ValueError(
                "The length of lamda must be equal to the maximum value in table_cap."
            )
        if len(self.lambda_) != len(self.service_time_means):
            raise ValueError(
                "The length of lambda must be equal to the length of "
                "service_time_means."
            )
        if len(self.service_time_means) != len(self.table_revenue):
            raise ValueError(
                "The length of service_time_means must be equal to the length of "
                "table_revenue."
            )
        return self


class TableAllocationMaxRevConfig(BaseModel):
    """Configuration model for Table Allocation Max Revenue Problem.

    Max Revenue for Restaurant Table Allocation simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default=(10, 5, 4, 2),
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


class TableAllocation(Model):
    """Table Allocation Model.

    A model that simulates a table capacity allocation problem at a restaurant
    with a homogenous Poisson arrvial process and exponential service times.
    Returns expected maximum revenue.
    """

    class_name_abbr: ClassVar[str] = "TABLEALLOCATION"
    class_name: ClassVar[str] = "Restaurant Table Allocation"
    config_class: ClassVar[type[BaseModel]] = TableAllocationConfig
    n_rngs: ClassVar[int] = 3
    n_responses: ClassVar[int] = 2

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Table Allocation Model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.arrival_time_model = Uniform()
        self.arrival_number_model = Poisson()
        self.group_size_model = WeightedChoice()
        self.service_time_model = Exp()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.arrival_time_model.set_rng(rng_list[0])
        self.arrival_number_model.set_rng(rng_list[0])
        self.group_size_model.set_rng(rng_list[1])
        self.service_time_model.set_rng(rng_list[2])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "total_revenue": Total revenue earned over the simulation period.
                    - "service_rate": Fraction of customer arrivals that are seated.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """

        def fast_weighted_choice(
            population: Sequence[int], weights: Sequence[float], rng: MRG32k3a
        ) -> int:
            """Select a single element from a population based on weights.

            Designed to be faster than `random.choices()` when only one element
            is needed.

            Args:
                population (Sequence[int]): The population to select from.
                weights (Sequence[float]): The weights for each element in the
                    population.
                rng (MRG32k3a): The random number generator to use for selection.

            Returns:
                int: The selected element from the population.
            """
            # Calculate cumulative weights
            cum_weights = list(itertools.accumulate(weights))
            # Generate a value somewhere between 0 and the sum of weights
            x = rng.random() * cum_weights[-1]
            # Find the index of the first cumulative weight that is >= x
            # Return the corresponding element from the population
            return population[bisect.bisect(cum_weights, x)]

        num_tables = self.factors["num_tables"]
        # TODO: figure out how floats are getting into the num_tables list
        num_tables = [int(n) for n in num_tables]
        n_hours = self.factors["n_hours"]
        f_lambda = self.factors["lambda"]
        table_cap = self.factors["table_cap"]
        max_table_cap = max(table_cap)
        service_time_means = self.factors["service_time_means"]
        table_revenue = self.factors["table_revenue"]
        # Track total revenue.
        total_rev = 0
        # Track table availability.
        # (i,j) is the time that jth table of size i becomes available.
        table_avail = np.zeros((4, max(num_tables)))
        # Generate total number of arrivals in the period
        n_arrivals = self.arrival_number_model.random(round(n_hours * sum(f_lambda)))
        # Generate arrival times in minutes
        arrival_times = 60 * np.sort(
            [self.arrival_time_model.random(0, n_hours) for _ in range(n_arrivals)]
        )
        # Track seating rate
        found = np.zeros(n_arrivals)
        # Precompute options for group sizes.
        group_size_options = list(range(1, max_table_cap + 1))
        # Pass through all arrivals of groups to the restaurants.
        for n in range(n_arrivals):
            # Determine group size.
            group_size = cast(
                int,
                self.group_size_model.random(
                    population=group_size_options,
                    weights=f_lambda,
                ),
            )

            # Find smallest table size to start search.
            table_size_idx = 0
            while table_cap[table_size_idx] < group_size:
                table_size_idx += 1

            # Find smallest available table.
            def find_table(table_size_idx: int, n: int) -> tuple[int, int] | None:
                for k in range(table_size_idx, len(num_tables)):
                    for j in range(num_tables[k]):
                        # Check if table is currently available.
                        if table_avail[k, j] < arrival_times[n]:
                            return k, j
                # Return None if no table is available.
                return None

            result = find_table(table_size_idx, n)
            # If no table is available, move on to next group.
            if result is None:
                continue
            k, j = result
            # Mark group as seated.
            found[n] = 1
            # Sample service time.
            service_time = self.service_time_model.random(
                1 / service_time_means[group_size - 1]
            )
            # Update table availability.
            table_avail[k, j] += service_time
            # Update revenue.
            total_rev += table_revenue[group_size - 1]
        # Calculate responses from simulation data.
        responses = {
            "total_revenue": total_rev,
            "service_rate": sum(found) / len(found),
        }
        return responses, {}


class TableAllocationMaxRev(Problem):
    """Class to make table allocation simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "TABLEALLOCATION-1"
    class_name: ClassVar[str] = "Max Revenue for Restaurant Table Allocation"
    config_class: ClassVar[type[BaseModel]] = TableAllocationMaxRevConfig
    model_class: ClassVar[type[Model]] = TableAllocation
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"num_tables"}

    @property
    def dim(self) -> int:  # noqa: D102
        return 4

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"num_tables": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (factor_dict["num_tables"],)

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["total_revenue"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return (
            np.sum(np.multiply(self.model.factors["table_cap"], x))
            <= self.model.factors["capacity"]
        )

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        # Add new tables of random size to the restaurant until the capacity is reached.
        # TODO: Replace this with call to integer_random_vector_from_simplex().
        # The different-weight case is not yet implemented.
        allocated = 0
        num_tables = [0, 0, 0, 0]
        while allocated < self.model.factors["capacity"]:
            table = rand_sol_rng.randint(0, len(self.model.factors["table_cap"]) - 1)
            if self.model.factors["table_cap"][table] <= (
                self.model.factors["capacity"] - allocated
            ):
                num_tables[table] += 1
                allocated += self.model.factors["table_cap"][table]
            elif self.model.factors["table_cap"][0] > (
                self.model.factors["capacity"] - allocated
            ):
                break
        return tuple(num_tables)
