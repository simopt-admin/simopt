"""Simulate multiple periods of arrival and seating at a restaurant."""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Sequence
from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import Exp, Poisson, Uniform, WeightedChoice
from simopt.utils import classproperty, override


class TableAllocation(Model):
    """Table Allocation Model.

    A model that simulates a table capacity allocation problem at a restaurant
    with a homogenous Poisson arrvial process and exponential service times.
    Returns expected maximum revenue.
    """

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Restaurant Table Allocation"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 3

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 2

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "n_hours": {
                "description": "number of hours to simulate",
                "datatype": float,
                "default": 5.0,
            },
            "capacity": {
                "description": "maximum capacity of restaurant",
                "datatype": int,
                "default": 80,
            },
            "table_cap": {
                "description": "seating capacity of each type of table",
                "datatype": list,
                "default": [2, 4, 6, 8],
            },
            "lambda": {
                "description": "average number of arrivals per hour",
                "datatype": list,
                "default": [3, 6, 3, 3, 2, 4 / 3, 6 / 5, 1],
            },
            "service_time_means": {
                "description": "mean service time (in minutes)",
                "datatype": list,
                "default": [20, 25, 30, 35, 40, 45, 50, 60],
            },
            "table_revenue": {
                "description": "revenue earned for each group size",
                "datatype": list,
                "default": [15, 30, 45, 60, 75, 90, 105, 120],
            },
            "num_tables": {
                "description": "number of tables of each capacity",
                "datatype": list,
                "default": [10, 5, 4, 2],
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "n_hours": self._check_n_hours,
            "capacity": self._check_capacity,
            "table_cap": self._check_table_cap,
            "lambda": self._check_lambda,
            "service_time_means": self._check_service_time_means,
            "table_revenue": self._check_table_revenue,
            "num_tables": self._check_num_tables,
        }

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

    # Check for simulatable factors
    def _check_n_hours(self) -> None:
        if self.factors["n_hours"] <= 0:
            raise ValueError("n_hours must be greater than 0.")

    def _check_capacity(self) -> None:
        if self.factors["capacity"] <= 0:
            raise ValueError("capacity must be greater than 0.")

    def _check_table_cap(self) -> None:
        if self.factors["table_cap"] <= [0, 0, 0, 0]:
            raise ValueError("All elements in table_cap must be greater than 0.")

    def _check_lambda(self) -> bool:
        return self.factors["lambda"] >= [0] * max(self.factors["table_cap"])

    def _check_service_time_means(self) -> bool:
        return self.factors["service_time_means"] > [0] * max(self.factors["table_cap"])

    def _check_table_revenue(self) -> bool:
        return self.factors["table_revenue"] >= [0] * max(self.factors["table_cap"])

    def _check_num_tables(self) -> None:
        if self.factors["num_tables"] < [0, 0, 0, 0]:
            raise ValueError(
                "Each element in num_tables must be greater than or equal to 0."
            )

    @override
    def check_simulatable_factors(self) -> bool:
        if len(self.factors["num_tables"]) != len(self.factors["table_cap"]):
            raise ValueError(
                "The length of num_tables must be equal to the length of table_cap."
            )
        if len(self.factors["lambda"]) != max(self.factors["table_cap"]):
            raise ValueError(
                "The length of lamda must be equal to the maximum value in table_cap."
            )
        if len(self.factors["lambda"]) != len(self.factors["service_time_means"]):
            raise ValueError(
                "The length of lambda must be equal to the length of "
                "service_time_means."
            )
        if len(self.factors["service_time_means"]) != len(
            self.factors["table_revenue"]
        ):
            raise ValueError(
                "The length of service_time_means must be equal to the length of "
                "table_revenue."
            )
        return True

    def before_replicate(self, rng_list):
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
        group_size_options = range(1, max_table_cap + 1)
        # Pass through all arrivals of groups to the restaurants.
        for n in range(n_arrivals):
            # Determine group size.
            group_size = self.group_size_model.random(
                population=group_size_options,
                weights=f_lambda,
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
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class TableAllocationMaxRev(Problem):
    """Class to make table allocation simulation-optimization problems."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "TABLEALLOCATION-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Max Revenue for Restaurant Table Allocation"

    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1

    @classproperty
    @override
    def n_stochastic_constraints(cls) -> int:
        return 0

    @classproperty
    @override
    def minmax(cls) -> tuple[int]:
        return (1,)

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.DETERMINISTIC

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.DISCRETE

    @classproperty
    @override
    def gradient_available(cls) -> bool:
        return False

    @classproperty
    @override
    def optimal_value(cls) -> None:
        return None

    @classproperty
    @override
    def optimal_solution(cls) -> None:
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"num_tables"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (10, 5, 4, 2),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
                "isDatafarmable": False,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @classproperty
    @override
    def dim(cls) -> int:
        return 4

    @classproperty
    @override
    def lower_bounds(cls) -> tuple:
        return (0,) * cls.dim

    @classproperty
    @override
    def upper_bounds(cls) -> tuple:
        return (np.inf,) * cls.dim

    def __init__(
        self,
        name: str = "TABLEALLOCATION-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the Table Allocation Problem.

        Args:
            name (str, optional): Name of the problem. Defaults to "TABLEALLOCATION-1".
            fixed_factors (dict, optional): Fixed factors for the problem.
                Defaults to None.
            model_fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=TableAllocation,
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"num_tables": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return (factor_dict["num_tables"],)

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["total_revenue"],)

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim,)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return (
            np.sum(np.multiply(self.model_fixed_factors["table_cap"], x))
            <= self.model_fixed_factors["capacity"]
        )

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # Add new tables of random size to the restaurant until the capacity is reached.
        # TODO: Replace this with call to integer_random_vector_from_simplex().
        # The different-weight case is not yet implemented.
        allocated = 0
        num_tables = [0, 0, 0, 0]
        while allocated < self.model_fixed_factors["capacity"]:
            table = rand_sol_rng.randint(
                0, len(self.model_fixed_factors["table_cap"]) - 1
            )
            if self.model_fixed_factors["table_cap"][table] <= (
                self.model_fixed_factors["capacity"] - allocated
            ):
                num_tables[table] += 1
                allocated += self.model_fixed_factors["table_cap"][table]
            elif self.model_fixed_factors["table_cap"][0] > (
                self.model_fixed_factors["capacity"] - allocated
            ):
                break
        return tuple(num_tables)
