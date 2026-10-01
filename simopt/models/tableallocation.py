"""Simulate multiple periods of arrival and seating at a restaurant."""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt import dsl
from simopt.base import (
    ConstraintType,
    Model,
    Problem,
    VariableType,
)
from simopt.input_models import Exp, Poisson, Uniform, WeightedChoice
from simopt.simulations.tableallocation import TableAllocationConfig, replicate
from simopt.utils import override


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


class TableAllocation(Model[TableAllocationConfig]):
    """Table Allocation Model.

    A model that simulates a table capacity allocation problem at a restaurant
    with a homogenous Poisson arrvial process and exponential service times.
    Returns expected maximum revenue.
    """

    class_name_abbr: ClassVar[str] = "TABLEALLOCATION"
    class_name: ClassVar[str] = "Restaurant Table Allocation"
    config_class: ClassVar[type[TableAllocationConfig]] = TableAllocationConfig
    n_rngs: ClassVar[int] = 4
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

    def replicate(self, factors: TableAllocationConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "total_revenue": Total revenue earned over the simulation period.
                    - "service_rate": Fraction of customer arrivals that are seated.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """

        total_rev, service_rate = replicate(
            factors,
            rngs,
            self.arrival_time_model,
            self.arrival_number_model,
            self.group_size_model,
            self.service_time_model,
        )
        responses = {
            "total_revenue": total_rev,
            "service_rate": service_rate,
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

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        num_tables = problem.add_integer_vector(
            lb=0,
            ub=np.inf,
            shape=(len(self.model.factors["table_cap"]),),
            initial=tuple(self.factors["initial_solution"]),
        )
        allocated_capacity = sum(
            table_capacity * table_count
            for table_capacity, table_count in zip(
                self.model.factors["table_cap"], num_tables, strict=True
            )
        )
        problem.add_linear_constraint(allocated_capacity <= self.model.factors["capacity"])
        simulation = self.add_simulation(problem, {"num_tables": num_tables})
        problem.maximize(dsl.mean(simulation.metric("total_revenue")))
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"num_tables": vector[:]}

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return (
            np.sum(np.multiply(self.model.factors["table_cap"], x))
            <= self.model.factors["capacity"]
        )

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
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
            elif self.model.factors["table_cap"][0] > (self.model.factors["capacity"] - allocated):
                break
        return tuple(num_tables)
