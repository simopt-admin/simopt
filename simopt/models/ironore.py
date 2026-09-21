"""Simulate production and sales over multiple periods for an iron ore inventory."""

# Changed get_random_solution quantiles
#     from 10 and 200 => mean=59.887, sd=53.338, p(X>100)=0.146
#     to 10 and 1000 => mean=199.384, sd=343.925, p(X>100)=0.5

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
from simopt.simulations.ironore import IronOreConfig, MovementInputModel, replicate
from simopt.utils import override


class IronOreMaxRevCntConfig(BaseModel):
    """Configuration model for Iron Ore Max Revenue Continuous Problem.

    Max Revenue for Continuous Iron Ore simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(80, 40, 100),
            description="initial solution",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=1000,
            description="max # of replications for a solver to take",
            gt=0,
        ),
    ]


class IronOreMaxRevConfig(BaseModel):
    """Configuration model for Iron Ore Max Revenue Problem.

    Max Revenue for Iron Ore simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(80, 7000, 40, 100),
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


class IronOre(Model):
    """Iron Ore Inventory Model.

    A model that simulates multiple periods of production and sales for an
    inventory problem with stochastic price determined by a mean-reverting
    random walk. Returns total profit, fraction of days producing iron, and
    mean stock.
    """

    class_name_abbr: ClassVar[str] = "IRONORE"
    class_name: ClassVar[str] = "Iron Ore"
    config_class: ClassVar[type[BaseModel]] = IronOreConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 3

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Iron Ore Inventory Model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.movement_model = MovementInputModel()

    def replicate(self, factors: IronOreConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "total_profit": The total profit over the time period.
                    - "frac_producing": The fraction of days spent producing iron ore.
                    - "mean_stock": The average stock over the time period.
                - gradients (dict): A dictionary of gradient estimates for each
                    response.
        """
        net_profit, frac_producing, mean_stock = replicate(factors, rngs, self.movement_model)
        # Calculate responses from simulation data.
        responses = {
            "total_profit": net_profit,
            "frac_producing": frac_producing,
            "mean_stock": mean_stock,
        }
        return responses, {}


class IronOreMaxRev(Problem):
    """Class to make iron ore inventory simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "IRONORE-1"
    class_name: ClassVar[str] = "Max Revenue for Iron Ore"
    config_class: ClassVar[type[BaseModel]] = IronOreMaxRevConfig
    model_class: ClassVar[type[Model]] = IronOre
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.MIXED
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {
        "price_prod",
        "inven_stop",
        "price_stop",
        "price_sell",
    }

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        initial_solution = self.factors["initial_solution"]
        price_prod = problem.add_continuous_variable(lb=0.0, ub=np.inf, initial=initial_solution[0])
        inven_stop = problem.add_integer_variable(lb=0, ub=np.inf, initial=initial_solution[1])
        price_stop = problem.add_continuous_variable(lb=0.0, ub=np.inf, initial=initial_solution[2])
        price_sell = problem.add_continuous_variable(lb=0.0, ub=np.inf, initial=initial_solution[3])
        simulation = self.add_simulation(
            problem,
            {
                "price_prod": price_prod,
                "inven_stop": inven_stop,
                "price_stop": price_stop,
                "price_sell": price_sell,
            },
        )
        problem.maximize(dsl.mean(simulation.metric("total_profit")))
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {
            "price_prod": vector[0],
            "inven_stop": vector[1],
            "price_stop": vector[2],
            "price_sell": vector[3],
        }

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # return (
        #     rand_sol_rng.randint(70, 90),
        #     rand_sol_rng.randint(2000, 8000),
        #     rand_sol_rng.randint(30, 50),
        #     rand_sol_rng.randint(90, 110),
        # )
        return (
            rand_sol_rng.lognormalvariate(10, 200),
            round(rand_sol_rng.lognormalvariate(1000, 10000)),
            rand_sol_rng.lognormalvariate(10, 200),
            rand_sol_rng.lognormalvariate(10, 200),
        )


class IronOreMaxRevCnt(Problem):
    """Class to make iron ore inventory simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "IRONORECONT-1"
    class_name: ClassVar[str] = "Max Revenue for Continuous Iron Ore"
    config_class: ClassVar[type[BaseModel]] = IronOreMaxRevCntConfig
    model_class: ClassVar[type[Model]] = IronOre
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {
        "price_prod",
        "price_stop",
        "price_sell",
    }

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        initial_solution = self.factors["initial_solution"]
        price_prod = problem.add_continuous_variable(lb=0.0, ub=np.inf, initial=initial_solution[0])
        price_stop = problem.add_continuous_variable(lb=0.0, ub=np.inf, initial=initial_solution[1])
        price_sell = problem.add_continuous_variable(lb=0.0, ub=np.inf, initial=initial_solution[2])
        simulation = self.add_simulation(
            problem,
            {
                "price_prod": price_prod,
                "price_stop": price_stop,
                "price_sell": price_sell,
            },
        )
        problem.maximize(dsl.mean(simulation.metric("total_profit")))
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {
            "price_prod": vector[0],
            "price_stop": vector[1],
            "price_sell": vector[2],
        }

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return x[0] >= 0 and x[1] >= 0 and x[2] >= 0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # return (
        #     rand_sol_rng.randint(70, 90),
        #     rand_sol_rng.randint(30, 50),
        #     rand_sol_rng.randint(90, 110),
        # )
        return (
            rand_sol_rng.lognormalvariate(10, 1000),
            rand_sol_rng.lognormalvariate(10, 1000),
            rand_sol_rng.lognormalvariate(10, 1000),
        )
