"""Simulate sales for a newsvendor under dynamic consumer substitution."""

from __future__ import annotations

from typing import Annotated, ClassVar, Final

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
from simopt.simulations.dynamnews import DynamNewsConfig, Utility, replicate
from simopt.utils import override

NUM_PRODUCTS: Final[int] = 10


class DynamNewsMaxProfitConfig(BaseModel):
    """Configuration model for Dynamic Newsvendor Max Profit Problem.

    A problem configuration that maximizes profit for a dynamic newsvendor
    with consumer substitution by optimizing initial inventory levels.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default_factory=lambda: (3,) * NUM_PRODUCTS,
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


class DynamNews(Model[DynamNewsConfig]):
    """Dynamic Newsvendor Model.

    A model that simulates a day's worth of sales for a newsvendor
    with dynamic consumer substitution. Returns the profit and the
    number of products that stock out.
    """

    class_name_abbr: ClassVar[str] = "DYNAMNEWS"
    class_name: ClassVar[str] = "Dynamic Newsvendor"
    config_class: ClassVar[type[DynamNewsConfig]] = DynamNewsConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 4

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.utility_model = Utility()

    def replicate(self, factors: DynamNewsConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "profit": Profit in this scenario.
                    - "n_prod_stockout": Number of products that are out of stock.
                    - "n_missed_orders": Number of unmet customer orders.
                    - "fill_rate": Fraction of customer orders fulfilled.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        profit, n_prod_stockout, n_missed_orders, fill_rate = replicate(
            factors, rngs, self.utility_model
        )
        # Compose responses and gradients.
        responses = {
            "profit": profit,
            "n_prod_stockout": n_prod_stockout,
            "n_missed_orders": n_missed_orders,
            "fill_rate": fill_rate,
        }
        return responses, {}


class DynamNewsMaxProfit(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "DYNAMNEWS-1"
    class_name: ClassVar[str] = "Max Profit for Dynamic Newsvendor"
    config_class: ClassVar[type[BaseModel]] = DynamNewsMaxProfitConfig
    model_class: ClassVar[type[Model]] = DynamNews
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"init_level"}

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        init_level = problem.add_continuous_vector(
            lb=0.0,
            ub=np.inf,
            shape=(self.model.factors["num_prod"],),
            initial=self.factors["initial_solution"],
        )
        simulation = self.add_simulation(problem, {"init_level": init_level})
        problem.maximize(dsl.mean(simulation.metric("profit")))
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"init_level": vector[:]}

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(x[j] > 0 for j in range(self.dim))

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple([rand_sol_rng.uniform(0, 10) for _ in range(self.dim)])
