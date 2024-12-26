"""
Summary
-------
Simulate multiple periods worth of sales for a (s,S) inventory problem
with continuous inventory.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/sscont.html>`__.
"""

from __future__ import annotations

from math import sqrt
from typing import Callable

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType


class SSCont(Model):
    """
    A model that simulates multiple periods' worth of sales for a (s,S)
    inventory problem with continuous inventory, exponentially distributed
    demand, and poisson distributed lead time. Returns the various types of
    average costs per period, order rate, stockout rate, fraction of demand
    met with inventory on hand, average amount backordered given a stockout
    occured, and average amount ordered given an order occured.

    Attributes
    ----------
    name : str
        name of model
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    check_factor_list : dict
        switch case for checking factor simulatability

    Parameters
    ----------
    fixed_factors : dict
        fixed_factors of the simulation model

        ``demand_mean``
            Mean of exponentially distributed demand in each period (`flt`)
        ``lead_mean``
            Mean of Poisson distributed order lead time (`flt`)
        ``backorder_cost``
            Cost per unit of demand not met with in-stock inventory (`flt`)
        ``holding_cost``
            Holding cost per unit per period (`flt`)
        ``fixed_cost``
            Order fixed cost (`flt`)
        ``variable_cost``
            Order variable cost per unit (`flt`)
        ``s``
            Inventory position threshold for placing order (`flt`)
        ``S``
            Max inventory position (`flt`)
        ``n_days``
            Number of periods to simulate (`int`)
        ``warmup``
            Number of periods as warmup before collecting statistics (`int`)
    See also
    --------
    base.Model
    """

    @property
    def name(self) -> str:
        return "SSCONT"

    @property
    def n_rngs(self) -> int:
        return 2

    @property
    def n_responses(self) -> int:
        return 7

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "demand_mean": {
                "description": "mean of exponentially distributed demand in each period",
                "datatype": float,
                "default": 100.0,
            },
            "lead_mean": {
                "description": "mean of Poisson distributed order lead time",
                "datatype": float,
                "default": 6.0,
            },
            "backorder_cost": {
                "description": "cost per unit of demand not met with in-stock inventory",
                "datatype": float,
                "default": 4.0,
            },
            "holding_cost": {
                "description": "holding cost per unit per period",
                "datatype": float,
                "default": 1.0,
            },
            "fixed_cost": {
                "description": "order fixed cost",
                "datatype": float,
                "default": 36.0,
            },
            "variable_cost": {
                "description": "order variable cost per unit",
                "datatype": float,
                "default": 2.0,
            },
            "s": {
                "description": "inventory threshold for placing order",
                "datatype": float,
                "default": 1000.0,
            },
            "S": {
                "description": "max inventory",
                "datatype": float,
                "default": 2000.0,
            },
            "n_days": {
                "description": "number of periods to simulate",
                "datatype": int,
                "default": 100,
                "isDatafarmable": False,
            },
            "warmup": {
                "description": "number of periods as warmup before collecting statistics",
                "datatype": int,
                "default": 20,
                "isDatafarmable": False,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "demand_mean": self.check_demand_mean,
            "lead_mean": self.check_lead_mean,
            "backorder_cost": self.check_backorder_cost,
            "holding_cost": self.check_holding_cost,
            "fixed_cost": self.check_fixed_cost,
            "variable_cost": self.check_variable_cost,
            "s": self.check_s,
            "S": self.check_S,
            "n_days": self.check_n_days,
            "warmup": self.check_warmup,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_demand_mean(self) -> None:
        if self.factors["demand_mean"] <= 0:
            raise ValueError("demand_mean must be greater than 0.")

    def check_lead_mean(self) -> None:
        if self.factors["lead_mean"] <= 0:
            raise ValueError("lead_mean must be greater than 0.")

    def check_backorder_cost(self) -> None:
        if self.factors["backorder_cost"] <= 0:
            raise ValueError("backorder_cost must be greater than 0.")

    def check_holding_cost(self) -> None:
        if self.factors["holding_cost"] <= 0:
            raise ValueError("holding_cost must be greater than 0.")

    def check_fixed_cost(self) -> None:
        if self.factors["fixed_cost"] <= 0:
            raise ValueError("fixed_cost must be greater than 0.")

    def check_variable_cost(self) -> None:
        if self.factors["variable_cost"] <= 0:
            raise ValueError("variable_cost must be greater than 0.")

    def check_s(self) -> None:
        if self.factors["s"] <= 0:
            raise ValueError("s must be greater than 0.")

    def check_S(self) -> None:  # noqa: N802
        if self.factors["S"] <= 0:
            raise ValueError("S must be greater than 0.")

    def check_n_days(self) -> None:
        if self.factors["n_days"] < 1:
            raise ValueError("n_days must be greater than or equal to 1.")

    def check_warmup(self) -> None:
        if self.factors["warmup"] < 0:
            raise ValueError("warmup must be greater than or equal to 0.")

    def check_simulatable_factors(self) -> bool:
        if self.factors["s"] >= self.factors["S"]:
            raise ValueError("s must be less than S.")
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [mrg32k3a.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest

            ``avg_backorder_costs``
                average backorder costs per period
            ``avg_order_costs``
                average order costs per period
            ``avg_holding_costs``
                average holding costs per period
            ``on_time_rate``
                fraction of demand met with stock on hand in store
            ``order_rate``
                fraction of periods an order was made
            ``stockout_rate``
                fraction of periods a stockout occured
            ``avg_stockout``
                mean amount of product backordered given a stockout occured
            ``avg_order``
                mean amount of product ordered given an order occured
        """
        # Designate random number generators.
        demand_rng = rng_list[0]
        lead_rng = rng_list[1]
        # Generate exponential random demands.
        demands = [
            demand_rng.expovariate(1 / self.factors["demand_mean"])
            for _ in range(self.factors["n_days"] + self.factors["warmup"])
        ]
        # Initialize starting and ending inventories for each period.
        start_inv = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        start_inv[0] = self.factors["s"]  # Start with s units at period 0.
        end_inv = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        # Initialize other quantities to track:
        #   - Amount of product to be received in each period.
        #   - Inventory position each period.
        #   - Amount of product ordered in each period.
        #   - Amount of product outstanding in each period.
        orders_received = np.zeros(
            self.factors["n_days"] + self.factors["warmup"]
        )
        inv_pos = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        orders_placed = np.zeros(
            self.factors["n_days"] + self.factors["warmup"]
        )
        orders_outstanding = np.zeros(
            self.factors["n_days"] + self.factors["warmup"]
        )
        # Run simulation over time horizon.
        for day in range(self.factors["n_days"] + self.factors["warmup"]):
            # Calculate end-of-period inventory on hand and inventory position.
            end_inv[day] = start_inv[day] - demands[day]
            inv_pos[day] = end_inv[day] + orders_outstanding[day]
            # Place orders, keeping track of outstanding orders and when they will be received.
            orders_placed[day] = np.max(
                (
                    (inv_pos[day] < self.factors["s"])
                    * (self.factors["S"] - inv_pos[day])
                ),
                0,
            )
            if orders_placed[day] > 0:
                lead = lead_rng.poissonvariate(self.factors["lead_mean"])
                for future_day in range(day + 1, day + lead + 1):
                    if (
                        future_day
                        < self.factors["n_days"] + self.factors["warmup"]
                    ):
                        orders_outstanding[future_day] = (
                            orders_outstanding[future_day] + orders_placed[day]
                        )
                if (
                    day + lead + 1
                    < self.factors["n_days"] + self.factors["warmup"]
                ):
                    orders_received[day + lead + 1] = (
                        orders_received[day + lead + 1] + orders_placed[day]
                    )
            # Calculate starting inventory for next period.
            if day < self.factors["n_days"] + self.factors["warmup"] - 1:
                start_inv[day + 1] = end_inv[day] + orders_received[day + 1]
        # Calculate responses from simulation data.
        order_rate = np.mean(orders_placed[self.factors["warmup"] :] > 0)
        stockout_rate = np.mean(end_inv[self.factors["warmup"] :] < 0)
        avg_order_costs = np.mean(
            self.factors["fixed_cost"]
            * (orders_placed[self.factors["warmup"] :] > 0)
            + self.factors["variable_cost"]
            * orders_placed[self.factors["warmup"] :]
        )
        avg_holding_costs = np.mean(
            self.factors["holding_cost"]
            * end_inv[self.factors["warmup"] :]
            * [end_inv[self.factors["warmup"] :] > 0]
        )
        on_time_rate = 1 - np.sum(
            np.min(
                np.vstack(
                    (
                        demands[self.factors["warmup"] :],
                        demands[self.factors["warmup"] :]
                        - start_inv[self.factors["warmup"] :],
                    )
                ),
                axis=0,
            )
            * (
                (
                    demands[self.factors["warmup"] :]
                    - start_inv[self.factors["warmup"] :]
                )
                > 0
            )
        ) / np.sum(demands[self.factors["warmup"] :])
        avg_backorder_costs = (
            self.factors["backorder_cost"]
            * (1 - on_time_rate)
            * np.sum(demands[self.factors["warmup"] :])
            / float(self.factors["n_days"])
        )
        if np.array(np.where(end_inv[self.factors["warmup"] :] < 0)).size == 0:
            avg_stockout = 0
        else:
            avg_stockout = -np.mean(
                end_inv[self.factors["warmup"] :][
                    np.where(end_inv[self.factors["warmup"] :] < 0)
                ]
            )
        if (
            np.array(np.where(orders_placed[self.factors["warmup"] :] > 0)).size
            == 0
        ):
            avg_order = 0
        else:
            avg_order = np.mean(
                orders_placed[self.factors["warmup"] :][
                    np.where(orders_placed[self.factors["warmup"] :] > 0)
                ]
            )
        # Compose responses and gradients.
        responses = {
            "avg_backorder_costs": avg_backorder_costs,
            "avg_order_costs": avg_order_costs,
            "avg_holding_costs": avg_holding_costs,
            "on_time_rate": on_time_rate,
            "order_rate": order_rate,
            "stockout_rate": stockout_rate,
            "avg_stockout": avg_stockout,
            "avg_order": avg_order,
        }
        gradients = {
            response_key: {
                factor_key: np.nan for factor_key in self.specifications
            }
            for response_key in responses
        }
        return responses, gradients


"""
Summary
-------
Minimize the expected total cost for (s, S) inventory system.
"""


class SSContMinCost(Problem):
    """
    Class to make (s,S) inventory simulation-optimization problems.

    Attributes
    ----------
    name : str
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : str
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : str
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : tuple
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : base.Model
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : [list]  [mrg32k3a.mrg32k3a.MRG32k3a]
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : tuple
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name of problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """

    @property
    def n_objectives(self) -> int:
        return 1

    @property
    def n_stochastic_constraints(self) -> int:
        return 0

    @property
    def minmax(self) -> tuple[int]:
        return (-1,)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BOX

    @property
    def variable_type(self) -> VariableType:
        return VariableType.CONTINUOUS

    @property
    def gradient_available(self) -> bool:
        return False

    @property
    def optimal_value(self) -> float | None:
        return None

    @property
    def optimal_solution(self) -> tuple | None:
        return None

    @property
    def model_default_factors(self) -> dict:
        return {"demand_mean": 100.0, "lead_mean": 6.0}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"s", "S"}

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution from which solvers start",
                "datatype": tuple,
                "default": (600, 600),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
                "isDatafarmable": False,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @property
    def dim(self) -> int:
        return 2

    @property
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "SSCONT-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=SSCont,
        )

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dict
            dictionary with factor keys and associated values
        """
        factor_dict = {"s": vector[0], "S": vector[0] + vector[1]}
        return factor_dict

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dict
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = (factor_dict["s"], factor_dict["S"] - factor_dict["s"])
        return vector

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dict
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (
            response_dict["avg_backorder_costs"]
            + response_dict["avg_order_costs"]
            + response_dict["avg_holding_costs"],
        )
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

        Arguments
        ---------
        response_dict : dict
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = ()
        return stoch_constraints

    def deterministic_objectives_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
        """
        Compute deterministic components of objectives for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (0,)
        det_objectives_gradients = ((0,),)
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
        """
        Compute deterministic components of stochastic constraints
        for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = ()
        det_stoch_constraints_gradients = ()
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """
        Check if a solution `x` satisfies the problem's deterministic
        constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        return x[0] >= 0 and x[1] >= 0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """

        # x = (rand_sol_rng.expovariate(1/300), rand_sol_rng.expovariate(1/300))
        # x = tuple(sorted([rand_sol_rng.lognormalvariate(600,1),rand_sol_rng.lognormalvariate(600,1)], key = float))
        mu_d = self.model_default_factors["demand_mean"]
        mu_l = self.model_default_factors["lead_mean"]
        x = (
            rand_sol_rng.lognormalvariate(
                mu_d * mu_l / 3, mu_d * mu_l + 2 * sqrt(2 * mu_d**2 * mu_l)
            ),
            rand_sol_rng.lognormalvariate(
                mu_d * mu_l / 3, mu_d * mu_l + 2 * sqrt(2 * mu_d**2 * mu_l)
            ),
        )
        return x


# If T is lead time and X is a single demand, then var(sum_{i=1}^T X_i) = E(T) var(X) + (E X))^2 var T
# var(S) = E var(S|T) + var E(S|T)
