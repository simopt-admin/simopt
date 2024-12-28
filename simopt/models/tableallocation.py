"""
Summary
-------
Simulate multiple periods of arrival and seating at a restaurant.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/tableallocation.html>`__.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType


class TableAllocation(Model):
    """
    A model that simulates a table capacity allocation problem at a restaurant
    with a homogenous Poisson arrvial process and exponential service times.
    Returns expected maximum revenue.

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

        ``n_hours``
            Number of hours to simulate (`int`)
        ``capacity``
            Maximum total capacity (`int`)
        ``table_cap``
            Capacity of each type of table (`int`)
        ``lambda``
            Average number of arrivals per hour (`flt`)
        ``service_time_means``
            Mean service time in minutes (`flt`)
        ``table_revenue``
            Per table revenue earned (`flt`)
        ``num_tables``
            Number of tables of each capacity (`int`)

    See also
    --------
    base.Model
    """

    @property
    def name(self) -> str:
        return "TABLEALLOCATION"

    @property
    def n_rngs(self) -> int:
        return 3

    @property
    def n_responses(self) -> int:
        return 2

    @property
    def specifications(self) -> dict[str, dict]:
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "n_hours": self.check_n_hours,
            "capacity": self.check_capacity,
            "table_cap": self.check_table_cap,
            "lambda": self.check_lambda,
            "service_time_means": self.check_service_time_means,
            "table_revenue": self.check_table_revenue,
            "num_tables": self.check_num_tables,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_n_hours(self) -> None:
        if self.factors["n_hours"] <= 0:
            raise ValueError("n_hours must be greater than 0.")

    def check_capacity(self) -> None:
        if self.factors["capacity"] <= 0:
            raise ValueError("capacity must be greater than 0.")

    def check_table_cap(self) -> None:
        if self.factors["table_cap"] <= [0, 0, 0, 0]:
            raise ValueError(
                "All elements in table_cap must be greater than 0."
            )

    def check_lambda(self) -> bool:
        return self.factors["lambda"] >= [0] * max(self.factors["table_cap"])

    def check_service_time_means(self) -> bool:
        return self.factors["service_time_means"] > [0] * max(
            self.factors["table_cap"]
        )

    def check_table_revenue(self) -> bool:
        return self.factors["table_revenue"] >= [0] * max(
            self.factors["table_cap"]
        )

    def check_num_tables(self) -> None:
        if self.factors["num_tables"] < [0, 0, 0, 0]:
            raise ValueError(
                "Each element in num_tables must be greater than or equal to 0."
            )

    def check_simulatable_factors(self) -> bool:
        if len(self.factors["num_tables"]) != len(self.factors["table_cap"]):
            raise ValueError(
                "The length of num_tables must be equal to the length of table_cap."
            )
        elif len(self.factors["lambda"]) != max(self.factors["table_cap"]):
            raise ValueError(
                "The length of lamda must be equal to the maximum value in table_cap."
            )
        elif len(self.factors["lambda"]) != len(
            self.factors["service_time_means"]
        ):
            raise ValueError(
                "The length of lambda must be equal to the length of service_time_means."
            )
        elif len(self.factors["service_time_means"]) != len(
            self.factors["table_revenue"]
        ):
            raise ValueError(
                "The length of service_time_means must be equal to the length of table_revenue."
            )
        else:
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

            ``total_revenue``
                Total revenue earned over the simulation period.
            ``service_rate``
                Fraction of customer arrivals that are seated.

        """
        # Designate separate random number generators.
        arrival_rng = rng_list[0]
        group_size_rng = rng_list[1]
        service_rng = rng_list[2]
        # Track total revenue.
        total_rev = 0
        # Track table availability.
        # (i,j) is the time that jth table of size i becomes available.
        table_avail = np.zeros((4, max(self.factors["num_tables"])))
        # Generate total number of arrivals in the period
        n_arrivals = arrival_rng.poissonvariate(
            round(self.factors["n_hours"] * sum(self.factors["lambda"]))
        )
        # Generate arrival times in minutes
        arrival_times = 60 * np.sort(
            [
                arrival_rng.uniform(0, self.factors["n_hours"])
                for _ in range(n_arrivals)
            ]
        )
        # Track seating rate
        found = np.zeros(n_arrivals)
        # Pass through all arrivals of groups to the restaurants.
        for n in range(n_arrivals):
            # Determine group size.
            group_size = group_size_rng.choices(
                population=range(1, max(self.factors["table_cap"]) + 1),
                weights=self.factors["lambda"],
            )[0]
            # Find smallest table size to start search.
            table_size_idx = 0
            while self.factors["table_cap"][table_size_idx] < group_size:
                table_size_idx = table_size_idx + 1
            # Initialize k and j to make sure they're not unbound
            k = 0
            j = 0
            # Find smallest available table.
            for k in range(table_size_idx, len(self.factors["num_tables"])):
                for j in range(self.factors["num_tables"][k]):
                    # Check if table is currently available.
                    if table_avail[k, j] < arrival_times[n]:
                        found[n] = 1
                        break
                if found[n] == 1:
                    break
            if found[n] == 1:
                # Sample service time.
                service_time = service_rng.expovariate(
                    lambd=1 / self.factors["service_time_means"][group_size - 1]
                )
                # Update table availability.
                table_avail[k, j] = table_avail[k, j] + service_time
                # Update revenue.
                total_rev = (
                    total_rev + self.factors["table_revenue"][group_size - 1]
                )
        # Calculate responses from simulation data.
        responses = {
            "total_revenue": total_rev,
            "service_rate": sum(found) / len(found),
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
Maximize the total expected revenue for a restaurant operation.
"""


class TableAllocationMaxRev(Problem):
    """
    Class to make table allocation simulation-optimization problems.

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
        return (1,)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.DETERMINISTIC

    @property
    def variable_type(self) -> VariableType:
        return VariableType.DISCRETE

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
        return {}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"num_tables"}

    @property
    def specifications(self) -> dict[str, dict]:
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @property
    def dim(self) -> int:
        return 4

    @property
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "TABLEALLOCATION-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=TableAllocation,
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
        factor_dict = {"num_tables": vector[:]}
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
        vector = (factor_dict["num_tables"],)
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
        objectives = (response_dict["total_revenue"],)
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
        det_objectives_gradients = ((0,) * self.dim,)
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
        return (
            np.sum(np.multiply(self.model_fixed_factors["table_cap"], x))
            <= self.model_fixed_factors["capacity"]
        )

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
        # Add new tables of random size to the restaurant until the capacity is reached.
        # TO DO: Replace this with call to integer_random_vector_from_simplex().
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
                num_tables[table] = num_tables[table] + 1
                allocated = (
                    allocated + self.model_fixed_factors["table_cap"][table]
                )
            elif self.model_fixed_factors["table_cap"][0] > (
                self.model_fixed_factors["capacity"] - allocated
            ):
                break
        return tuple(num_tables)
