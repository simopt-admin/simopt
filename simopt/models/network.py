"""
Summary
-------
Simulate messages being processed in a queueing network.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/network.html>`__.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType

NUM_NETWORKS: Final = 10


class Network(Model):
    """
    Simulate messages being processed in a queueing network.

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

    See also
    --------
    base.Model
    """

    @property
    def name(self) -> str:
        return "NETWORK"

    @property
    def n_rngs(self) -> int:
        return 3

    @property
    def n_responses(self) -> int:
        return 1

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "process_prob": {
                "description": "probability that a message will go through a particular network i",
                "datatype": list,
                "default": [0.1] * NUM_NETWORKS,
            },
            "cost_process": {
                "description": "message processing cost of network i",
                "datatype": list,
                "default": [0.1 / (x + 1) for x in range(NUM_NETWORKS)],
            },
            "cost_time": {
                "description": "cost for the length of time a message spends in a network i per each unit of time",
                "datatype": list,
                "default": [0.005] * NUM_NETWORKS,
            },
            "mode_transit_time": {
                "description": "mode time of transit for network i following a triangular distribution",
                "datatype": list,
                "default": [x + 1 for x in range(NUM_NETWORKS)],
            },
            "lower_limits_transit_time": {
                "description": "lower limits for the triangular distribution for the transit time",
                "datatype": list,
                "default": [0.5 + x for x in range(NUM_NETWORKS)],
            },
            "upper_limits_transit_time": {
                "description": "upper limits for the triangular distribution for the transit time",
                "datatype": list,
                "default": [1.5 + x for x in range(NUM_NETWORKS)],
            },
            "arrival_rate": {
                "description": "arrival rate of messages following a Poisson process",
                "datatype": float,
                "default": 1.0,
            },
            "n_messages": {
                "description": "number of messages that arrives and needs to be routed",
                "datatype": int,
                "default": 1000,
            },
            "n_networks": {
                "description": "number of networks",
                "datatype": int,
                "default": NUM_NETWORKS,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "process_prob": self.check_process_prob,
            "cost_process": self.check_cost_process,
            "cost_time": self.check_cost_time,
            "mode_transit_time": self.check_mode_transit_time,
            "lower_limits_transit_time": self.check_lower_limits_transit_time,
            "upper_limits_transit_time": self.check_upper_limits_transit_time,
            "arrival_rate": self.check_arrival_rate,
            "n_messages": self.check_n_messages,
            "n_networks": self.check_n_networks,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_process_prob(self) -> None:
        # Make sure probabilities are between 0 and 1.
        # Make sure probabilities sum up to 1.
        if (
            any(
                [
                    1.0 < prob_i or prob_i < 0
                    for prob_i in self.factors["process_prob"]
                ]
            )
            or round(sum(self.factors["process_prob"]), 10) != 1.0
        ):
            raise ValueError(
                "All elements in process_prob must be between 0 and 1 and the sum of all of the elements in process_prob must equal 1."
            )

    def check_cost_process(self) -> None:
        if any(cost_i <= 0 for cost_i in self.factors["cost_process"]):
            raise ValueError(
                "All elements in cost_process must be greater than 0."
            )

    def check_cost_time(self) -> None:
        if any(cost_time_i <= 0 for cost_time_i in self.factors["cost_time"]):
            raise ValueError(
                "All elements in cost_time must be greater than 0."
            )

    def check_mode_transit_time(self) -> None:
        if any(
            transit_time_i <= 0
            for transit_time_i in self.factors["mode_transit_time"]
        ):
            raise ValueError(
                "All elements in mode_transit_time must be greater than 0."
            )

    def check_lower_limits_transit_time(self) -> None:
        if any(
            lower_i <= 0
            for lower_i in self.factors["lower_limits_transit_time"]
        ):
            raise ValueError(
                "All elements in lower_limits_transit_time must be greater than 0."
            )

    def check_upper_limits_transit_time(self) -> None:
        if any(
            upper_i <= 0
            for upper_i in self.factors["upper_limits_transit_time"]
        ):
            raise ValueError(
                "All elements in upper_limits_transit_time must be greater than 0."
            )

    def check_arrival_rate(self) -> None:
        if self.factors["arrival_rate"] <= 0:
            raise ValueError("arrival_rate must be greater than 0.")

    def check_n_messages(self) -> None:
        if self.factors["n_messages"] <= 0:
            raise ValueError("n_messages must be greater than 0.")

    def check_n_networks(self) -> None:
        if self.factors["n_networks"] <= 0:
            raise ValueError("n_networks must be greater than 0.")

    def check_simulatable_factors(self) -> bool:
        if len(self.factors["process_prob"]) != self.factors["n_networks"]:
            raise ValueError(
                "The length of process_prob must equal n_networks."
            )
        elif len(self.factors["cost_process"]) != self.factors["n_networks"]:
            raise ValueError(
                "The length of cost_process must equal n_networks."
            )
        elif len(self.factors["cost_time"]) != self.factors["n_networks"]:
            raise ValueError("The length of cost_time must equal n_networks.")
        elif (
            len(self.factors["mode_transit_time"]) != self.factors["n_networks"]
        ):
            raise ValueError(
                "The length of mode_transit_time must equal n_networks."
            )
        elif (
            len(self.factors["lower_limits_transit_time"])
            != self.factors["n_networks"]
        ):
            raise ValueError(
                "The length of lower_limits_transit_time must equal n_networks."
            )
        elif (
            len(self.factors["upper_limits_transit_time"])
            != self.factors["n_networks"]
        ):
            raise ValueError(
                "The length of upper_limits_transit_time must equal n_networks."
            )
        elif any(
            [
                self.factors["mode_transit_time"][i]
                < self.factors["lower_limits_transit_time"][i]
                for i in range(self.factors["n_networks"])
            ]
        ):
            raise ValueError(
                "The mode_transit time must be greater than or equal to the corresponding lower_limits_transit_time for each network."
            )
        elif any(
            [
                self.factors["upper_limits_transit_time"][i]
                < self.factors["mode_transit_time"][i]
                for i in range(self.factors["n_networks"])
            ]
        ):
            raise ValueError(
                "The mode_transit time must be less than or equal to the corresponding upper_limits_transit_time for each network."
            )
        else:
            return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measure of interest
            "total_cost": total cost spent to route all messages
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Determine total number of arrivals to simulate.
        total_arrivals = self.factors["n_messages"]
        # Designate separate random number generators.
        arrival_rng = rng_list[0]
        network_rng = rng_list[1]
        transit_rng = rng_list[2]
        # Generate all interarrival, network routes, and service times before the simulation run.
        arrival_times = [
            arrival_rng.expovariate(self.factors["arrival_rate"])
            for _ in range(total_arrivals)
        ]
        network_routes = network_rng.choices(
            range(self.factors["n_networks"]),
            weights=self.factors["process_prob"],
            k=total_arrivals,
        )
        service_times = [
            transit_rng.triangular(
                low=self.factors["lower_limits_transit_time"][
                    network_routes[i]
                ],
                high=self.factors["upper_limits_transit_time"][
                    network_routes[i]
                ],
                mode=self.factors["mode_transit_time"][network_routes[i]],
            )
            for i in range(total_arrivals)
        ]
        # Create matrix storing times and metrics for each message:
        #     column 0 : arrival time to queue;
        #     column 1 : network route;
        #     column 2 : service time;
        #     column 3 : service completion time;
        #     column 4 : sojourn time;
        #     column 5 : waiting time;
        #     column 6 : processing cost;
        #     column 7 : time cost;
        #     column 8 : total cost;
        message_mat = np.zeros((total_arrivals, 9))
        message_mat[:, 0] = np.cumsum(arrival_times)
        message_mat[:, 1] = network_routes
        message_mat[:, 2] = service_times
        # Fill in entries for remaining messages' metrics.
        # Create a list recording the index of the last customer sent to each network.
        # Starting with -1, indicating no one is in line.
        last_in_line = [-1 for _ in range(self.factors["n_networks"])]
        # Start of the simulation run.
        for i in range(total_arrivals):
            # Identify the network that message i was routed to.
            network = int(message_mat[i, 1])
            # Check if there was a message in line previously sent to that network.
            if last_in_line[network] == -1:
                # With no message in line, message i's service completion time is its arrival time plus its service time.
                message_mat[i, 3] = message_mat[i, 0] + message_mat[i, 2]
            else:
                # With a message in line, message i's service completion time will be calculated using Lindley's recursion method.
                # We first choose the maximum between the arrival time of message i and the last in line message's service completion time.
                # We then add message's i service time to this value.
                message_mat[i, 3] = (
                    max(
                        message_mat[i, 0], message_mat[last_in_line[network], 3]
                    )
                    + message_mat[i, 2]
                )
            # Calculate other statistics.
            message_mat[i, 4] = message_mat[i, 3] - message_mat[i, 0]
            message_mat[i, 5] = message_mat[i, 4] - message_mat[i, 2]
            message_mat[i, 6] = self.factors["cost_process"][network]
            message_mat[i, 7] = (
                self.factors["cost_time"][network] * message_mat[i, 4]
            )
            message_mat[i, 8] = message_mat[i, 6] + message_mat[i, 7]
            last_in_line[network] = i
        # Compute total costs for the simulation run.
        total_cost = sum(message_mat[:, 8])
        responses = {"total_cost": total_cost}
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
Minimize the expected total cost routing the messages though the communication network.
"""


class NetworkMinTotalCost(Problem):
    """
    Base class to implement simulation-optimization problems.

    Attributes
    ----------
    name : string
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
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
    model : Model object
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed_factors : dict
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
        return ConstraintType.DETERMINISTIC

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
        return {}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"process_prob"}

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (0.1,) * NUM_NETWORKS,
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
        return self.model.factors["n_networks"]

    @property
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (1,) * self.dim

    def __init__(
        self,
        name: str = "NETWORK-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=Network,
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
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        factor_dict = {"process_prob": vector[:]}
        return factor_dict

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = tuple(factor_dict["process_prob"])
        return vector

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (response_dict["total_cost"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

        Arguments
        ---------
        response_dict : dictionary
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
        det_objectives_gradients = (0,) * self.model.factors["n_networks"]
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
            vector of deterministic components of stochastic
            constraints
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
        # Check box constraints.
        box_feasible = super().check_deterministic_constraints(x)
        # Check constraint that probabilities sum to one.
        probability_feasible = round(sum(x), 10) == 1.0
        return box_feasible and probability_feasible

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        # Generating a random pmf with length equal to number of networks.
        x = rand_sol_rng.continuous_random_vector_from_simplex(
            n_elements=self.model.factors["n_networks"],
            summation=1.0,
            exact_sum=True,
        )
        return tuple(x)
