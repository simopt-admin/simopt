"""Simulate messages being processed in a queueing network."""

from __future__ import annotations

from enum import IntEnum
from typing import Annotated, ClassVar, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import Exp, InputModel, Triangular
from simopt.utils import override

NUM_NETWORKS: Final = 10


class NetworkConfig(BaseModel):
    process_prob: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.1] * NUM_NETWORKS,
            description=(
                "probability that a message will go through a particular network i"
            ),
        ),
    ]
    cost_process: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.1 / (x + 1) for x in range(NUM_NETWORKS)],
            description="message processing cost of network i",
        ),
    ]
    cost_time: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.005] * NUM_NETWORKS,
            description=(
                "cost for the length of time a message spends in a network i per "
                "each unit of time"
            ),
        ),
    ]
    mode_transit_time: Annotated[
        list[float],
        Field(
            default_factory=lambda: [x + 1 for x in range(NUM_NETWORKS)],
            description=(
                "mode time of transit for network i following a triangular "
                "distribution"
            ),
        ),
    ]
    lower_limits_transit_time: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.5 + x for x in range(NUM_NETWORKS)],
            description=(
                "lower limits for the triangular distribution for the transit time"
            ),
        ),
    ]
    upper_limits_transit_time: Annotated[
        list[float],
        Field(
            default_factory=lambda: [1.5 + x for x in range(NUM_NETWORKS)],
            description=(
                "upper limits for the triangular distribution for the transit time"
            ),
        ),
    ]
    arrival_rate: Annotated[
        float,
        Field(
            default=1.0,
            description="arrival rate of messages following a Poisson process",
            gt=0,
        ),
    ]
    n_messages: Annotated[
        int,
        Field(
            default=1000,
            description="number of messages that arrives and needs to be routed",
            gt=0,
        ),
    ]
    n_networks: Annotated[
        int,
        Field(
            default=NUM_NETWORKS,
            description="number of networks",
            gt=0,
        ),
    ]

    def _check_process_prob(self) -> None:
        # Make sure probabilities are between 0 and 1.
        # Make sure probabilities sum up to 1.
        if (
            any(prob_i > 1.0 or prob_i < 0 for prob_i in self.process_prob)
            or round(sum(self.process_prob), 10) != 1.0
        ):
            raise ValueError(
                "All elements in process_prob must be between 0 and 1 and the sum of "
                "all of the elements in process_prob must equal 1."
            )

    def _check_cost_process(self) -> None:
        if any(cost_i <= 0 for cost_i in self.cost_process):
            raise ValueError("All elements in cost_process must be greater than 0.")

    def _check_cost_time(self) -> None:
        if any(cost_time_i <= 0 for cost_time_i in self.cost_time):
            raise ValueError("All elements in cost_time must be greater than 0.")

    def _check_mode_transit_time(self) -> None:
        if any(transit_time_i <= 0 for transit_time_i in self.mode_transit_time):
            raise ValueError(
                "All elements in mode_transit_time must be greater than 0."
            )

    def _check_lower_limits_transit_time(self) -> None:
        if any(lower_i <= 0 for lower_i in self.lower_limits_transit_time):
            raise ValueError(
                "All elements in lower_limits_transit_time must be greater than 0."
            )

    def _check_upper_limits_transit_time(self) -> None:
        if any(upper_i <= 0 for upper_i in self.upper_limits_transit_time):
            raise ValueError(
                "All elements in upper_limits_transit_time must be greater than 0."
            )

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_process_prob()
        self._check_cost_process()
        self._check_cost_time()
        self._check_mode_transit_time()
        self._check_lower_limits_transit_time()
        self._check_upper_limits_transit_time()

        if len(self.process_prob) != self.n_networks:
            raise ValueError("The length of process_prob must equal n_networks.")
        if len(self.cost_process) != self.n_networks:
            raise ValueError("The length of cost_process must equal n_networks.")
        if len(self.cost_time) != self.n_networks:
            raise ValueError("The length of cost_time must equal n_networks.")
        if len(self.mode_transit_time) != self.n_networks:
            raise ValueError("The length of mode_transit_time must equal n_networks.")
        if len(self.lower_limits_transit_time) != self.n_networks:
            raise ValueError(
                "The length of lower_limits_transit_time must equal n_networks."
            )
        if len(self.upper_limits_transit_time) != self.n_networks:
            raise ValueError(
                "The length of upper_limits_transit_time must equal n_networks."
            )

        if any(
            self.mode_transit_time[i] < self.lower_limits_transit_time[i]
            for i in range(self.n_networks)
        ):
            raise ValueError(
                "The mode_transit time must be greater than or equal to the "
                "corresponding lower_limits_transit_time for each network."
            )
        if any(
            self.upper_limits_transit_time[i] < self.mode_transit_time[i]
            for i in range(self.n_networks)
        ):
            raise ValueError(
                "The mode_transit time must be less than or equal to the corresponding "
                "upper_limits_transit_time for each network."
            )

        return self


class NetworkMinTotalCostConfig(BaseModel):
    """Configuration model for Network Min Total Cost Problem.

    Min Total Cost for Communication Networks System simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (0.1,) * NUM_NETWORKS,
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


class RouteInputModel(InputModel):
    """Input model for routing choices in the network."""

    def set_rng(self, rng: random.Random) -> None:  # noqa: D102
        self.rng = rng

    def unset_rng(self) -> None:  # noqa: D102
        self.rng = None

    def random(self, choices: list, weights: list, k: int) -> list:  # noqa: D102
        return self.rng.choices(choices, weights, k=k)


class Network(Model):
    """Simulate messages being processed in a queueing network."""

    config_class: ClassVar[type[BaseModel]] = NetworkConfig
    class_name: str = "Communication Networks System"
    n_rngs: int = 3
    n_responses: int = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Network model.

        Args:
            fixed_factors (dict): Fixed factors for the model.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.arrival_model = Exp()
        self.route_model = RouteInputModel()
        self.service_model = Triangular()

    @override
    def check_simulatable_factors(self) -> bool:
        if len(self.factors["process_prob"]) != self.factors["n_networks"]:
            raise ValueError("The length of process_prob must equal n_networks.")
        if len(self.factors["cost_process"]) != self.factors["n_networks"]:
            raise ValueError("The length of cost_process must equal n_networks.")
        if len(self.factors["cost_time"]) != self.factors["n_networks"]:
            raise ValueError("The length of cost_time must equal n_networks.")
        if len(self.factors["mode_transit_time"]) != self.factors["n_networks"]:
            raise ValueError("The length of mode_transit_time must equal n_networks.")
        if len(self.factors["lower_limits_transit_time"]) != self.factors["n_networks"]:
            raise ValueError(
                "The length of lower_limits_transit_time must equal n_networks."
            )
        if len(self.factors["upper_limits_transit_time"]) != self.factors["n_networks"]:
            raise ValueError(
                "The length of upper_limits_transit_time must equal n_networks."
            )
        if any(
            self.factors["mode_transit_time"][i]
            < self.factors["lower_limits_transit_time"][i]
            for i in range(self.factors["n_networks"])
        ):
            raise ValueError(
                "The mode_transit time must be greater than or equal to the "
                "corresponding lower_limits_transit_time for each network."
            )
        if any(
            self.factors["upper_limits_transit_time"][i]
            < self.factors["mode_transit_time"][i]
            for i in range(self.factors["n_networks"])
        ):
            raise ValueError(
                "The mode_transit time must be less than or equal to the corresponding "
                "upper_limits_transit_time for each network."
            )
        return True

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.arrival_model.set_rng(rng_list[0])
        self.route_model.set_rng(rng_list[1])
        self.service_model.set_rng(rng_list[2])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measure of interest, including:
                    - "total_cost": Total cost spent to route all messages.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        # Determine total number of arrivals to simulate.
        total_arrivals = self.factors["n_messages"]
        arrival_rate = self.factors["arrival_rate"]
        n_networks = self.factors["n_networks"]
        process_prob = self.factors["process_prob"]
        lower_limits_transit_time = self.factors["lower_limits_transit_time"]
        upper_limits_transit_time = self.factors["upper_limits_transit_time"]
        mode_transit_time = self.factors["mode_transit_time"]
        cost_process = self.factors["cost_process"]
        cost_time = self.factors["cost_time"]

        # Generate all interarrival, network routes, and service times before the
        # simulation run.
        arrival_times = [
            self.arrival_model.random(arrival_rate) for _ in range(total_arrivals)
        ]
        network_routes = self.route_model.random(
            range(n_networks),
            weights=process_prob,
            k=total_arrivals,
        )
        service_times = [
            self.service_model.random(
                low=lower_limits_transit_time[route],
                high=upper_limits_transit_time[route],
                mode=mode_transit_time[route],
            )
            for route in network_routes
        ]

        # Alias columns by index
        class Col(IntEnum):
            ARR = 0  # arrival time to queue
            ROUTE = 1  # network route
            SVC = 2  # service time
            DONE = 3  # service completion time
            SOJ = 4  # sojourn time
            WAIT = 5  # waiting time
            PROC_COST = 6  # processing cost
            TIME_COST = 7  # time cost
            TOTAL_COST = 8  # total cost

        message_mat = np.zeros((total_arrivals, 9))
        message_mat[:, Col.ARR] = np.cumsum(arrival_times)
        message_mat[:, Col.ROUTE] = network_routes
        message_mat[:, Col.SVC] = service_times
        # Fill in entries for remaining messages' metrics.
        # Create a list recording the index of the last customer sent to each network.
        # Starting with -1, indicating no one is in line.
        routes = message_mat[:, Col.ROUTE].astype(int)
        arrival = message_mat[:, Col.ARR]
        service = message_mat[:, Col.SVC]

        # Initialize completion time tracking per network
        last_in_line = [-1] * n_networks

        for i in range(total_arrivals):
            net = routes[i]
            arr_i = arrival[i]
            svc_i = service[i]

            if last_in_line[net] == -1:
                done = arr_i + svc_i
            else:
                done = max(arr_i, message_mat[last_in_line[net], Col.DONE]) + svc_i

            curr_message = message_mat[i]
            curr_message[Col.DONE] = done
            curr_message[Col.SOJ] = done - arr_i
            curr_message[Col.WAIT] = curr_message[Col.SOJ] - svc_i
            last_in_line[net] = i

        # Vectorized cost computations after SOJ is known
        message_mat[:, Col.PROC_COST] = np.array(cost_process)[routes]
        message_mat[:, Col.TIME_COST] = (
            np.array(cost_time)[routes] * message_mat[:, Col.SOJ]
        )
        message_mat[:, Col.TOTAL_COST] = (
            message_mat[:, Col.PROC_COST] + message_mat[:, Col.TIME_COST]
        )

        routes = message_mat[:, Col.ROUTE].astype(int)
        message_mat[:, Col.PROC_COST] = np.array(cost_process)[routes]
        message_mat[:, Col.TIME_COST] = (
            np.array(cost_time)[routes] * message_mat[:, Col.SOJ]
        )
        message_mat[:, Col.TOTAL_COST] = (
            message_mat[:, Col.PROC_COST] + message_mat[:, Col.TIME_COST]
        )
        # Compute total costs for the simulation run.
        total_cost = np.sum(message_mat[:, Col.TOTAL_COST])
        responses = {"total_cost": total_cost}
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class NetworkMinTotalCost(Problem):
    """Base class to implement simulation-optimization problems."""

    config_class: ClassVar[type[BaseModel]] = NetworkMinTotalCostConfig
    model_class: ClassVar[type[Model]] = Network
    class_name_abbr: str = "NETWORK-1"
    class_name: str = "Min Total Cost for Communication Networks System"
    n_objectives: int = 1
    n_stochastic_constraints: int = 0
    minmax: tuple[int] = (-1,)
    constraint_type: ConstraintType = ConstraintType.DETERMINISTIC
    variable_type: VariableType = VariableType.CONTINUOUS
    gradient_available: bool = False
    optimal_value: float | None = None
    optimal_solution: tuple | None = None
    model_default_factors: dict = {}
    model_decision_factors: set[str] = {"process_prob"}

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["n_networks"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (1,) * self.dim

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"process_prob": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["process_prob"])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["total_cost"],)

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (0,)
        det_objectives_gradients = (0,) * self.model.factors["n_networks"]
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        # Check box constraints.
        box_feasible = super().check_deterministic_constraints(x)
        if not box_feasible:
            return False

        # Check constraint that probabilities sum to one.
        return round(sum(x), 10) == 1.0

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # Generating a random pmf with length equal to number of networks.
        x = rand_sol_rng.continuous_random_vector_from_simplex(
            n_elements=self.model.factors["n_networks"],
            summation=1.0,
            exact_sum=True,
        )
        return tuple(x)
