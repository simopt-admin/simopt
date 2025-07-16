"""Simulate a single day of operation for an amusement park queuing problem."""

from __future__ import annotations

import math as math
from typing import Callable, Final

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import Exp, Gamma, WeightedChoice
from simopt.utils import classproperty, override

INF = float("inf")

# Default values for the model
PARK_CAPACITY: Final[int] = 350
NUM_ATTRACTIONS: Final[int] = 7


class AmusementPark(Model):
    """Amusement Park Model.

    A model that simulates a single day of operation for an
    amusement park queuing problem based on a poisson distributed tourist
    arrival rate, a next attraction transition matrix, and attraction
    durations based on an Erlang distribution. Returns the total number
    and percent of tourists to leave the park due to full queues.
    """

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 3

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 4

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "park_capacity": {
                "description": (
                    "The total number of tourists waiting for attractions that can be "
                    "maintained through park facilities, distributed across the "
                    "attractions."
                ),
                "datatype": int,
                "default": PARK_CAPACITY,
            },
            "number_attractions": {
                "description": "The number of attractions in the park.",
                "datatype": int,
                "default": NUM_ATTRACTIONS,
                "isDatafarmable": False,
            },
            "time_open": {
                "description": "The number of minutes per day the park is open.",
                "datatype": float,
                "default": 480.0,
            },
            "erlang_shape": {
                "description": (
                    "The shape parameter of the Erlang distribution for each "
                    "attraction duration."
                ),
                "datatype": list,
                "default": [2] * NUM_ATTRACTIONS,
            },
            "erlang_scale": {
                "description": (
                    "The rate parameter of the Erlang distribution for each attraction "
                    "duration."
                ),
                "datatype": list,
                "default": [1 / 9] * NUM_ATTRACTIONS,
            },
            "queue_capacities": {
                "description": (
                    "The capacity of the queue for each attraction based on the "
                    "portion of facilities allocated."
                ),
                "datatype": list,
                "default": [50] * NUM_ATTRACTIONS,
            },
            "depart_probabilities": {
                "description": (
                    "The probability that a tourist will depart the park after "
                    "visiting an attraction."
                ),
                "datatype": list,
                "default": [0.2] * NUM_ATTRACTIONS,
            },
            "arrival_gammas": {
                "description": (
                    "The gamma values for the poisson distributions dictating the "
                    "rates at which tourists entering the park arrive at each "
                    "attraction"
                ),
                "datatype": list,
                "default": [1] * NUM_ATTRACTIONS,
            },
            "transition_probabilities": {
                "description": (
                    "The transition matrix that describes the probability of a tourist "
                    "visiting each attraction after their current attraction."
                ),
                "datatype": list,
                "default": [
                    [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                    [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                    [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                    [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                    [0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.3],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0, 0.3],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
                ],
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        """Switch case for checking factor simulatability."""
        return {
            "park_capacity": self._check_park_capacity,
            "number_attractions": self._check_number_attractions,
            "time_open": self._check_time_open,
            "queue_capacities": self._check_queue_capacities,
            "depart_probabilities": self._check_depart_probabilities,
            "arrival_gammas": self._check_arrival_gammas,
            "transition_probabilities": self._check_transition_probabilities,
            "erlang_shape": self._check_erlang_shape,
            "erlang_scale": self._check_erlang_scale,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Amusement Park Model."""
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.arrival_model = Exp()
        self.attraction_model = WeightedChoice()
        self.destination_model = WeightedChoice()
        self.service_models = []
        for _ in range(self.factors["number_attractions"]):
            self.service_models.append(Gamma())

    # Check for simulatable factors.
    def _check_park_capacity(self) -> None:
        if self.factors["park_capacity"] < 0:
            raise ValueError("Park capacity must be greater than or equal to 0.")

    def _check_number_attractions(self) -> None:
        if self.factors["number_attractions"] < 0:
            raise ValueError("Number of attractions must be greater than 0.")

    def _check_time_open(self) -> None:
        if self.factors["time_open"] < 0:
            raise ValueError("Time open must be greater than or equal to 0.")

    def _check_queue_capacities(self) -> bool:
        return all(cap >= 0 for cap in self.factors["queue_capacities"])

    def _check_depart_probabilities(self) -> bool:
        if (
            len(self.factors["depart_probabilities"])
            != self.factors["number_attractions"]
        ):
            raise ValueError(
                "The number of departure probabilities must match the number of "
                "attractions."
            )
        return all(0 <= prob <= 1 for prob in self.factors["depart_probabilities"])

    def _check_arrival_gammas(self) -> bool:
        if len(self.factors["arrival_gammas"]) != self.factors["number_attractions"]:
            raise ValueError(
                "The number of arrivals must match the number of attractions."
            )
        return all(gamma >= 0 for gamma in self.factors["arrival_gammas"])

    def _check_transition_probabilities(self) -> bool:
        """Validate the structure and consistency of the transition matrix.

        Checks that the transition matrix is square (same number of rows and columns),
        and that the sum of each row and its corresponding departure probability equals
        1.

        Returns:
            bool: True if all checks pass.

        Raises:
            ValueError: If any row has the wrong shape or an invalid total probability.
        """
        transition_sums = list(map(sum, self.factors["transition_probabilities"]))
        if all(
            len(row) == len(self.factors["transition_probabilities"])
            for row in self.factors["transition_probabilities"]
        ) and all(
            transition_sums[i] + self.factors["depart_probabilities"][i] == 1
            for i in range(self.factors["number_attractions"])
        ):
            return True
        raise ValueError(
            "The values you entered are invalid. "
            "Check that each row and depart probability sums to 1."
        )

    def _check_erlang_shape(self) -> bool:
        """Validate the Erlang shape parameters for each attraction.

        Checks that the number of shape parameters matches the number of attractions,
        and that all shape values are non-negative.

        Returns:
            bool: True if all shape parameters are valid.

        Raises:
            ValueError: If the number of shape parameters is incorrect.
        """
        if len(self.factors["erlang_shape"]) != self.factors["number_attractions"]:
            raise ValueError(
                "The number of attractions must equal the number of Erlang shape "
                "parameters."
            )
        return all(gamma >= 0 for gamma in self.factors["erlang_shape"])

    def _check_erlang_scale(self) -> bool:
        """Validate the Erlang scale parameters for each attraction.

        Checks that the number of scale parameters matches the number of attractions,
        and that all scale values are non-negative.

        Returns:
            bool: True if all scale parameters are valid.

        Raises:
            ValueError: If the number of scale parameters is incorrect.
        """
        if len(self.factors["erlang_scale"]) != self.factors["number_attractions"]:
            raise ValueError(
                "The number of attractions must equal the number of Erlang scales."
            )
        return all(gamma >= 0 for gamma in self.factors["erlang_scale"])

    @override
    def check_simulatable_factors(self) -> bool:
        if sum(self.factors["queue_capacities"]) > self.factors["park_capacity"]:
            raise ValueError(
                "The sum of the queue capacities must be less than or equal to the "
                "park capacity"
            )
        return True

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:
        self.arrival_model.set_rng(rng_list[0])
        self.attraction_model.set_rng(rng_list[0])
        self.destination_model.set_rng(rng_list[1])
        for service in self.service_models:
            service.set_rng(rng_list[2])

    def replicate(self) -> tuple[dict[str, float | list[float]], dict]:
        """Simulate a single replication using current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used during the
                simulation.

        Returns:
            tuple: A tuple containing:
                - dict[str, float | list[float]]: Performance metrics from the simulation:
                    - "total_departed_tourists": Total number of tourists who left due to full queues.
                    - "percent_departed_tourists": Percentage of tourists who left due to full queues.
                    - "average_number_in_system": Average number of tourists in the park at a given time.
                    - "attraction_utilization_percentages": Utilization percentage of each attraction.
                - dict: Gradients of the performance measures with respect to model factors.
        """  # noqa: E501

        def set_completion(i: int, new_time: float) -> None:
            """Set the completion time for an attraction.

            Updates the minimum completion time and index if necessary.
            This function doesn't offer much (if any) performance gain with small
            numbers of attractions, but with larger numbers it is significantly faster.

            Args:
                i (int): The index of the attraction.
                new_time (float): The new completion time for the attraction.
            """
            nonlocal min_completion_time, min_completion_index
            completion_times[i] = new_time

            if new_time < min_completion_time:
                min_completion_time = new_time
                min_completion_index = i
            elif i == min_completion_index:
                # Grab the min index and time with one scanning pass
                min_completion_time = min(completion_times)
                min_completion_index = completion_times.index(min_completion_time)

        # Keep local copies of factors to prevent excessive lookups
        num_attactions: int = self.factors["number_attractions"]
        arrival_gammas: list[int] = self.factors["arrival_gammas"]
        time_open: float = self.factors["time_open"]
        erlang_shape: list[int] = self.factors["erlang_shape"]
        erlang_scale: list[float] = self.factors["erlang_scale"]
        queue_capacities: list[int] = self.factors["queue_capacities"]
        transition_probabilities: list[list[float]] = self.factors[
            "transition_probabilities"
        ]
        depart_probabilities: list[float] = self.factors["depart_probabilities"]

        # initialize list of attractions to be selected upon arrival.
        attraction_range = range(num_attactions)
        destination_range = range(num_attactions + 1)
        depart_idx = destination_range[-1]
        # initialize lists of each attraction's next completion time
        completion_times = [INF] * num_attactions
        min_completion_time = INF
        min_completion_index = -1
        # initialize actual queues.
        queues = [0] * num_attactions

        # create external arrival probabilities for each attraction.
        arrival_prob_sum = sum(arrival_gammas)
        arrival_probabalities = [
            arrival_gammas[i] / arrival_prob_sum for i in attraction_range
        ]

        # Initiate clock variables for statistics tracking and event handling.
        clock = 0
        previous_clock = 0
        next_arrival = self.arrival_model.random(arrival_prob_sum)

        # Initialize quantities to track:
        total_visitors = 0
        total_departed = 0
        # initialize time average and utilization quantities.
        in_system = 0
        time_average = 0
        cumulative_util = [0.0] * num_attactions

        # Run simulation over time horizon.
        while clock < time_open:
            # Count number of tourists on attractions and in queues.
            riders = 0
            delta_time = clock - previous_clock
            for i in attraction_range:
                if not math.isinf(completion_times[i]):
                    riders += 1
                    cumulative_util[i] += delta_time
            in_system = sum(queues) + riders
            time_average += in_system * (delta_time)

            previous_clock = clock
            if next_arrival < min_completion_time:
                # Next event is external tourist arrival.
                total_visitors += 1
                # Select attraction.
                attraction_selection = self.attraction_model.random(
                    attraction_range,
                    arrival_probabalities,
                )
                # Check if attraction is currently available.
                # If available, arrive at that attraction. Otherwise check queue.
                if math.isinf(completion_times[attraction_selection]):
                    # Generate completion time if attraction available.
                    completion_time = next_arrival + self.service_models[
                        attraction_selection
                    ].random(
                        alpha=erlang_shape[attraction_selection],
                        beta=erlang_scale[attraction_selection],
                    )
                    set_completion(attraction_selection, completion_time)
                # If unavailable, check if current queue is less than capacity.
                # If queue is not full, join queue.
                elif (
                    queues[attraction_selection]
                    < queue_capacities[attraction_selection]
                ):
                    queues[attraction_selection] += 1
                # If queue is full, leave park + 1.
                else:
                    total_departed += 1
                # Use superposition of Poisson processes to generate next arrival time.
                next_arrival += self.arrival_model.random(arrival_prob_sum)
            else:
                # Next event is the completion of an attraction.
                # Identify finished attraction.
                finished_attraction = completion_times.index(min_completion_time)
                # Check if there is a queue for that attraction.
                # If so then start new completion time and subtract 1 from queue.
                alpha = erlang_shape[finished_attraction]
                beta = erlang_scale[finished_attraction]
                if queues[finished_attraction] > 0:
                    completion_time = min_completion_time + self.service_models[
                        finished_attraction
                    ].random(
                        alpha=alpha,
                        beta=beta,
                    )
                    set_completion(finished_attraction, completion_time)
                    queues[finished_attraction] -= 1
                else:  # If attraction queue is empty, set next completion to infinity.
                    set_completion(finished_attraction, INF)
                # Check if that person will leave the park.
                next_destination = self.destination_model.random(
                    destination_range,
                    transition_probabilities[finished_attraction]
                    + [depart_probabilities[finished_attraction]],
                )

                # Check if tourist leaves park.
                if next_destination != depart_idx:
                    # Check if attraction is currently available.
                    # If available, arrive at that attraction. Otherwise check queue.
                    if math.isinf(completion_times[next_destination]):
                        # Generate completion time if attraction available.
                        completion_time = min_completion_time + self.service_models[
                            next_destination
                        ].random(alpha, beta)
                        set_completion(next_destination, completion_time)
                    # If unavailable, check if current queue is less than capacity.
                    # If queue is not full, join queue.
                    elif queues[next_destination] < queue_capacities[next_destination]:
                        queues[next_destination] += 1
                    # If queue is full, leave park + 1.
                    else:
                        total_departed += 1
            # End of while loop.
            # Check if any attractions are available.
            clock = min(next_arrival, min_completion_time)
        # End of simulation.

        # Calculate overall percent utilization calculation for each attraction.
        cumulative_util = [cumulative_util[i] / time_open for i in attraction_range]

        # Calculate responses from simulation data.
        responses = {
            "total_departed": total_departed,
            "percent_departed": total_departed / total_visitors,
            "average_number_in_system": time_average / time_open,
            "attraction_utilization_percentages": cumulative_util,
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class AmusementParkMinDepart(Problem):
    """Class to make amusement park simulation-optimization problems."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "AMUSEMENTPARK-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Total Departed Visitors for Amusement Park"

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
        return (-1,)

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
    def optimal_value(cls) -> float | None:
        return None

    @classproperty
    @override
    def optimal_solution(cls) -> tuple | None:
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"queue_capacities"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (PARK_CAPACITY - NUM_ATTRACTIONS + 1,)
                + (1,) * (NUM_ATTRACTIONS - 1),
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 100,
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

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["number_attractions"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (self.model.factors["park_capacity"],) * self.dim

    def __init__(
        self,
        name: str = "AMUSEMENTPARK-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the Amusement Park Minimize Departures Problem.

        Args:
            name (str): User-specified name of the problem.
            fixed_factors (dict | None): Dictionary of user-specified problem factors.
            model_fixed_factors (dict | None): Subset of user-specified non-decision
                factors to pass through to the model.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=AmusementPark,
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict[str, tuple]:
        return {
            "queue_capacities": vector[:],
        }

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["queue_capacities"])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["total_departed"],)

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (0,)
        det_objectives_gradients = ()
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        # Check box constraints.
        if not super().check_deterministic_constraints(x):
            return False
        # Check if sum of queue capacities is less than park capacity.
        return sum(x) <= self.model.factors["park_capacity"]

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        num_elements: int = self.model.factors["number_attractions"]
        summation: int = self.model.factors["park_capacity"]
        vector = rand_sol_rng.integer_random_vector_from_simplex(
            n_elements=num_elements, summation=summation, with_zero=False
        )
        return tuple(vector)
