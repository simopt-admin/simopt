"""Simulate a single day of operation for an amusement park queuing problem."""

from __future__ import annotations

import math as math
from typing import Annotated, ClassVar, Final, Self, cast

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
from simopt.input_models import Exp, Gamma, WeightedChoice
from simopt.models._ext import patch_model

INF = float("inf")

# Default values for the model
PARK_CAPACITY: Final[int] = 350
NUM_ATTRACTIONS: Final[int] = 7


class AmusementParkConfig(BaseModel):
    """Configuration for the Amusement Park model."""

    park_capacity: Annotated[
        int,
        Field(
            default=PARK_CAPACITY,
            description=(
                "The total number of tourists waiting for attractions that can be "
                "maintained through park facilities, distributed across the "
                "attractions."
            ),
            ge=0,
        ),
    ]
    number_attractions: Annotated[
        int,
        Field(
            default=NUM_ATTRACTIONS,
            description="The number of attractions in the park.",
            # FIXME: strictly copying the original specification
            ge=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    time_open: Annotated[
        float,
        Field(
            default=480.0,
            description="The number of minutes per day the park is open.",
            ge=0,
        ),
    ]
    erlang_shape: Annotated[
        list[int],
        Field(
            default_factory=lambda: [2] * NUM_ATTRACTIONS,
            description=(
                "The shape parameter of the Erlang distribution for each "
                "attraction duration."
            ),
        ),
    ]
    erlang_scale: Annotated[
        list[float],
        Field(
            default_factory=lambda: [1 / 9] * NUM_ATTRACTIONS,
            description=(
                "The rate parameter of the Erlang distribution for each attraction "
                "duration."
            ),
        ),
    ]
    queue_capacities: Annotated[
        list[int],
        Field(
            default_factory=lambda: [50] * NUM_ATTRACTIONS,
            description=(
                "The capacity of the queue for each attraction based on the "
                "portion of facilities allocated."
            ),
        ),
    ]
    depart_probabilities: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.2] * NUM_ATTRACTIONS,
            description=(
                "The probability that a tourist will depart the park after "
                "visiting an attraction."
            ),
        ),
    ]
    arrival_gammas: Annotated[
        list[int],
        Field(
            default_factory=lambda: [1] * NUM_ATTRACTIONS,
            description=(
                "The gamma values for the poisson distributions dictating the "
                "rates at which tourists entering the park arrive at each "
                "attraction"
            ),
        ),
    ]
    transition_probabilities: Annotated[
        list[list[float]],
        Field(
            default_factory=lambda: [
                [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                [0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.3],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0, 0.3],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            ],
            description=(
                "The transition matrix that describes the probability of a tourist "
                "visiting each attraction after their current attraction."
            ),
        ),
    ]

    def _check_queue_capacities(self) -> None:
        if not all(cap >= 0 for cap in self.queue_capacities):
            raise ValueError("All queue capacities must be non-negative.")

    def _check_depart_probabilities(self) -> None:
        if len(self.depart_probabilities) != self.number_attractions:
            raise ValueError(
                "The number of departure probabilities must match the number of "
                "attractions."
            )
        if not all(0 <= prob <= 1 for prob in self.depart_probabilities):
            raise ValueError("All departure probabilities must be between 0 and 1.")

    def _check_arrival_gammas(self) -> None:
        if len(self.arrival_gammas) != self.number_attractions:
            raise ValueError(
                "The number of arrivals must match the number of attractions."
            )
        if not all(gamma >= 0 for gamma in self.arrival_gammas):
            raise ValueError("All arrival gammas must be non-negative.")

    def _check_transition_probabilities(self) -> None:
        """Validate the structure and consistency of the transition matrix.

        Checks that the transition matrix is square (same number of rows and columns),
        and that the sum of each row and its corresponding departure probability equals
        1.

        Returns:
            bool: True if all checks pass.

        Raises:
            ValueError: If any row has the wrong shape or an invalid total probability.
        """
        transition_sums = [sum(row) for row in self.transition_probabilities]
        if not (
            all(
                len(row) == len(self.transition_probabilities)
                for row in self.transition_probabilities
            )
            and all(
                transition_sums[i] + self.depart_probabilities[i] == 1
                for i in range(self.number_attractions)
            )
        ):
            raise ValueError(
                "The values you entered are invalid. "
                "Check that each row and depart probability sums to 1."
            )

    def _check_erlang_shape(self) -> None:
        """Validate the Erlang shape parameters for each attraction.

        Checks that the number of shape parameters matches the number of attractions,
        and that all shape values are non-negative.

        Returns:
            bool: True if all shape parameters are valid.

        Raises:
            ValueError: If the number of shape parameters is incorrect.
        """
        if len(self.erlang_shape) != self.number_attractions:
            raise ValueError(
                "The number of attractions must equal the number of Erlang shape "
                "parameters."
            )
        if not all(gamma >= 0 for gamma in self.erlang_shape):
            raise ValueError("All Erlang shape parameters must be non-negative.")

    def _check_erlang_scale(self) -> None:
        """Validate the Erlang scale parameters for each attraction.

        Checks that the number of scale parameters matches the number of attractions,
        and that all scale values are non-negative.

        Returns:
            bool: True if all scale parameters are valid.

        Raises:
            ValueError: If the number of scale parameters is incorrect.
        """
        if len(self.erlang_scale) != self.number_attractions:
            raise ValueError(
                "The number of attractions must equal the number of Erlang scales."
            )
        if not all(gamma >= 0 for gamma in self.erlang_scale):
            raise ValueError("All Erlang scale parameters must be non-negative.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_queue_capacities()
        self._check_depart_probabilities()
        self._check_arrival_gammas()
        self._check_transition_probabilities()
        self._check_erlang_shape()
        self._check_erlang_scale()

        if sum(self.queue_capacities) > self.park_capacity:
            raise ValueError(
                "The sum of the queue capacities must be less than or equal to the "
                "park capacity"
            )
        return self


class AmusementParkMinDepartConfig(BaseModel):
    """Configuration model for Amusement Park Min Depart Problem.

    A problem configuration that minimizes the total number of departed
    visitors from an amusement park by optimizing queue capacities.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default_factory=lambda: (PARK_CAPACITY - NUM_ATTRACTIONS + 1,)
            + (1,) * (NUM_ATTRACTIONS - 1),
            description="Initial solution from which solvers start.",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=100,
            description="Max # of replications for a solver to take.",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class AmusementPark(Model):
    """Amusement Park Model.

    A model that simulates a single day of operation for an
    amusement park queuing problem based on a poisson distributed tourist
    arrival rate, a next attraction transition matrix, and attraction
    durations based on an Erlang distribution. Returns the total number
    and percent of tourists to leave the park due to full queues.
    """

    class_name_abbr: ClassVar[str] = "AMUSEMENTPARK"
    class_name: ClassVar[str] = "Amusement Park"
    config_class: ClassVar[type[BaseModel]] = AmusementParkConfig
    n_rngs: ClassVar[int] = 3
    n_responses: ClassVar[int] = 4

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

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
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
        num_attractions: int = self.factors["number_attractions"]
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
        attraction_range = range(num_attractions)
        destination_range = range(num_attractions + 1)
        depart_idx = destination_range[-1]
        # initialize lists of each attraction's next completion time
        completion_times: list[float] = [INF] * num_attractions
        min_completion_time = INF
        min_completion_index = -1
        # initialize actual queues.
        queues: list[int] = [0] * num_attractions

        # create external arrival probabilities for each attraction.
        arrival_prob_sum: float = float(sum(arrival_gammas))
        arrival_probabilities: list[float] = [
            arrival_gammas[i] / arrival_prob_sum for i in attraction_range
        ]

        # Initiate clock variables for statistics tracking and event handling.
        clock: float = 0.0
        previous_clock: float = 0.0
        next_arrival = self.arrival_model.random(arrival_prob_sum)

        # Initialize quantities to track:
        total_visitors: int = 0
        total_departed: int = 0
        # initialize time average and utilization quantities.
        in_system: int = 0
        time_average: float = 0.0
        cumulative_util: list[float] = [0.0] * num_attractions

        # Run simulation over time horizon.
        while clock < time_open:
            # Count number of tourists on attractions and in queues.
            riders: int = 0
            delta_time: float = clock - previous_clock
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
                attraction_selection = cast(
                    int,
                    self.attraction_model.random(
                        attraction_range,
                        arrival_probabilities,
                    ),
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
                next_destination = cast(
                    int,
                    self.destination_model.random(
                        destination_range,
                        transition_probabilities[finished_attraction]
                        + [depart_probabilities[finished_attraction]],
                    ),
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
        percent_departed = total_departed / total_visitors if total_visitors else 0
        responses = {
            "total_departed": total_departed,
            "percent_departed": percent_departed,
            "average_number_in_system": time_average / time_open,
            "attraction_utilization_percentages": cumulative_util,
        }
        return responses, {}


class AmusementParkMinDepart(Problem):
    """Class to make amusement park simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "AMUSEMENTPARK-1"
    class_name: ClassVar[str] = "Min Total Departed Visitors for Amusement Park"
    config_class: ClassVar[type[BaseModel]] = AmusementParkMinDepartConfig
    model_class: ClassVar[type[Model]] = AmusementPark
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"queue_capacities"}

    @property
    def dim(self) -> int:  # noqa: D102
        return self.model.factors["number_attractions"]

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (self.model.factors["park_capacity"],) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict[str, tuple]:  # noqa: D102
        return {
            "queue_capacities": vector[:],
        }

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["queue_capacities"])

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        return RepResult(objectives=[Objective(stochastic=responses["total_departed"])])

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        # Check box constraints.
        if not super().check_deterministic_constraints(x):
            return False
        # Check if sum of queue capacities is less than park capacity.
        return sum(x) <= self.model.factors["park_capacity"]

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        num_elements: int = self.model.factors["number_attractions"]
        summation: int = self.model.factors["park_capacity"]
        vector = rand_sol_rng.integer_random_vector_from_simplex(
            n_elements=num_elements, summation=summation, with_zero=False
        )
        return tuple(vector)


patch_model(AmusementPark)
