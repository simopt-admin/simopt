"""Simulate a single day of traffic for queuing problem."""

import itertools
import logging
import math
from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override


class Road:
    """Defines the Road object class."""

    @property
    def roadid(self) -> int:
        """ID of the road."""
        return self._roadid

    @roadid.setter
    def roadid(self, value: int) -> None:
        self._roadid = value

    @property
    def startpoint(self) -> str:
        """Starting point of the road."""
        return self._startpoint

    @startpoint.setter
    def startpoint(self, value: str) -> None:
        self._startpoint = value

    @property
    def endpoint(self) -> str:
        """Ending point of the road."""
        return self._endpoint

    @endpoint.setter
    def endpoint(self, value: str) -> None:
        self._endpoint = value

    @property
    def direction(self) -> str:
        """Direction of the road."""
        return self._direction

    @direction.setter
    def direction(self, value: str) -> None:
        self._direction = value

    @property
    def queue(self) -> list:
        """Queue of cars on the road."""
        return self._queue

    @queue.setter
    def queue(self, value: list) -> None:
        self._queue = value

    @property
    def status(self) -> bool:
        """Status of the road light."""
        return self._status

    @status.setter
    def status(self, value: bool) -> None:
        self._status = value

    @property
    def queue_hist(self) -> dict:
        """History of queue lengths."""
        return self._queue_hist

    @queue_hist.setter
    def queue_hist(self, value: dict) -> None:
        self._queue_hist = value

    @property
    def road_length(self) -> float:
        """Length of the road."""
        return self._road_length

    @road_length.setter
    def road_length(self, value: float) -> None:
        self._road_length = value

    @property
    def overflow(self) -> bool:
        """Indicates if the road is overflowed."""
        return self._overflow

    @overflow.setter
    def overflow(self, value: bool) -> None:
        self._overflow = value

    @property
    def overflow_queue(self) -> dict:
        """Queue length of incoming roads when overflowed."""
        return self._overflow_queue

    @overflow_queue.setter
    def overflow_queue(self, value: dict) -> None:
        self._overflow_queue = value

    @property
    def incoming_roads(self) -> list:
        """List of incoming roads."""
        return self._incoming_roads

    @incoming_roads.setter
    def incoming_roads(self, value: list) -> None:
        self._incoming_roads = value

    @property
    def schedule(self) -> list:
        """Schedule of the road light."""
        return self._schedule

    @schedule.setter
    def schedule(self, value: list) -> None:
        self._schedule = value

    @property
    def nextchange(self) -> float:
        """Next change time of the road light."""
        return self._nextchange

    @nextchange.setter
    def nextchange(self, value: float) -> None:
        self._nextchange = value

    def __init__(
        self, roadid: int, startpoint: str, endpoint: str, direction: str
    ) -> None:
        """Initialize the Road object."""
        self._roadid = roadid
        self._startpoint = startpoint
        self._endpoint = endpoint
        self._direction = direction
        self._queue = []
        self._status = False
        self._queue_hist = {}  # to store queue length at each time point
        self._road_length = 10
        self._overflow = False
        # to store queue length of incoming roads at each time point when overflowed
        self._overflow_queue = {}
        self._incoming_roads = []
        self._schedule = []
        self._nextchange = 0  # TODO: confirm this

    def update_light(self, schedule: list, t: float) -> None:
        """Updates the light from red to green and vice versa.

        Args:
            schedule (list): all times where a light changes status
            t (float): current time in system
        """
        for time in schedule:
            if time == t:
                if self.status:
                    self.status = False
                else:
                    self.status = True
                    if len(self.queue) > 0 and self.queue[0] != 0:
                        self.queue[0].starttime = t


class Intersection:
    """Defines the Intersection object class."""

    @property
    def name(self) -> str:
        """Name of the intersection."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def schedule(self) -> list:
        """Schedule of the intersection."""
        return self._schedule

    @schedule.setter
    def schedule(self, value: list) -> None:
        self._schedule = value

    @property
    def horizontalroads(self) -> list[Road]:
        """List of horizontal roads."""
        return self._horizontalroads

    @horizontalroads.setter
    def horizontalroads(self, value: list[Road]) -> None:
        self._horizontalroads = value

    @property
    def verticalroads(self) -> list[Road]:
        """List of vertical roads."""
        return self._verticalroads

    @verticalroads.setter
    def verticalroads(self, value: list[Road]) -> None:
        self._verticalroads = value

    @property
    def offset(self) -> int:
        """Offset of the intersection."""
        return self._offset

    @offset.setter
    def offset(self, value: int) -> None:
        self._offset = value

    def __init__(self, name: str, _roads: list[Road]) -> None:
        """Initialize the Intersection object.

        Args:
            name (str): Name of the intersection.
            _roads: list[Road]: List of all roads in the system.
        """
        self._name = name
        self._schedule = []
        self._horizontalroads = []
        self._verticalroads = []
        self._offset = 0
        # self._roads = _roads

    def connect_roads(self, roads: list[Road], offset: int) -> None:
        """Sets specific roads as attributes of the intersection they belong to.

        Args:
            roads (list[Road]): list of all roads in the system
            offset (int): offset of the intersection
        """
        for road in roads:
            # Only add roads that are connected to the intersection
            if road.endpoint != self.name:
                continue
            # Add the road to the appropriate list based on its direction
            if road.direction in ["East", "West"]:
                self.horizontalroads.append(road)
                road.status = offset == 0
            else:
                self.verticalroads.append(road)
                road.status = offset != 0


class Car:
    """Defines the Car object class."""

    @property
    def identify(self) -> int:
        """ID of the car."""
        return self._identify

    @identify.setter
    def identify(self, value: int) -> None:
        self._identify = value

    @property
    def arrival(self) -> float:
        """Arrival time of the car."""
        return self._arrival

    @arrival.setter
    def arrival(self, value: float) -> None:
        self._arrival = value

    @property
    def initialarrival(self) -> float:
        """Initial arrival time of the car."""
        return self._initialarrival

    @initialarrival.setter
    def initialarrival(self, value: float) -> None:
        self._initialarrival = value

    @property
    def finishtime(self) -> float:
        """Final time of the car in the system."""
        return self._finishtime

    @finishtime.setter
    def finishtime(self, value: float) -> None:
        self._finishtime = value

    @property
    def path(self) -> list:
        """Path of the car."""
        return self._path

    @path.setter
    def path(self, value: list) -> None:
        self._path = value

    @property
    def locationindex(self) -> int:
        """Index of the car's current location."""
        return self._locationindex

    @locationindex.setter
    def locationindex(self, value: int) -> None:
        self._locationindex = value

    @property
    def timewaiting(self) -> float:
        """Time the car has been waiting."""
        return self._timewaiting

    @timewaiting.setter
    def timewaiting(self, value: float) -> None:
        self._timewaiting = value

    @property
    def primarrival(self) -> float:
        """Primary arrival time of the car."""
        return self._primarrival

    @primarrival.setter
    def primarrival(self, value: float) -> None:
        self._primarrival = value

    @property
    def place_in_queue(self) -> int | None:
        """Position of the car in the queue."""
        return self._place_in_queue

    @place_in_queue.setter
    def place_in_queue(self, value: int | None) -> None:
        self._place_in_queue = value

    @property
    def nextstart(self) -> float:
        """Next start time of the car."""
        return self._nextstart

    @nextstart.setter
    def nextstart(self, value: float) -> None:
        self._nextstart = value

    @property
    def moving(self) -> bool:
        """Indicates if the car is moving."""
        return self._moving

    @moving.setter
    def moving(self, value: bool) -> None:
        self._moving = value

    @property
    def next_sec_arrival(self) -> float:
        """Next second arrival time of the car."""
        return self._next_sec_arrival

    @next_sec_arrival.setter
    def next_sec_arrival(self, value: float) -> None:
        self._next_sec_arrival = value

    @property
    def prevstop(self) -> float:
        """Previous stop of the car."""
        return self._prevstop

    @prevstop.setter
    def prevstop(self, value: float) -> None:
        self._prevstop = value

    @property
    def visits(self) -> list:
        """List of visits of the car."""
        return self._visits

    @visits.setter
    def visits(self, value: list) -> None:
        self._visits = value

    @property
    def finished(self) -> bool:
        """Indicates if the car has finished its path."""
        return self._finished

    @finished.setter
    def finished(self, value: bool) -> None:
        self._finished = value

    def __init__(
        self, carid: int, arrival: int | float, path: list[Road], visits: list[str]
    ) -> None:
        """Initialize the Car object."""
        self._identify = carid
        self._arrival = arrival
        self._initialarrival = arrival
        self._finishtime = 0
        self._path = path
        self._locationindex = 0
        self._timewaiting = 0
        self._primarrival = arrival
        self._place_in_queue = None
        self._nextstart = 0
        self._moving = False
        self._next_sec_arrival = 0
        self._prevstop = 0
        self._visits = visits
        self._finished = False

    def update_location(self) -> None:
        """Update the car's location index."""
        self.locationindex += 1


class TrafficLight(Model):
    """Traffic Light Model.
    
    A model that simulates a series of intersections and their light \
    schedules. As cars travel through the system, their waiting \
    time is tracked.
    """

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 3

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 1

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "lambdas": {
                "description": (
                    "Rate parameter of the time interval distribution (seconds) "
                    "for generating each car."
                ),
                "datatype": list,
                "default": [4, 4, 1, 1],
            },
            "runtime": {
                "description": "The number of seconds that the traffic model runs",
                "datatype": int,
                "default": 7200,
            },
            "numintersections": {
                "description": "The number of intersections in the traffic model",
                "datatype": int,
                "default": 4,
                "isDatafarmable": False,
            },
            "decision_vector": {
                "description": (
                    "Delay (seconds) in light schedule based on distance from "
                    "first intersection (intersection A in docs)."
                ),
                "datatype": list,
                "default": [1.0, 2.0, 3.0],
            },
            "speed": {
                "description": "Constant speed in meter/second for the cars",
                "datatype": float,
                "default": 5,
            },
            "carlength": {
                "description": "Length (meters) of each car",
                "datatype": float,
                "default": 4.5,
            },
            "reaction": {
                "description": "Reaction time (seconds) of cars in queue",
                "datatype": float,
                "default": 0.1,
            },
            "transition_probs": {
                "description": (
                    "The transition probability of a car end at each point from their "
                    "current starting point"
                    "from N, S, E and W."
                ),
                "datatype": list,
                "default": [[0, 1, 2, 3], [1, 0, 2, 3], [1, 2, 0, 3], [1, 2, 3, 0]],
            },
            "pause": {
                "description": "The pause (seconds) before move on a green light",
                "datatype": float,
                "default": 0.1,
            },
            "car_distance": {
                "description": "The distance (meters) between cars",
                "datatype": float,
                "default": 0.5,
            },
            "length_arteries": {
                "description": "The length (meters) of artery roads",
                "datatype": float,
                "default": 100,
            },
            "length_veins": {
                "description": "The length (meters) of vein road",
                "datatype": float,
                "default": 100,
            },
            "redlight_arteries": {
                "description": (
                    "The length of redlight duration of artery roads in "
                    "each intersection"
                ),
                "datatype": list,
                "default": [10, 10, 10, 10],
            },
            "redlight_veins": {
                "description": (
                    "The length of redlight duration of vein roads in each intersection"
                ),
                "datatype": list,
                "default": [20, 20, 20, 20],
            },
            "n_veins": {
                "description": ("The number of vein roads in the system"),
                "datatype": int,
                "default": 2,
            },
            "n_arteries": {
                "description": ("The number of artery roads in the system"),
                "datatype": int,
                "default": 2,
            },
            "nodes": {
                "description": ("The number of nodes in the system"),
                "datatype": int,
                "default": 8,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "runtime": self._check_runtime,
            "lambdas": self._check_lambdas,
            "numintersections": self._check_numintersections,
            "decision_vector": self._check_decision_vector,
            "speed": self._check_speed,
            "carlength": self._check_carlength,
            "reaction": self._check_reaction,
            "transition_probs": self._check_transition_probs,
            "pause": self._check_pause,
            "car_distance": self._check_car_distance,
            "length_arteries": self._check_length_arteries,
            "length_veins": self._check_length_veins,
            "redlight_arteries": self._check_redlight_arteries,
            "redlight_veins": self._check_redlight_veins,
            "n_veins": self._check_n_veins,
            "n_arteries": self._check_n_arteries,
            "nodes": self._check_nodes,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Traffic Light Model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def _check_lambdas(self) -> None:
        # not zero #arteries > veins
        if len(self.factors["lambdas"]) != 4:
            raise ValueError(
                "Lambdas must be a list of 4 elements, representing the "
                "rate parameters for each direction (N, S, E, W)."
            )

    def _check_runtime(self) -> None:
        if self.factors["runtime"] <= 0:
            raise ValueError("Runtime must be greater than 0.")

    def _check_numintersections(self) -> None:
        if self.factors["numintersections"] <= 0:
            raise ValueError("Number of intersections must be greater than 0.")

    def _check_decision_vector(self) -> None:
        all_positive = all(value >= 0 for value in self.factors["decision_vector"])
        if not all_positive:
            raise ValueError(
                "Decision vector values must be greater than or equal to 0."
            )

    def _check_speed(self) -> None:
        if self.factors["speed"] <= 0:
            raise ValueError("Speed must be greater than 0.")

    def _check_carlength(self) -> None:
        if self.factors["carlength"] <= 0:
            raise ValueError("Car length must be greater than 0.")

    def _check_reaction(self) -> None:
        if self.factors["reaction"] <= 0:
            raise ValueError("Reaction time must be greater than 0.")

    def _check_transition_probs(self) -> None:
        for prob in list(
            itertools.chain.from_iterable(self.factors["transition_probs"])
        ):
            if prob < 0:
                raise ValueError(
                    "Transition probabilities must be greater than or equal to 0."
                )

    def _check_pause(self) -> None:
        if self.factors["pause"] <= 0:
            raise ValueError("Pause time must be greater than 0.")

    def _check_car_distance(self) -> None:
        if self.factors["car_distance"] <= 0:
            raise ValueError("Car distance must be greater than 0.")

    def _check_length_arteries(self) -> None:
        if self.factors["length_arteries"] <= 0:
            raise ValueError("Length of arteries must be greater than 0.")

    def _check_length_veins(self) -> None:
        if self.factors["length_veins"] <= 0:
            raise ValueError("Length of veins must be greater than 0.")

    def _check_redlight_arteries(self) -> None:
        all_positive = all(
            redlight > 0 for redlight in self.factors["redlight_arteries"]
        )
        if not all_positive:
            raise ValueError("Redlight duration of arteries must be greater than 0.")

    def _check_redlight_veins(self) -> None:
        all_positive = all(redlight > 0 for redlight in self.factors["redlight_veins"])
        if not all_positive:
            raise ValueError("Redlight duration of veins must be greater than 0.")

    def _check_n_veins(self) -> None:
        if self.factors["n_veins"] <= 0:
            raise ValueError("Number of veins must be greater than 0.")

    def _check_n_arteries(self) -> None:
        if self.factors["n_arteries"] <= 0:
            raise ValueError("Number of arteries must be greater than 0.")

    def _check_nodes(self) -> None:
        if self.factors["nodes"] != (2 * self.factors["n_veins"]) + (
            2 * self.factors["n_arteries"]
        ):
            raise ValueError(
                "Number of nodes must be equal to "
                "(2 * number of veins) + (2 * number of arteries)."
            )

    @override
    def check_simulatable_factors(self) -> bool:
        if len(self.factors["decision_vector"]) != self.factors["numintersections"] - 1:
            raise ValueError(
                "Decision vectors must be equal to the number of intersections - 1."
            )
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing two dictionaries:
                - responses (dict): Performance measures of interest.
                    - "WaitingTime" = average time waiting at a light
                - gradients (dict): Gradient estimates for each response.
        """
        # Designate separate RNGs for start, end positions and interarrival times
        start_rng = rng_list[0]
        end_rng = rng_list[1]
        arrival_rng = rng_list[2]

        # Initializes variables to start the simulation
        t = 0
        next_car_gen = 0
        outbounds: int = self.factors["runtime"] + 1
        car_sim_index = 0
        next_start = outbounds
        next_sec_arrival = outbounds
        min_prim_arrival = 0

        def start_prob(n_vein: int, n_artery: int, lambdas: list[float]) -> list[float]:
            start_prob: list[float] = []
            i = 0
            while i < n_artery:
                start_prob.append((lambdas[0] / sum(lambdas)) / n_artery)
                i += 1
            while i < 2 * n_artery:
                start_prob.append(lambdas[1] / sum(lambdas) / n_artery)
                i += 1
            j = 0
            while j < n_vein:
                if j % 2 == 0:
                    start_prob.append(0)
                else:
                    start_prob.append(lambdas[2] / sum(lambdas))
                j += 1
            k = 0
            while k < n_vein:
                if k % 2 == 0:
                    start_prob.append(lambdas[3] / sum(lambdas))
                else:
                    start_prob.append(0)
                k += 1
            return start_prob

        start_prob = start_prob(
            self.factors["n_veins"], self.factors["n_arteries"], self.factors["lambdas"]
        )

        # offset
        def offset(
            decision_vector: list[float],
            redlight_veins: list[float],
            n_artery: int,
            n_vein: int,
        ) -> list[int]:
            offset = [0]
            count = 0

            count += 1
            i = 0
            while i < (n_artery - 1):
                offset.append(decision_vector[i])
                i += 1
            j = 0
            while j < (n_artery):
                offset.append(decision_vector[j] + redlight_veins[j])
                j += 1
            offset.append(0)
            k = 0
            while k < (n_artery - 1):
                offset.append(decision_vector[k])
                k += 1
            count = 1
            while count < n_vein:
                count += 1
                a = 0
                i = n_artery - 1
                while a < n_artery:
                    offset.append(decision_vector[i])
                    a += 1
                    i += 1
                j = n_artery
                b = 0
                while b < n_artery:
                    offset.append(decision_vector[j - 1] + redlight_veins[j - 1])
                    b += 1
                    j += 1
                c = 0
                k = n_artery - 1
                while c < (n_artery):
                    offset.append(decision_vector[k])
                    c += 1
                    k += 1
                if count == n_vein:
                    break
            return offset

        self.factors["offset"] = offset(
            self.factors["decision_vector"],
            self.factors["redlight_veins"],
            self.factors["n_arteries"],
            self.factors["n_veins"],
        )

        def transition_matrix(
            n_vein: int, n_artery: int, transition_probs: list[list[float]]
        ) -> list[list[float]]:
            transition_matrix = [
                [0 for _ in range((2 * n_artery) + (2 * n_vein))]
                for _ in range((2 * n_artery) + (2 * n_vein))
            ]

            transition_probs_sum = []
            for i in range(4):
                transition_probs_sum.append(sum(transition_probs[i]))
            # NORTH 1
            # S1:
            transition_matrix[0][(2 * n_artery) + n_vein - 1] = (
                1 / (transition_probs[0][1] + transition_probs[0][3])
            ) * transition_probs[0][1]
            # West:
            count_west = 0
            for i in range((2 * n_artery) + n_vein, (2 * n_artery) + (2 * n_vein)):
                if i % 2 == 0:
                    count_west += 1
                    transition_matrix[0][i] = (
                        (1 / (transition_probs[0][1] + transition_probs[0][3]))
                        * transition_probs[0][3]
                    ) / count_west

            # OTHER NORTH
            for j in range(1, n_artery):
                # E:
                count_east = 0
                for i in range((n_artery), n_artery + n_vein - 1):
                    if i % 2 == 0:
                        count_east += 1
                        transition_matrix[j][i] = (
                            (1 / transition_probs_sum[0]) * transition_probs[0][2]
                        ) / count_east
                # S:
                for i in range(n_artery + n_vein, (2 * n_artery) + n_vein - 1):
                    transition_matrix[j][i] = (
                        (1 / transition_probs_sum[0]) * transition_probs[0][1]
                    ) / (n_artery - 1)
                # W:
                for i in range((2 * n_artery) + n_vein, (2 * n_artery) + (2 * n_vein)):
                    if i % 2 == 0:
                        transition_matrix[j][i] = (
                            (1 / transition_probs_sum[0]) * transition_probs[0][3]
                        ) / count_west

            # LAST SOUTH
            # lastN:
            transition_matrix[n_vein + n_artery][n_artery - 1] = (
                1 / (transition_probs[1][0] + transition_probs[1][2])
            ) * transition_probs[1][0]
            # East:
            for i in range(n_artery, (n_artery + n_vein)):
                if i % 2 == 0:
                    transition_matrix[2 * n_artery][i] = (
                        (1 / (transition_probs[1][0] + transition_probs[1][2]))
                        * transition_probs[1][2]
                    ) / count_east

            # OTHER SOUTH
            for j in range((n_vein + n_artery) + 1, (2 * n_artery) + n_vein):
                # N:
                for i in range(0, n_artery - 1):
                    transition_matrix[j][i] = (
                        (1 / transition_probs_sum[1]) * transition_probs[1][0]
                    ) / (n_artery - 1)

                # E:
                for i in range((n_artery), (n_vein + n_artery) - 1):
                    if i % 2 == 0:
                        transition_matrix[j][i] = (
                            (1 / transition_probs_sum[1]) * transition_probs[1][2]
                        ) / count_east

                # W:
                for i in range((2 * n_artery) + n_vein, (2 * n_artery) + (2 * n_vein)):
                    if i % 2 == 0:
                        transition_matrix[j][i] = (
                            (1 / transition_probs_sum[1]) * transition_probs[1][3]
                        ) / count_west

            # EAST
            for j in range(n_artery + 1, (n_artery + n_vein)):
                if j % 2 != 0:
                    # N:
                    for i in range(0, n_artery):
                        transition_matrix[j][i] = (
                            (1 / transition_probs_sum[2]) * transition_probs[2][0]
                        ) / n_artery

                    # S:
                    for i in range((n_artery + n_vein), ((2 * n_artery) + n_vein - 1)):
                        transition_matrix[j][i] = (
                            (1 / transition_probs_sum[2]) * transition_probs[2][1]
                        ) / (n_artery - 1)

                    # W:
                    for i in range(
                        ((2 * n_artery) + n_vein), ((2 * n_vein) + (2 * n_artery))
                    ):
                        if i % 2 == 0:
                            transition_matrix[j][i] = (
                                (1 / transition_probs_sum[2]) * transition_probs[2][3]
                            ) / count_west

            # WEST
            for j in range((2 * n_artery) + n_vein, (2 * n_artery) + (2 * n_vein)):
                if j % 2 != 0:
                    # N:
                    for i in range(0, n_artery - 1):
                        transition_matrix[j][i] = (
                            (1 / transition_probs_sum[3]) * transition_probs[3][0]
                        ) / (n_artery - 1)

                    # E:
                    for i in range(n_artery, (n_artery + n_vein)):
                        if i % 2 == 0:
                            transition_matrix[j][i] = (
                                (1 / transition_probs_sum[3]) * transition_probs[3][2]
                            ) / count_east

                    # S:
                    for i in range((n_artery + n_vein), ((2 * n_artery) + n_vein)):
                        transition_matrix[j][i] = (
                            (1 / transition_probs_sum[3]) * transition_probs[3][1]
                        ) / n_artery
            return transition_matrix

        transition_matrix = np.array(
            transition_matrix(
                self.factors["n_veins"],
                self.factors["n_arteries"],
                self.factors["transition_probs"],
            )
        )

        # Draw out map of all locations in system

        def roadmap(n_vein: int, n_artery: int) -> tuple[list, list]:
            def num_to_label(n: int) -> str:
                label = ""
                while n >= 0:
                    label = chr(n % 26 + ord("A")) + label
                    n = n // 26 - 1
                return label

            rows = n_vein + 2
            cols = n_artery + 2
            roadmap = [[0 for _ in range(cols)] for _ in range(rows)]

            roadmap[0][0] = ""
            roadmap[-1][-1] = ""
            roadmap[0][-1] = ""
            roadmap[-1][0] = ""

            for col in range(1, cols - 1):
                roadmap[0][col] = f"N{col}"
                roadmap[rows - 1][col] = f"S{col}"

            for row in range(1, rows - 1):
                roadmap[row][0] = f"W{row}"
                roadmap[row][cols - 1] = f"E{row}"

            counter = 0
            label = []
            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    roadmap[row][col] = num_to_label(counter)
                    label.append(num_to_label(counter))
                    counter += 1

            return roadmap, label

        roadmap, labels = roadmap(self.factors["n_veins"], self.factors["n_arteries"])
        roadmap = np.array(roadmap)

        # List each location and the locations that are next accessible
        def graph(
            roadmap: np.ndarray, label: list[str], n_vein: int, n_artery: int
        ) -> dict[str, list[str]]:
            my_dict: dict[str, list[str]] = {}
            j = 0
            vein = 1
            count = 0
            while j < n_vein:
                if j % 2 == 0:
                    for a in range(n_artery):
                        list1 = []
                        list1 = [
                            roadmap[vein - 1][a + 1],
                            roadmap[vein][a + 2],
                            roadmap[vein + 1][a + 1],
                        ]
                        my_dict[label[count]] = list1
                        count += 1
                    vein += 1
                elif j % 2 != 0:
                    for a in range(n_artery):
                        list2 = []
                        list2 = [
                            roadmap[vein - 1][a + 1],
                            roadmap[vein][a],
                            roadmap[vein + 1][a + 1],
                        ]
                        my_dict[label[count]] = list2
                        count += 1
                    vein += 1
                j += 1
            # north
            for i in range(n_artery):
                my_dict[roadmap[0][i + 1]] = [roadmap[1][i + 1]]

            # east
            for i in range(n_vein):
                if i % 2 == 0:
                    my_dict[roadmap[i + 1][-1]] = []
                else:
                    my_dict[roadmap[i + 1][-1]] = [roadmap[i + 1][-2]]
            # south
            for i in range(n_artery):
                my_dict[roadmap[-1][i + 1]] = [roadmap[-2][i + 1]]

            # west
            for i in range(n_vein):
                if i % 2 == 0:
                    my_dict[roadmap[i + 1][0]] = [roadmap[i + 1][1]]
                else:
                    my_dict[roadmap[i + 1][0]] = []
            return my_dict

        graph = graph(
            roadmap, labels, self.factors["n_veins"], self.factors["n_arteries"]
        )

        # Lists each location in the system
        points = list(graph.keys())
        # Lists each location in the system

        def find_shortest_path(
            graph: dict[str, list[str]],
            start: str,
            end: str,
            path: list[str] | None = None,
        ) -> list[str] | None:
            """Find the shortest path between two points.

            NOTE: NO LEFT TURN

            Args:
                graph (dict[str, list[str]]): dictionary with all locations and their
                    connections
                start (str): name of starting location
                end (str): name of ending location
                path (list[str] | None, optional): list of locations that represent the
                    car's path

            Returns:
                list[str] | None: list of locations that represent the shortest path
                    from start to finish. None if no path exists.
            """
            if path is None:
                path = []
            path = [*path, start]
            # Path starts and ends at the same point
            if start == end:
                return path

            shortest = None
            for node in graph[start]:
                # if node not in path:
                if sum(x == node for x in path) < 2:
                    if len(path) >= 2:  # no left turn
                        direction1 = find_direction(path[-2], path[-1], roadmap)
                        direction2 = find_direction(path[-1], node, roadmap)
                        turn = find_turn(direction1 + direction2)
                        if turn in ["Right", "Straight"]:
                            newpath = find_shortest_path(graph, node, end, path)
                            if newpath and (
                                not shortest or len(newpath) < len(shortest)
                            ):
                                shortest = newpath
                    else:
                        newpath = find_shortest_path(graph, node, end, path)
                        if newpath and (not shortest or len(newpath) < len(shortest)):
                            shortest = newpath
            return shortest

        def generate_path(start: int) -> list[str]:
            """Generates shortest path through two random start and end locations.

            Args:
                start (int): starting point of the car

            Returns:
                list[str]: list of locations that car visits
            """
            path = None
            while path is None:
                end = end_rng.choices(
                    population=range(self.factors["nodes"]),
                    weights=transition_matrix[start],
                )[0]
                path = find_shortest_path(
                    graph,
                    points[start + self.factors["numintersections"]],
                    points[end + self.factors["numintersections"]],
                )
            return path

        def find_direction(start: str, end: str, roadmap: np.ndarray) -> str:
            """Takes in road and finds its direction based on the map.

            Args:
                start (str): starting point of the road
                end (str): ending point of the road
                roadmap (np.ndarray): map of all locations in the system

            Returns:
                str: direction of the road
            """
            yloc1, xloc1 = np.where(roadmap == start)
            yloc2, xloc2 = np.where(roadmap == end)
            if xloc1 > xloc2:
                direction = "West"
            elif xloc1 < xloc2:
                direction = "East"
            elif yloc1 > yloc2:
                direction = "North"
            else:
                direction = "South"
            return direction

        def find_turn(roadcombo: str) -> str:
            """Assigns the direction of a turn when given two roads.

            Args:
                roadcombo (str): combined directions of roads

            Returns:
                str: direction of turn
            """
            turnkey = {
                "Straight": ["WestWest", "EastEast", "SouthSouth", "NorthNorth"],
                "Left": ["NorthWest", "EastNorth", "SouthEast", "WestSouth"],
                "Right": ["NorthEast", "WestNorth", "SouthWest", "EastSouth"],
                "Uturn": ["NorthSouth", "SouthNorth", "EastWest", "WestEast"],
            }
            turn = ""
            for key, values in turnkey.items():
                for value in values:
                    if roadcombo == value:
                        turn = key
            return turn

        road_pair = []
        for key, values in graph.items():
            for node in values:
                road_pair.append((key, node))

        def exit_nodes(n_vein: int, n_artery: int, roadmap: np.ndarray) -> list[str]:
            exit_nodes = []
            for i in range(1, n_artery + 1):
                exit_nodes.append(roadmap[0][i])
            for i in range(1, n_artery + 1):
                exit_nodes.append(roadmap[-1][i])
            for i in range(1, n_vein + 1):
                if i % 2 != 0:
                    exit_nodes.append(roadmap[i][-1])
                else:
                    exit_nodes.append(roadmap[i][0])
            return list({str(x) for x in exit_nodes})

        road_pair = sorted(
            road_pair,
            key=lambda x: (
                x[1]
                in exit_nodes(
                    self.factors["n_veins"], self.factors["n_arteries"], roadmap
                ),
            ),
        )
        # Generates list of all road objects in the system
        roads: list[Road] = []
        for roadid, (key, value) in enumerate(road_pair):
            direction = find_direction(key, value, roadmap)
            roads.append(Road(roadid, key, value, direction))
            roads[roadid].nextchange = outbounds  # TODO: fix this
            if direction == "West" or direction == "East":
                roads[roadid].road_length = self.factors["length_veins"]
            else:
                roads[roadid].road_length = self.factors["length_arteries"]
            roads[roadid].queue.append(0)

        # add incoming roads
        # only when startpoints are A, B, C, D, etc... have the overflow with overflow
        # queue length issue
        start_points = np.array([key for key, _ in road_pair])
        for road in roads:
            endpoint = road.endpoint
            if endpoint in labels:
                indices = np.where(endpoint == start_points)[0]
                for i in indices:
                    if i < self.factors["numintersections"] * 3:
                        turn = find_turn(road.direction + roads[i].direction)
                        if turn != "Left":
                            roads[i].incoming_roads.append(road)

        def find_roads(visits: list[str]) -> list[Road]:
            """Finds the roads that a car will take on its path.

            Args:
                visits (list[str]): all locations in a car's path

            Returns:
                list[Road]: list of road objects that the car travels on
            """
            path: list[Road] = []
            for i in range(len(visits) - 1):
                for road in roads:
                    if road.startpoint == visits[i] and road.endpoint == visits[i + 1]:
                        path.append(road)
            return path

        # Generates list of all intersection objects
        intersections = []
        decision_vectors: list[float] = self.factors["decision_vector"]
        for i in range(self.factors["numintersections"]):
            location = points[i]
            intersections.append(Intersection(location, roads))
            offset = 0 if i == 0 else decision_vectors[i - 1]
            intersections[i].connect_roads(roads, offset)

        # Generates light schedule of road

        greenlight_arteries = self.factors["redlight_veins"]
        greenlight_veins = self.factors["redlight_arteries"]

        for roadid in range(3 * self.factors["numintersections"]):
            offset = self.factors["offset"][roadid]
            road = roads[roadid]
            ind = labels.index(road.endpoint)
            if road.direction == "West" or road.direction == "East":
                interval = [self.factors["redlight_veins"][ind], greenlight_veins[ind]]
            else:
                interval = [
                    greenlight_arteries[ind],
                    self.factors["redlight_arteries"][ind],
                ]
            for i in range(math.ceil(self.factors["runtime"] / min(interval)) + 2):
                if i == 0:
                    road.schedule.append(0)
                elif i % 2 == 0:  # even time index, A: red -> green, V: green -> red
                    road.schedule.append(road.schedule[-1] + interval[1])
                else:  # odd time index, A: green -> red, V: red -> green
                    if i == 1:
                        offsetcalc = offset % interval[0]
                        if offsetcalc == 0:
                            offsetcalc = interval[0]
                        road.schedule.append(road.schedule[-1] + offsetcalc)
                    else:
                        road.schedule.append(road.schedule[-1] + interval[0])

        def find_nextlightchange_road(roads: list[Road], t: float) -> float:
            """Finds the next time any intersection light will change signal.

            Args:
                roads (list): list of all intersection objects
                t (float): current time in system

            Returns:
                float: time that the next light changes
            """
            # TODO: figure out if this also needs returned:
            # list
            #     list of locations that a light changes at

            mintimechange: float = self.factors["runtime"]
            # Loops through roads to find a minimum light changing time
            for road in roads[: 3 * self.factors["numintersections"]]:
                nextchange = min([i for i in road.schedule if i > t])
                if nextchange <= mintimechange:
                    mintimechange = nextchange
            return mintimechange

        def update_road_lights(t: float, roads: list[Road]) -> None:
            """Updates the intersections with their new light status.

            Args:
                t (float): current time in system
                roads (list[Road]): list of all intersection objects
            """
            if t == 0:
                nextlightlocation = roads[: 3 * self.factors["numintersections"]]
            else:
                nextlightlocation: list[Road] = []
                for road in roads[: 3 * self.factors["numintersections"]]:
                    if t in road.schedule:
                        nextlightlocation.append(road)
            for road in nextlightlocation:
                road.update_light(road.schedule, t)
                road.nextchange = min(i for i in road.schedule if i > t)
            status = []
            nextc = []
            for road in roads[: 3 * self.factors["numintersections"]]:
                light_status = "Green" if road.status else "Red"
                status.append(light_status)
                nextc.append(road.nextchange)

        cars: list[Car] = []
        lambda_sum = sum(self.factors["lambdas"])

        def gen_car(t: float) -> float:
            """Generates list of all car objects as they are created.

            Args:
                t (float): current time in system

            Returns:
                float: time that the next car is introduced to the system
            """
            # Set arrival time of next car
            initialarrival = t + arrival_rng.expovariate(lambda_sum)
            # Determine arrival location of next car
            start = start_rng.choices(
                population=range(self.factors["nodes"]), weights=start_prob
            )[0]
            visits = generate_path(start)
            while visits is None or len(visits) == 1:
                visits = generate_path(start)
            identify = len(cars)
            path = find_roads(visits)
            cars.append(Car(identify, initialarrival, path, visits))
            cars[identify].nextstart = outbounds
            cars[identify].next_sec_arrival = outbounds
            return initialarrival

        def find_place_in_queue(car: Car, road: Road, t: float) -> None:
            """Finds a car's place in queue and assigns it a new start time.

            Args:
                car (Car): car object
                road (Road): road object
                t (float): current time in system
            """
            queueindex = len(road.queue) - 1
            while road.queue[queueindex] == 0 and queueindex > 0:
                queueindex -= 1
            # Car is not the first in its queue
            if queueindex != 0 or road.queue[0] != 0:
                # Car is second in queue
                if len(road.queue) == queueindex + 1:
                    road.queue.append(car)
                # Car is third or later in queue   duplicated with at first
                # else:
                #    road.queue[queueindex + 1] = car
                car.place_in_queue = queueindex + 1
                car.nextstart = (
                    road.queue[queueindex].nextstart + self.factors["reaction"]
                )
            # Car is the first in its queue
            else:
                road.queue[queueindex] = car
                car.place_in_queue = queueindex
                # Car is at the end of its path
                if car.locationindex == len(car.path) - 1:
                    car.nextstart = outbounds
                    car.next_sec_arrival = outbounds
                # Car still has a road to travel to
                else:
                    # Light is green on the road that the car is on
                    if road.status:
                        car.nextstart = t
                    # Light is red on the road that the car is on
                    else:
                        car.nextstart = road.nextchange

        # Lights are turned on and the first car is created
        update_road_lights(0, roads)
        gen_car(next_car_gen)
        currentcar = cars[car_sim_index]
        movingcar = cars[0]
        arrivingcar = movingcar
        # Loops through time until runtime is reached
        sumwait = 0
        finishedcars = 0
        overflow_total = {}
        overflow_len_total = {}
        avg_wait = 0
        last_avg_wait = 0
        avg_wait_over_time: dict[float, float] = {}
        percent_done = 0
        min_car_index = 0
        while t < self.factors["runtime"]:
            # Log the percentage done every time it changes
            if t / self.factors["runtime"] > percent_done + 0.01:
                percent_done = round(t / self.factors["runtime"], 2)
                percent_done_int = int(percent_done * 100)
                logging.debug(f"Replication is {percent_done_int}% done")
            # Assigns the next time a light changes
            next_light_time = find_nextlightchange_road(roads, t)
            next_event_time = min(
                min_prim_arrival,
                next_light_time,
                next_start,
                next_sec_arrival,
                next_car_gen,
            )
            # The next event is a car being introduced to the system
            if next_event_time == min_prim_arrival:
                t = min_prim_arrival
                for i in range(12, 18):
                    roads[i].nextchange = t

                car = cars[car_sim_index]
                car.prevstop = t
                car_idx = car.identify
                position = car.visits[car.locationindex]
                road = car.path[car.locationindex].roadid
                logging.debug(f"{t}, Car {car_idx}, Start, Pos {position}, Road {road}")
                # A new car is generated
                next_car_gen = gen_car(t)
                min_prim_arrival = next_car_gen
                car_sim_index += 1

                # The arriving car arrives into the system based on its path
                currentcar = cars[car_sim_index - 1]
                initroad = currentcar.path[currentcar.locationindex]
                find_place_in_queue(currentcar, initroad, t)

            # The next event is a light changing
            elif next_event_time == next_light_time:
                t = next_light_time
                for i in range(12, 18):
                    roads[i].nextchange = t
                # Roads that change lights at this time are updated
                update_road_lights(t, roads)

            # The next event is a car starting to move
            elif next_event_time == next_start:
                t = next_start
                for i in range(12, 18):
                    roads[i].nextchange = t

                logging.debug(
                    f"{t}, Car {movingcar.identify}, Leave, "
                    f"Pos {movingcar.visits[movingcar.locationindex]}, "
                    f"Road {movingcar.path[movingcar.locationindex].roadid}"
                )

                # Car is the first in its queue
                if movingcar.place_in_queue == 0:
                    # Car's next arrival is set
                    # movingcar.next_sec_arrival = t + (
                    #     self.factors["distance"] / self.factors["speed"]
                    # )
                    # change the distance to by road
                    movingcar.next_sec_arrival = (
                        t
                        + self.factors["pause"]
                        + (
                            movingcar.path[movingcar.locationindex].road_length
                            / self.factors["speed"]
                        )
                    )

                # Car is not the first in its queue
                else:
                    # Car's next arrival time is set
                    movingcar.next_sec_arrival = t + (
                        self.factors["car_distance"]
                        + self.factors["carlength"] / self.factors["speed"]
                    )

                # Car leaves its current queue and is 'moving'
                movingcar.path[movingcar.locationindex].queue[
                    movingcar.place_in_queue
                ] = 0
                movingcar.moving = True
                movingcar.timewaiting += t - movingcar.prevstop
                movingcar.nextstart = outbounds
                next_start = outbounds

            # The next event is a car arriving within the system
            elif next_event_time == next_sec_arrival:
                t = next_sec_arrival
                for i in range(12, 18):
                    roads[i].nextchange = t

                # Car is first in its queue
                if arrivingcar.place_in_queue == 0:
                    # Car changes the road it is traveling on
                    arrivingcar.update_location()
                    currentroad = arrivingcar.path[arrivingcar.locationindex]
                    # Car is assigned its location and given a new start time
                    find_place_in_queue(arrivingcar, currentroad, t)
                # Car is not the first in its queue
                elif arrivingcar.place_in_queue is not None:
                    # Car moves up in its queue
                    currentroad = arrivingcar.path[arrivingcar.locationindex]
                    currentroad.queue[arrivingcar.place_in_queue] = 0
                    currentroad.queue[arrivingcar.place_in_queue - 1] = arrivingcar
                    arrivingcar.place_in_queue -= 1
                    # Current road has a green light
                    if currentroad.status:
                        arrivingcar.nextstart = t
                    # Current road has a red light
                    else:
                        arrivingcar.nextstart = currentroad.nextchange
                # Car is supposed to move up the queue, but it is not in the queue
                else:
                    error_msg = "Car is not in queue"
                    logging.error(error_msg)
                    raise Exception(error_msg)

                logging.debug(
                    f"{t}, Car {arrivingcar.identify}, Arrival, "
                    f"Pos {arrivingcar.visits[arrivingcar.locationindex]}, "
                    f"Road {arrivingcar.path[arrivingcar.locationindex].roadid}"
                )

                # Car is no longer 'moving'
                movingcar.moving = False
                arrivingcar.next_sec_arrival = outbounds
                next_sec_arrival = outbounds
                arrivingcar.prevstop = t
            elif next_event_time == next_car_gen:
                logging.warning("Unexpected car generation")
            else:
                error_msg = "Unexpected event time"
                logging.error(error_msg)
                raise Exception(error_msg)

            # Index to the next car (skip finished cars)
            carindex = min_car_index
            while carindex < len(cars) - 1 and cars[carindex].finished:
                carindex += 1
                min_car_index += 1
            min_sec_arrival = outbounds
            min_start = outbounds
            # Finds the next car to start moving and the next car to arrive
            while carindex < len(cars) - 1:
                testcar = cars[carindex]
                # Car is eligible to be the next starting car
                if (
                    min(next_start, testcar.nextstart) == testcar.nextstart
                    and testcar.nextstart != outbounds
                ):
                    min_start = testcar.nextstart
                    movingcar = testcar
                # Car is eligible to be the next arriving car
                if (
                    min(min_sec_arrival, testcar.next_sec_arrival)
                    == testcar.next_sec_arrival
                    and testcar.next_sec_arrival != outbounds
                ):
                    min_sec_arrival = testcar.next_sec_arrival
                    arrivingcar = testcar
                # Next car is tested and the next events are set
                carindex += 1
                next_sec_arrival = min_sec_arrival
                next_start = min_start

            # sumwait = 0
            # finishedcars = 0
            carindex = min_car_index
            while carindex < len(cars):
                car = cars[carindex]
                # If car has finished, record its stats
                if not car.finished and (car.locationindex == len(car.path) - 1):
                    logging.debug(
                        f"{t}, Car {car.identify}, Finish, "
                        f"Pos {car.visits[car.locationindex]}, "
                        f"Road {car.path[car.locationindex].roadid}"
                    )
                    car.finished = True
                    car.finishtime = t - car.initialarrival
                    finishedcars += 1
                    sumwait += car.timewaiting

                carindex += 1
            # Only update waiting stats if there are cars that have finished
            # Otherwise, we'll divide by zero
            if finishedcars > 0:
                avg_wait = sumwait / finishedcars
                # Only add to the list if the value is different from the last
                # Prevents adding a ton of duplicate values
                if avg_wait != last_avg_wait:
                    avg_wait_over_time[t] = avg_wait
                    last_avg_wait = avg_wait

            # record queue length of each road and update overflow status
            overflow_total[t] = 0
            for roadid in range(3 * self.factors["numintersections"]):
                # num of cars in queue
                cars_in_queue = sum([x != 0 for x in roads[roadid].queue])
                # queue length =
                #   (car_length+ car_distance) * num(cars in queue) - car_distance
                roads[roadid].queue_hist[t] = max(
                    0,
                    cars_in_queue
                    * (self.factors["car_distance"] + self.factors["carlength"])
                    - self.factors["car_distance"],
                )
                # if there is overflow status
                if roads[roadid].road_length <= roads[roadid].queue_hist[t]:
                    roads[roadid].overflow = True
                    overflow_total[t] = 1
                else:
                    roads[roadid].overflow = False

            # after queue len is calculated, deal with overflow when a road
            # is overflowed:
            #   1. cars in incoming roads cannot get in
            #   2. calculated oveflow len
            current_overflow_len = []
            for roadid in range(3 * self.factors["numintersections"]):
                if roads[roadid].overflow and len(roads[roadid].incoming_roads) > 0:
                    # calculate oeverflow queue lenth
                    overflow_queue = [
                        r.queue_hist[t] for r in roads[roadid].incoming_roads
                    ]
                    roads[roadid].overflow_queue[t] = sum(overflow_queue)
                    current_overflow_len.append(sum(overflow_queue))
                    # update start time of cars in queue of incoming roads
                    # start time of the last car in overflowed road
                    last_start = max(
                        [x.nextstart for x in roads[roadid].queue if x != 0]
                    )
                    for r in roads[roadid].incoming_roads:
                        # if there are cars in queue
                        for car_q in r.queue:
                            last_start += self.factors["reaction"]
                            if car_q != 0:
                                # TODO: figure out if these are supposed to be different
                                # first car in queue in the incoming road
                                if car_q.place_in_queue == 0:
                                    car_q.nextstart = max(car_q.nextstart, last_start)
                                # second and later car in queue in the incoming road
                                else:
                                    car_q.nextstart = max(car_q.nextstart, last_start)
            if len(current_overflow_len) > 0:
                overflow_len_total[t] = sum(current_overflow_len) / len(
                    current_overflow_len
                )
            else:
                overflow_len_total[t] = 0

        total_waiting = 0
        for car in cars:
            if car.finished:
                total_waiting += car.timewaiting
            else:
                if not car.moving:
                    car.timewaiting = t - car.prevstop
                total_waiting += car.timewaiting

        avg_wait_over_time[t] = total_waiting / len(cars)

        system_time = 0
        for car in cars:
            if car.finished:
                system_time += car.finishtime
            else:
                if not car.moving:
                    car.finishtime = t - car.initialarrival
                system_time += car.finishtime

        avg_system_time = system_time / len(cars)

        # compute ave queue length for each road
        avg_queue_length = 0
        overflow_percentage = []
        overflow_avg_len = []
        for roadid in range(3 * self.factors["numintersections"]):
            # use dictionary to get list of t and queue_len
            # (t - t-1)*queue_len/max(t)
            queue_len = list(roads[roadid].queue_hist.values())
            queue_time = list(roads[roadid].queue_hist.keys())
            time_dif = [
                queue_time[i + 1] - queue_time[i] for i in range(len(queue_time) - 1)
            ]
            avg_queue = sum([x * y for x, y in zip(queue_len[:-1], time_dif)]) / t
            avg_queue_length = avg_queue_length + avg_queue
            # overflow queue time of each road
            overflow_ind = [1 * (q >= roads[roadid].road_length) for q in queue_len]
            # Overflow_total.append(overflow_ind[:-1])
            overflow_duration = sum(
                [x * y for x, y in zip(overflow_ind[:-1], time_dif)]
            )
            overflow_perc = overflow_duration / t * 100
            overflow_percentage.append(overflow_perc)
            # overflow queue length
            # never overflow
            if overflow_duration == 0:
                overflow_len_dur = 0
            else:
                overflow_len = list(roads[roadid].overflow_queue.values())
                overflow_len_sum = sum(
                    [x * y for x, y in zip(overflow_len[:-1], time_dif)]
                )
                overflow_len_dur = overflow_len_sum / overflow_duration
            overflow_avg_len.append(overflow_len_dur)

        # total overflow index
        overflow_total_ind = list(overflow_total.values())
        overflow_total_time = list(overflow_total.keys())
        overflow_total_len = list(overflow_len_total.values())
        time_dif = [
            overflow_total_time[i + 1] - overflow_total_time[i]
            for i in range(len(overflow_total_time) - 1)
        ]
        overflow_system_duration = sum(
            [x * y for x, y in zip(overflow_total_ind[:-1], time_dif)]
        )
        overflow_system_perc = overflow_system_duration / t * 100
        overflow_system_perc_over_51 = not overflow_system_perc < 51
        # overflow queue length
        if overflow_system_duration == 0:
            overflow_avg_len_system = 0
        else:
            overflow_system_len = sum(
                [x * y for x, y in zip(overflow_total_len[:-1], time_dif)]
            )
            overflow_avg_len_system = overflow_system_len / overflow_system_duration

        # average queue
        avg_queue_length = avg_queue_length / 3 * self.factors["numintersections"]

        avg_total = total_waiting / (len(cars) if cars else 0)

        # Compose responses and gradients.
        responses = {
            "AvgWaitTime": avg_total,
            "AvgWaitTimeOverTime": avg_wait_over_time,
            "SystemTime": avg_system_time,
            "AvgQueueLen": avg_queue_length,
            "OverflowPercentage": overflow_system_perc,
            "OverflowPercentageOver51": overflow_system_perc_over_51,
            "OverflowAveLen": overflow_avg_len_system,
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class MinWaitingTime(Problem):
    """Minimum waiting time problem."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "TRAFFICCONTROL-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Waiting Time for Traffic Light"

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
        return ConstraintType.BOX

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS

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
        return {"runtime": 50}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"decision_vector"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (1, 1, 1),
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 125,
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
        return 3

    @classproperty
    @override
    def lower_bounds(cls) -> tuple:
        return (0,) * cls.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        bound = min(
            self.model.factors["redlight_arteries"]
            + self.model.factors["redlight_veins"]
        )
        return (bound,) * self.dim

    def __init__(
        self,
        name: str = "TRAFFICCONTROL-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the MinWaitingTime problem.

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
            model=TrafficLight,
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"decision_vector": vector}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        # TODO: change this to decision variable
        return tuple(factor_dict["decision_vector"])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["AvgWaitTime"],)

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim,)
        return det_objectives, det_objectives_gradients

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple(
            [
                rand_sol_rng.uniform(
                    0,
                    min(
                        self.model.factors["redlight_arteries"]
                        + self.model.factors["redlight_veins"]
                    ),
                )
                for _ in range(self.dim)
            ]
        )
