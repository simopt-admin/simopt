"""
Summary
-------
Simulate a single day of operation for an amusement park queuing problem.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/amusementpark.html>`_.
"""
import numpy as np
import math as math

from ..base import Model, Problem


class AmusementPark(Model):
    """
    A model that simulates a single day of operation for an
    amusement park queuing problem based on a poisson distributed tourist
    arrival rate, a next attraction transition matrix, and attraction
    durations based on an Erlang distribution. Returns the total number
    and percent of tourists to leave the park due to full queues.

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

    Arguments
    ---------
    fixed_factors : dict
        fixed_factors of the simulation model

    See also
    --------
    base.Model
    """

    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "AMUSEMENTPARK"
        self.n_rngs = 3
        self.n_responses = 4
        self.factors = fixed_factors
        self.specifications = {
            "park_capacity": {
                "description": "The total number of tourists waiting for \
                                attractions that can be maintained through \
                                park facilities, distributed across the attractions.",
                "datatype": int,
                "default": 350
            },
            "number_attractions": {
                "description": "The number of attractions in the park.",
                "datatype": int,
                "default": 7
            },
            "time_open": {
                "description": "The number of minutes per day the park is open.",
                "datatype": float,
                "default": 480.0
            },
            "erlang_shape": {
                "description": "The shape parameter of the Erlang distribution for each attraction"
                               "duration.",
                "datatype": list,
                "default": [2, 2, 2, 2, 2, 2, 2]
            },
            "erlang_scale": {
                "description": "The rate parameter of the Erlang distribution for each attraction"
                               "duration.",
                "datatype": list,
                "default": [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9]
            },
            "queue_capacities": {
                "description": "The capacity of the queue for each attraction \
                                based on the portion of facilities allocated.",
                "datatype": list,
                "default": [50, 50, 50, 50, 50, 50, 50]
            },
            "depart_probabilities": {
                "description": "The probability that a tourist will depart the \
                                park after visiting an attraction.",
                "datatype": list,
                "default": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
            },
            "arrival_gammas": {
                "description": "The gamma values for the poisson distributions dictating the rates at which \
                                tourists entering the park arrive at each attraction",
                "datatype": list,
                "default": [1, 1, 1, 1, 1, 1, 1]
            },
            "transition_probabilities": {
                "description": "The transition matrix that describes the probability \
                                of a tourist visiting each attraction after their current attraction.",
                "datatype": list,
                "default": [[0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                            [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                            [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                            [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                            [0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.3],
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0, 0.3],
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]]
            }
        }

        self.check_factor_list = {
            "park_capacity": self.check_park_capacity,
            "number_attractions": self.check_number_attractions,
            "time_open": self.check_time_open,
            "queue_capacities": self.check_queue_capacities,
            "depart_probabilities": self.check_depart_probabilities,
            "arrival_gammas": self.check_arrival_gammas,
            "transition_probabilities": self.check_transition_probabilities,
            "erlang_shape": self.check_erlang_shape,
            "erlang_scale": self.check_erlang_scale
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    # Check for simulatable factors.
    def check_park_capacity(self):
        return self.factors["park_capacity"] > 0

    def check_number_attractions(self):
        return self.factors["number_attractions"] > 0

    def check_time_open(self):
        return self.factors["time_open"] >= 0

    def check_queue_capacities(self):
        return all([cap >= 0 for cap in self.factors["queue_capacities"]])

    def check_depart_probabilities(self):
        if len(self.factors["depart_probabilities"]) != self.factors["number_attractions"]:
            return False
        else:
            return all([0 <= prob <= 1 for prob in self.factors["depart_probabilities"]])

    def check_arrival_gammas(self):
        if len(self.factors["arrival_gammas"]) != self.factors["number_attractions"]:
            return False
        else:
            return all([gamma >= 0 for gamma in self.factors["arrival_gammas"]])

    # Check if transition matrix has same number of rows and columns and that each row + depart probability sums to 1.
    def check_transition_probabilities(self):
        transition_sums = list(map(sum, self.factors["transition_probabilities"]))
        if all([len(row) == len(self.factors["transition_probabilities"]) for row in self.factors["transition_probabilities"]]) & \
                all(transition_sums[i] + self.factors["depart_probabilities"][i] == 1 for i in range(self.factors["number_attractions"])):
            return True
        else:
            return False

    def check_erlang_shape(self):
        if len(self.factors["erlang_shape"]) != self.factors["number_attractions"]:
            return False
        else:
            return all([gamma >= 0 for gamma in self.factors["erlang_shape"]])

    def check_erlang_scale(self):
        if len(self.factors["erlang_scale"]) != self.factors["number_attractions"]:
            return False
        else:
            return all([gamma >= 0 for gamma in self.factors["erlang_scale"]])

    def check_simulatable_factors(self):
        return sum(self.factors["queue_capacities"]) <= self.factors["park_capacity"]

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "total_departed_tourists": The total number of tourists to leave the park due
                to full queues
            "percent_departed_tourists": The percentage of tourists to leave the park due
                to full queues
        """
        # Designate random number generators.
        arrival_rng = rng_list[0]
        transition_rng = rng_list[1]
        time_rng = rng_list[2]

        # Initiate clock variables for statistics tracking and event handling.
        clock = 0
        previous_clock = 0
        next_arrival = arrival_rng.expovariate(sum(self.factors["arrival_gammas"]))

        # initialize list of attractions to be selected upon arrival.
        potential_attractions = range(self.factors["number_attractions"])

        # create list of each attraction's next completion time and initialize to infinity.
        completion_times = [math.inf for _ in range(self.factors["number_attractions"])]

        # initialize actual queues.
        queues = [0 for _ in range(self.factors["number_attractions"])]

        # create external arrival probabilities for each attraction.
        arrival_probabalities = [self.factors["arrival_gammas"][i] / sum(self.factors["arrival_gammas"]) for i in
                                 self.factors["arrival_gammas"]]

        # Initialize quantities to track:
        total_visitors = 0
        total_departed = 0
        # initialize time average and utilization quantities.
        in_system = 0
        time_average = 0
        cumulative_util = [0 for _ in range(self.factors["number_attractions"])]

        # Run simulation over time horizon.
        while min(next_arrival, min(completion_times)) < self.factors["time_open"]:
            # Count number of tourists on attractions and in queues.
            clock = min(next_arrival, min(completion_times))
            riders = 0
            for i in range(self.factors["number_attractions"]):
                if completion_times[i] != math.inf:
                    riders += 1
                    cumulative_util[i] += (clock - previous_clock)
            in_system = sum(queues) + riders
            time_average += in_system * (clock - previous_clock)

            previous_clock = clock
            if next_arrival < min(completion_times):  # Next event is external tourist arrival.
                total_visitors += 1
                # Select attraction.
                attraction_selection = arrival_rng.choices(population=potential_attractions,
                                                           weights=arrival_probabalities)[0]
                # Check if attraction is currently available.
                # If available, arrive at that attraction. Otherwise check queue.
                if completion_times[attraction_selection] == math.inf:
                    # Generate completion time if attraction available.
                    completion_times[attraction_selection] = next_arrival + time_rng.gammavariate(alpha=self.factors["erlang_shape"][attraction_selection],
                                                                                                  beta=self.factors["erlang_scale"][attraction_selection])
                # If unavailable, check if current queue is less than capacity. If queue is not full, join queue.
                elif queues[attraction_selection] < self.factors["queue_capacities"][attraction_selection]:
                    queues[attraction_selection] += 1
                # If queue is full, leave park + 1.
                else:
                    total_departed += 1
                # Use superposition of Poisson processes to generate next arrival time.
                next_arrival += arrival_rng.expovariate(sum(self.factors["arrival_gammas"]))

            else:  # Next event is the completion of an attraction.
                finished_attraction = completion_times.index(min(completion_times))  # Identify finished attraction.
                # Check if there is a queue for that attraction.
                # If so then start new completion time and subtract 1 from queue.
                if queues[finished_attraction] > 0:
                    completion_times[finished_attraction] = min(completion_times) + time_rng.gammavariate(alpha=self.factors["erlang_shape"][finished_attraction],
                                                                                                          beta=self.factors["erlang_scale"][finished_attraction])
                    queues[finished_attraction] -= 1
                else:  # If no one in queue, set next completion of that attraction to infinity.
                    completion_times[finished_attraction] = math.inf

                # Check if that person will leave the park.
                potential_destinations = range(self.factors["number_attractions"] + 1)
                next_destination = transition_rng.choices(population=potential_destinations,
                                                          weights=self.factors["transition_probabilities"][finished_attraction] + [self.factors["depart_probabilities"][finished_attraction]]
                                                          )[0]

                # Check if tourist leaves park.
                if next_destination != potential_destinations[-1]:
                    # Check if attraction is currently available.
                    # If available, arrive at that attraction. Otherwise check queue.
                    if completion_times[next_destination] == math.inf:
                        # Generate completion time if attraction available.
                        completion_times[next_destination] = min(completion_times) + time_rng.gammavariate(alpha=self.factors["erlang_shape"][finished_attraction],
                                                                                                           beta=self.factors["erlang_scale"][finished_attraction])
                    # if unavailable, check if current queue is less than capacity. If queue is not full, join queue.
                    elif queues[next_destination] < self.factors["queue_capacities"][next_destination]:
                        queues[next_destination] += 1
                    # If queue is full, leave park + 1.
                    else:
                        total_departed += 1
        # End of simulation.

        # Calculate overall percent utilization calculation for each attraction.
        for i in range(self.factors["number_attractions"]):
            cumulative_util[i] = cumulative_util[i] / self.factors["time_open"]

        # Calculate responses from simulation data.
        responses = {"total_departed": total_departed,
                     "percent_departed": total_departed / total_visitors,
                     "average_number_in_system": time_average / self.factors["time_open"],
                     "attraction_utilization_percentages": cumulative_util
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in
                     responses}
        return responses, gradients


"""
Summary
-------
Minimize the total departed tourists.
"""


class AmusementParkMinDepart(Problem):
    """
    Class to make amusement park simulation-optimization problems.

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
    rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
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
    def __init__(self, name="AMUSEMENTPARK-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "deterministic"
        self.variable_type = "discrete"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"queue_capacities"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (344, 1, 1, 1, 1, 1, 1)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 100
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = AmusementPark(self.model_fixed_factors)
        self.dim = self.model.factors["number_attractions"]
        self.lower_bounds = tuple(0 for _ in range(self.model.factors["number_attractions"]))
        self.upper_bounds = tuple(self.model.factors["park_capacity"] for _ in range(self.model.factors["number_attractions"]))

    def vector_to_factor_dict(self, vector):
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
        factor_dict = {
            "queue_capacities": vector[:],
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
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
        vector = tuple(factor_dict["queue_capacities"])
        return vector

    def response_dict_to_objectives(self, response_dict):
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
        objectives = (response_dict["total_departed"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
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
        stoch_constraints = None
        return stoch_constraints

    def deterministic_objectives_and_gradients(self, x):
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
        det_objectives_gradients = None
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self, x):
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
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x):
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
        capacity_feasible = sum(x) == self.model.factors["park_capacity"]
        return box_feasible * capacity_feasible

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.mrg32k3a.MRG32k3a
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = tuple(rand_sol_rng.integer_random_vector_from_simplex(n_elements=self.model.factors["number_attractions"],
                                                                  summation=self.model.factors["park_capacity"],
                                                                  with_zero=False))
        return x
