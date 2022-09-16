"""
Summary
-------
Simulate the expected cost of a bike sharing system in different days.
A detailed description of the model/problem can be found
`here <TODO: no documentation>`_.
"""
import numpy as np
import copy
from scipy.spatial import distance_matrix

import heapq

from ..base import Model, Problem


class BikeShare(Model):
    """
    A model that simulates a day of bike sharing program. Returns the
    total cost of distribution, total penalty incurred during the operation hours.

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
    ----------
    fixed_factors : dict
        fixed_factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "BIKESHARING"
        self.n_rngs = 4 # TODO: number of rng used in the model
        self.n_responses = 1 # TODO: modify if more responses are added
        self.factors = fixed_factors 

        locations = [[i, j] for i in range(15) for j in range(15)]
        dist_mat = distance_matrix(locations, locations, p=1)

        # locations = [[i, j] for i in range(2) for j in range(2)]
        # dist_mat = distance_matrix(locations, locations, p=1)

        self.specifications = {
            "num_bikes": {
                "description": "total number of bikes in the city",
                "datatype": int,
                "default": 3200
            },
            "num_stations": {
                "description": "total number of stations in the city",
                "datatype": int,
                "default": 225
            },
            "num_bikes_start":{
                "description": "(decision var) number of bikes to start at each station at the beginning of the day",
                "datatype": list,
                "default": [14] * 175 + [15] * 50
            },
            "day_length": {
                "description": "the length of a day in operation in hours",
                "datatype": int,
                "default": 16
            },
            "station_capacities": {
                "description": "the capacity of each corresponding stations",
                "datatype": list,
                "default": [20] * 225 
            },
            "empty_penalty_constant": {
                "description": "the penalty constant for when a station has no bike",
                "datatype": float,
                "default": 50.0
            },
            "full_penalty_constant": {
                "description": "the penalty constant for when a station is full",
                "datatype": float,
                "default": 50.0
            },
            "arrival_rates": {
                "description": "user arrival rates to each corresponding stations (in this model, we assume a homogeneous Poisson process for each station)",
                "datatype": list,
                "default": [1/6] * 225 # mean interarrival time = 10min for each station
            },
            "gamma_mean_const": {
                "description": "scalar for the mean time it takes the user to return the bike",
                "datatype": float,
                "default": 1/3
            },
            "gamma_variance_const": {
                "description": "scalar for the variance of time it takes the user to return the bike",
                "datatype": float,
                "default": 1/144
            },
            "gamma_mean_const_s": {
                "description": "mean time it takes the user to return bike to the same station",
                "datatype": float,
                "default": 3/4
            },
            "gamma_variance_const_s": {
                "description": "variance for time it takes the user to return bike to the same station",
                "datatype": float,
                "default": 49/60
            },
            "rebalancing_constant": {
                "description": "constant multiple for the cost of rebalancing bikes",
                "datatype": float,
                "default": 5
            },
            "distance": {
                "description": "An s x s matrix containing distance from each pair of stations",
                "datatype": list,
                "default": dist_mat 
            }
        }

        self.check_factor_list = {
            "num_bikes": self.check_num_bikes,
            "num_stations": self.check_num_stations,
            "num_bikes_start": self.check_num_bikes_start,
            "day_length": self.check_day_length,
            "station_capacities": self.check_station_capacities,
            "empty_penalty_constant": self.check_empty_penalty_constant,
            "full_penalty_constant": self.check_full_penalty_constant,
            "arrival_rates": self.check_arrival_rates,
            "gamma_mean_const": self.check_gamma_mean_const,
            "gamma_variance_const": self.check_gamma_variance_const,
            "gamma_mean_const_s": self.check_gamma_mean_const_s,
            "gamma_variance_const_s": self.check_gamma_variance_const_s,
            "rebalancing_constant": self.check_rebalancing_constant,
            "distance": self.check_distance,
        }
        # Set factors of the simulation model
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_num_bikes(self):
        return self.factors["num_bikes"] > 0

    def check_num_stations(self):
        return self.factors["num_stations"] > 0

    def check_num_bikes_start(self):
        return all(rates > 0 for rates in self.factors["num_bikes_start"])

    def check_day_length(self):
        return self.factors["day_length"] >= 0 and self.factors["day_length"] <= 24

    def check_station_capacities(self):
        return all (cap >= 0 for cap in self.factors["station_capacities"])

    def check_empty_penalty_constant(self):
        return self.factors["empty_penalty_constant"] > 0

    def check_full_penalty_constant(self):
        return self.factors["full_penalty_constant"] > 0

    def check_arrival_rates(self):
        return all(rates > 0 for rates in self.factors["arrival_rates"])

    def check_gamma_mean_const(self):
        return self.factors["gamma_mean_const"] > 0

    def check_gamma_variance_const(self):
        return self.factors["gamma_variance_const"] > 0

    def check_gamma_mean_const_s(self):
        return self.factors["gamma_mean_const_s"] > 0

    def check_gamma_variance_const_s(self):
        return self.factors["gamma_variance_const_s"] > 0

    def check_rebalancing_constant(self):
        return self.factors["rebalancing_constant"] > 0

    def check_distance(self):
        return True # TODO: check distances

    def replicate(self, rng_list):
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
            "total cost" = The total operations cost over a running day
        """

        events = {0: "arrive", 1: "return"} # list of events: 0 indexing arrival and 1 indexing return

        t = 0
        event_list = [] # [time, event, station]

        # print(self.factors["num_bikes_start"])
        num_bikes = copy.deepcopy(self.factors["num_bikes_start"])
        target_num_bikes = copy.deepcopy(self.factors["num_bikes_start"])
        capacity = self.factors["station_capacities"]
        arrival_rates = self.factors["arrival_rates"]

        # switches indicating whether a station is empty or full or not
        empty_since = [-1] * self.factors["num_stations"]
        full_since = [-1] * self.factors["num_stations"]

        penalty_full = 0
        penalty_empty = 0

        # Generate the first event for each station in a day
        for i, rate in enumerate(arrival_rates):
            int_arr_time = rng_list[3].expovariate(rate)
            event_list.append([int_arr_time, 0, i])

        # Simulate a working day
        while t <= self.factors["day_length"]:
            
            event_list.sort(key = lambda x:x[0])
            # print("events:", event_list)
            # print("full since", full_since)
            # print("empty since", empty_since)
            # print("num bikes", num_bikes)
            
            t, event, station = event_list.pop(0)
            # print(t, events[event], station)

            # Arrival Event
            if event == 0:
                # No bikes in the station
                if num_bikes[station] -1 < 1:
                    num_bikes[station] -= 1  
                    empty_since[station] = t # customer is lost, start counting penalty hours
                elif num_bikes[station] == capacity[station]:
                    assert full_since[station] >= 0
                    penalty_full += self.factors["full_penalty_constant"] * (t - full_since[station])
                    full_since[station] = -1 
                else:
                    num_bikes[station] -= 1  
                    station_to = int(rng_list[0].random() * self.factors["num_stations"])
                    if station_to != station:
                        dist = self.factors["distance"][station][station_to]
                        mean = self.factors["gamma_mean_const"] * dist
                        var = self.factors["gamma_variance_const"] * dist
                        time_out = dist * rng_list[1].gammavariate(mean**2/var * 0.0001, mean/var) #TODO: check if this is correct
                    else:
                        mean = self.factors["gamma_mean_const_s"]
                        var = self.factors["gamma_variance_const_s"]
                        time_out = rng_list[2].gammavariate(mean**2/var * 0.0001, mean/var)
                    # print("return time:", t + time_out)
                    if (t + time_out) < self.factors["day_length"]:
                        event_list.append([t+time_out, 1, station_to])
                # Generate the next arrival for this station
                int_arr_time = rng_list[3].expovariate(arrival_rates[station])

                event_list.append([t+int_arr_time, 0, station])

            # Return Event
            if event == 1:
                if num_bikes[station] + 1 == capacity[station]:
                    num_bikes[station] += 1
                    full_since[station] = t
                elif full_since[station] >= 0:
                    dist = self.factors["distance"][station][station_to]
                    mean = self.factors["gamma_mean_const"] * dist
                    var = self.factors["gamma_variance_const"] * dist
                    time_out = dist * rng_list[1].gammavariate(mean**2/var * 0.0001, mean/var)
                    station_to = station + 1
                    event_list.append([t+time_out, 1, station_to])
                elif num_bikes[station] == 0:
                    assert empty_since[station] >= 0
                    penalty_empty += self.factors["empty_penalty_constant"] * (t - empty_since[station])
                    empty_since[station] = -1
                else:
                    num_bikes[station] += 1
        
        print("End simulation, start surplus calculation")
        # print(self.factors["num_bikes_start"])
        # print("target", target_num_bikes)
        distribution_cost = 0
        surplus_pointer = 0
        lack_pointer = 0

        # print(num_bikes, len(num_bikes), sum(num_bikes), self.factors["num_bikes_start"])
        print("Num bikes", sum(num_bikes))
        # Calculate the redistribution cost
        while surplus_pointer < self.factors["num_stations"] and lack_pointer < self.factors["num_stations"]:
            print("surplus pointer", surplus_pointer)
            if num_bikes[surplus_pointer] > target_num_bikes[surplus_pointer]:
                surplus = num_bikes[surplus_pointer] - target_num_bikes[surplus_pointer]
                print("surplus", surplus)
                while surplus > 0 and lack_pointer < self.factors["num_stations"]:
                    print("lack pointer", lack_pointer)
                    print(num_bikes[lack_pointer], target_num_bikes[lack_pointer])
                    if num_bikes[lack_pointer] < target_num_bikes[lack_pointer]:
                        need = target_num_bikes[lack_pointer] - num_bikes[lack_pointer]
                        print("need", need)
                        # station needs more than the surplus
                        if need >= surplus:
                            num_distribute = surplus 
                            surplus = 0
                            num_bikes[lack_pointer] += surplus
                        else:
                            num_distribute = need 
                            surplus -= need 
                            lack_pointer += 1
                        print("debug checker", self.factors["distance"][surplus_pointer][lack_pointer], num_distribute)
                        print("* redistribution cost", self.factors["distance"][surplus_pointer][lack_pointer] * \
                            self.factors["rebalancing_constant"] * num_distribute)
                        distribution_cost += self.factors["distance"][surplus_pointer][lack_pointer] * \
                            self.factors["rebalancing_constant"] * num_distribute
                    else:
                        lack_pointer += 1
            surplus_pointer += 1
        penalty = penalty_empty + penalty_full

        print(penalty, distribution_cost)

        responses = {"cost": penalty + distribution_cost}
        gradient = {} # TODO: implement gradient

        return responses, gradient
       

"""
Summary
-------
Minimize the cost of operation of bike sharing in a city.
"""


class BikeShareMinCost(Problem):
    """
    Class to make bike sharing simulation-optimization problems.

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
    optimal_value : float
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
    def __init__(self, name="BIKESHARE-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        # self.dim = 225
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "deterministic"
        self.variable_type = "discrete"
        # self.lower_bounds = tuple(np.zeros(225))
        # self.upper_bounds = tuple([30] * 225)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"num_bikes_start"} 
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": tuple([14] * 175 + [15] * 50)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000
            },
            "": {}
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = BikeShare(self.model_fixed_factors)

        self.dim = self.model.factors["num_bikes"]
        self.lower_bounds = tuple(np.zeros(self.model.factors["num_bikes"]))
        self.upper_bounds = tuple("station_capacities")

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
            "num_bikes_start": vector[0]
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
        vector = (factor_dict["num_bikes_start"])
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
        objectives = (response_dict["cost"],)
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
        det_objectives_gradients = ((0),) # TODO: debug checks
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
        return box_feasible and sum(x) == self.model.factors["num_bikes"]

    def get_random_solution(self, rand_sol_rng):
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
        x = tuple(rand_sol_rng.integer_random_from_simplex(self.model.factors["num_stations"], self.model.factors["num_bikes"]))
        return x