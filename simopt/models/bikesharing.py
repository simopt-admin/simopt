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
from decimal import * 

from ..base import Model, Problem


class BikeShare(Model):
    """
    A model that simulates a day of bike sharing program. Returns 
    total penalty incurred during the operation hours.

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

        self.specifications = {
            "map_dim":{
                "description": "dimsion of the grid map",
                "datatype": int,
                "default": 5
            },
            "num_bikes": {
                "description": "total number of bikes in the city",
                "datatype": int,
                "default": 375 #3200
            },
            "num_bikes_start":{
                "description": "(decision var) number of bikes to start at each station at the beginning of the day",
                "datatype": list,
                "default": tuple([15] * 25)
            },
            "day_length": {
                "description": "the length of a day in operation in hours",
                "datatype": int,
                "default": 16
            },
            "station_capacities": {
                "description": "the capacity of each corresponding stations",
                "datatype": list,
                "default": 18
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
            "gamma_mean_const": {
                "description": "scalar for the mean time it takes the user to return the bike",
                "datatype": float,
                "default": 1/3
            },
            "gamma_variance_const": {
                "description": "scalar for the variance of time it takes the user to return the bike",
                "datatype": float,
                "default": 1/12
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
            }
        }

        self.check_factor_list = {
            "map_dim": self.check_map_dim,
            "num_bikes": self.check_num_bikes,
            "num_bikes_start": self.check_num_bikes_start,
            "day_length": self.check_day_length,
            "station_capacities": self.check_station_capacities,
            "empty_penalty_constant": self.check_empty_penalty_constant,
            "full_penalty_constant": self.check_full_penalty_constant,
            "gamma_mean_const": self.check_gamma_mean_const,
            "gamma_variance_const": self.check_gamma_variance_const,
            "gamma_mean_const_s": self.check_gamma_mean_const_s,
            "gamma_variance_const_s": self.check_gamma_variance_const_s,
            "rebalancing_constant": self.check_rebalancing_constant
        }
        # Set factors of the simulation model
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_map_dim(self):
        return self.factors["map_dim"] > 0

    def check_num_bikes(self):
        return self.factors["num_bikes"] > 0

    def check_num_bikes_start(self):
        return all(rates > 0 for rates in self.factors["num_bikes_start"])

    def check_day_length(self):
        return self.factors["day_length"] >= 0 and self.factors["day_length"] <= 24

    def check_station_capacities(self):
        return self.factors["station_capacities"] >= 0

    def check_empty_penalty_constant(self):
        return self.factors["empty_penalty_constant"] > 0

    def check_full_penalty_constant(self):
        return self.factors["full_penalty_constant"] > 0

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

        def gen_arrival_rate(alpha = 4):
            """
            Return the time-dependent arrival rates of bikeville in the 
            morning, noon, and evening
            """
            dim = self.factors["map_dim"]
            morning = np.ones(shape=(dim, dim))
            noon = np.ones(shape=(dim, dim)) * 2
            evening = np.ones(shape=(dim, dim))

            for i in range(dim):
                for j in range(dim):
                    morning[i, j] = alpha * (np.abs(i - dim//2) + np.abs(j - dim//2)) / dim
                    evening[i, j] = alpha - alpha * (np.abs(i - dim//2) + np.abs(j - dim//2)) / dim
            # print(morning, evening)
            # print(morning, noon, evening)
            return morning.flatten(), noon.flatten(), evening.flatten()
        
        def gen_distance():
            """
            Returns:
                list[list]: adjacency matrix containing distance between 
                each pair of stations
            """
            dim = self.factors["map_dim"]
            locations = [[i, j] for i in range(dim) for j in range(dim)]
            dist_mat = distance_matrix(locations, locations, p=1)
            return dist_mat
        
        def alias_init(dist):
            """
            Initialize the alias method (Adapted from Joe's implementation)
            (referencing https://github.com/asmith26/Vose-Alias-Method/blob/main/vose_sampler/vose_sampler.py).
            Parameters
            ---------
            dist : 'dictionary'
                A probability distribution for discrete weighted random variables that maps the values to their probabilities.
            Returns
            -------
            table_prob: dictionary
                table of probabilities
            table_alias : dictionary
                table of alias
            """
            n = len(dist)
            table_prob = {}
            table_alias = {}
            small = [] # stack for probabilities smaller that 1
            large = [] # stack for probabilities greater than or equal to 1

            # Construct and sort the scaled probabilities into their appropriate stacks
            for val, prob in dist.items():
                table_prob[val] = Decimal(prob) * n
                if table_prob[val] < 1:
                    small.append(val)
                else:
                    large.append(val)

            # Construct the probability and alias tables
            while small and large:
                l = small.pop()
                g = large.pop()
                table_alias[l] = g
                table_prob[g] = (table_prob[g] + table_prob[l] - Decimal(1))
                if table_prob[g] < 1:
                    small.append(g)
                else:
                    large.append(g)

            # The remaining outcomes (of one stack) must have probability 1
            while large:
                table_prob[large.pop()] = Decimal(1)

            while small:
                table_prob[small.pop()] = Decimal(1)
            return table_prob, table_alias


        def alias(table_prob, table_alias):
            """Generate a discrete random variate in constant time.
            Parameters
            ---------
            table_prob : dictionary
                table of probabilities
            table_alias : dictionary
                table of alias
            Returns
            -------
            int
                a discrete random variate from the specified distribution.
            """
            # Determine which column of table_prob to inspect
            i = int(np.floor(np.random.rand() * len(table_prob)))
            # Determine which outcome to pick in that column
            if np.random.rand() < table_prob[i]:
                return i
            else:
                return table_alias[i]

        t = 0
        event_list = [] # [time, event, station]; event: 0 indexing arrival and 1 indexing return
        
        num_stations = self.factors["map_dim"] ** 2

        num_bikes = np.array(copy.deepcopy(self.factors["num_bikes_start"]))
        capacity = [self.factors["station_capacities"]] * num_stations 
        morning_arrival_rates, arrival_rates, evening_arrival_rates = gen_arrival_rate()
        distance = gen_distance()
        
        # Generate prob for alias method 
        morning_prob, evening_prob = {}, {}
        norm_morn = morning_arrival_rates/sum(morning_arrival_rates)
        norm_even = evening_arrival_rates/sum(evening_arrival_rates)
        for i in range(len(arrival_rates)):
            morning_prob[i] = norm_even[i]
            evening_prob[i] = norm_morn[i]
        morn_table_prob, morn_table_alias = alias_init(morning_prob)
        even_table_prob, even_table_alias = alias_init(evening_prob)
        
        # Generate morning, mid-day, evening division
        day_length = self.factors["day_length"]
        morning = int(day_length * (1/3))
        mid_day = morning * 2
        
        empty_count = 0 # Number of times a customer arrives and find station empty
        full_count = 0 # Number of times a customer returns and find station full
        grad = [0] * num_stations

        # Generate the first arrival event for each station in a day
        for i, rate in enumerate(morning_arrival_rates):
            int_arr_time = rng_list[3].expovariate(rate)
            event_list.append([int_arr_time, 0, i])

        # Simulate a work day
        while t <= self.factors["day_length"]:
        
            event_list.sort(key = lambda x:x[0])
            t, event, station = event_list.pop(0)

            # Arrival Event
            if event == 0:
                # No bikes in the station
                if num_bikes[station] < 1:
                    empty_count += 1 # customer is lost, empty count increment
                    grad[station] -= 1
                else:
                    num_bikes[station] -= 1  
                    if t < morning:
                        station_to = alias(morn_table_prob, morn_table_alias)
                    elif t < mid_day:
                        station_to = int(rng_list[0].random() * num_stations) if int(rng_list[0].random() * num_stations) < num_stations else num_stations - 1
                    else:
                        station_to = alias(even_table_prob, even_table_alias)
                    if station_to != station:
                        dist = distance[station][station_to]
                        mean = self.factors["gamma_mean_const"]
                        var = self.factors["gamma_variance_const"] 
                        time_out = dist * rng_list[1].gammavariate(mean**2/var, var/mean) 
                    else:
                        mean = self.factors["gamma_mean_const_s"]
                        var = self.factors["gamma_variance_const_s"]
                        time_out = rng_list[1].gammavariate(mean**2/var, var/mean)
                    if (t + time_out) < self.factors["day_length"]:
                        event_list.append([t+time_out, 1, station_to])
                # Different arrival rates during the day
                if t <= morning:
                    int_arr_time = rng_list[3].expovariate(morning_arrival_rates[station])
                elif t <= mid_day:
                    int_arr_time = rng_list[3].expovariate(arrival_rates[station])
                else:
                    int_arr_time = rng_list[3].expovariate(evening_arrival_rates[station])
                event_list.append([t+int_arr_time, 0, station])

            # Return Event
            if event == 1:
                try:
                    assert num_bikes[station] <= capacity[station]
                except: # Debug check
                    print(num_bikes)
                    1/0
                if num_bikes[station] == capacity[station]:
                    full_count += 1
                    grad[station] += 1
                    new_station_to = station + 1 if station < num_stations/2 else station - 1
                    assert new_station_to < num_stations
                    dist = distance[station][new_station_to]
                    mean = self.factors["gamma_mean_const"] * dist
                    var = self.factors["gamma_variance_const"] * dist
                    time_out = dist * rng_list[1].gammavariate(mean**2/var, var/mean)
                    event_list.append([t+time_out, 1, new_station_to])
                else:
                    num_bikes[station] += 1
        
        ##### We have decided to ignore the distribution cost for now
        ##### If needed, please comment this section back
        # # Calculate the redistribution cost
        # distribution_cost = 0
        # surplus_pointer = 0
        # lack_pointer = 0
        # while surplus_pointer < num_stations and lack_pointer < num_stations:
        #     if num_bikes[surplus_pointer] > target_num_bikes[surplus_pointer]:
        #         surplus = num_bikes[surplus_pointer] - target_num_bikes[surplus_pointer]
        #         while surplus > 0 and lack_pointer < num_stations:
        #             if num_bikes[lack_pointer] < target_num_bikes[lack_pointer]:
        #                 need = target_num_bikes[lack_pointer] - num_bikes[lack_pointer]
        #                 # station needs more than the surplus
        #                 if need >= surplus:
        #                     num_distribute = surplus 
        #                     surplus = 0
        #                     num_bikes[lack_pointer] += surplus
        #                 else:
        #                     num_distribute = need 
        #                     surplus -= need 
        #                     lack_pointer += 1
        #                 distribution_cost += distance[surplus_pointer][lack_pointer] * \
        #                     self.factors["rebalancing_constant"] * num_distribute
        #             else:
        #                 lack_pointer += 1
        #     surplus_pointer += 1

        empty_penalty = self.factors["empty_penalty_constant"]
        full_penalty = self.factors["full_penalty_constant"]
        penalty = empty_penalty * empty_count + full_penalty * full_count

        responses = {"cost": penalty}
        gradient = {"cost": {"num_bikes_start": grad}}
        
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
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "deterministic"
        self.variable_type = "discrete"
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"num_bikes_start"} 
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": tuple([15] * 25)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = BikeShare(self.model_fixed_factors)

        self.dim = self.model.factors["map_dim"]**2
        self.lower_bounds = tuple(np.zeros(self.dim))
        self.upper_bounds = tuple(self.model.factors["station_capacities"] * np.ones(self.dim))

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
            "num_bikes_start": vector
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
        det_objectives_gradients = ((0,) * self.dim, ) 
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
        x = rand_sol_rng.integer_random_vector_from_simplex(self.model.factors["map_dim"]**2, self.model.factors["num_bikes"])

        capacity = self.model.factors["station_capacities"]
        surplus = 0
        for i, num in enumerate(x):
            if num > capacity[i]:
                surplus += num - capacity[i]
                x[i] = capacity[i]

        rand_x = np.random.permutation(len(x))
        for i in rand_x:
            if surplus <= 0:
                break
            elif x[i] < capacity[i]:
                surplus = surplus - (capacity[i] - x[i])
                x[i] = capacity[i] + min(0, surplus)
        return tuple(x)