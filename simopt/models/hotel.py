"""
Summary
-------
Simulate expected revenue for a hotel.
"""
import numpy as np
from scipy import special

from base import Model, Problem


class Hotel(Model):
    """
    A model that simulates business of a hotel with Poisson arrival rate.

    Attributes
    ----------
    name : string
        name of model
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI and data validation)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : nested dict
        fixed factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors={}):
        self.name = "HOTEL"
        self.n_rngs = 1
        self.n_responses = 1
        self.specifications = {
            "num_products": {
                "description": "Number of products.",
                "datatype": int,
                "default": 56
            },
            "lambda": {
                "description": "Arrival rates for each product.",
                "datatype": list,
                "default": ((1/168)*np.array([1, 1, 2, 2, 3, 3, 2, 2, 1, 1, .5, .5, .25, .25, 
                                    1, 1, 2, 2, 3, 3, 2, 2, 1, 1, .5, .5, 1, 1,
                                    2, 2, 3, 3, 2, 2, 1, 1, 1, 1, 2, 2, 3, 3, 
                                    2, 2, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 1, 1])).tolist()
            },
            "C": {
                "description": "Hotel capacity.",
                "datatype": int,
                "default": 100
            },
            "pd": {
                "description": "Discount rate.",
                "datatype": int,
                "default": 100
            },
            "pf": {
                "description": "Rack rate.",
                "datatype": int,
                "default": 200
            },
            "A": {
                "description": "Incidence matrix",
                "datatype": list,
                "default": [[1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	1,	1,	0,	0,	1,	1,	1,	1]]
            },
            "time_limit": {
                "description": "Time after which orders of each product no longer arrive (e.g. Mon night stops at 3am Tues or t=27).",
                "datatype": list,
                "default": np.concatenate((27*np.ones(14), 51*np.ones(12), 75*np.ones(10), 99*np.ones(8), 123*np.ones(6), 144*np.ones(4), 168*np.ones(2)), axis=None).tolist()
            },
            "time_before": {
                "description": "Hours before t=0 to start running (e.g. 168 means start at time -168).",
                "datatype": int,
                "default": 168
            },
            "runlength": {
                "description": "Runlength of a day.",
                "datatype": int,
                "default": 168
            },
            "b": {
                "description": "Initial solution of booking limits.",
                "datatype": tuple,
                "default": tuple([100 for _ in range(56)])
            }
        }
        self.check_factor_list = {
            "num_products": self.check_num_products,
            "lambda": self.check_lambda,
            "C": self.check_C,
            "pd": self.check_pd,
            "pf": self.check_pf,
            "A": self.check_A,
            "time_limit": self.check_time_limit,
            "time_before": self.check_time_before,
            "runlength": self.check_runlength,
            "b": self.check_b
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_num_products(self):
        return self.factors["num_products"] > 0
    
    def check_lambda(self):
        for i in self.factors["lambda"]:
            if i <= 0:
                return False
        return len(self.factors["lambda"]) == self.factors["num_products"]

    def check_C(self):
        return self.factors["C"] > 0

    def check_pd(self):
        return self.factors["pd"] > 0

    def check_pf(self):
        return self.factors["pf"] > 0

    def check_A(self):
        m, n = self.factors["A"].shape
        for i in range(m):
            for j in range(n):
                if self.factors["A"][i, j] <= 0:
                    return False
        return m*n == self.factors["num_products"]

    def check_time_limit(self):
        for i in self.factors["time_limit"]:
            if i <= 0:
                return False
        return len(self.factors["time_limit"]) == self.factors["num_products"]

    def check_time_before(self):
        return self.factors["time_before"] > 0

    def check_runlength(self):
        return self.factors["runlength"] > 0
      
    def check_b(self):
        for i in list(self.factors["b"]):
            if i <= 0 or i > self.factors["C"]:
                return False
        return len(self.factors["b"]) == self.factors["num_products"]

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "revenue" = expected revenue
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        arr_rng = rng_list[0]

        total_revenue = 0
        b = list(self.factors["b"])
        A = np.array(self.factors["A"])
        # Vector of next arrival time per product
        arrival = np.zeros(self.factors["num_products"]) - self.factors["time_before"]
        # Upper bound on number of arrivals over the time period
        arr_bound = 10*round(168*np.sum(self.factors["lambda"]))
        arr_time = np.zeros((self.factors["num_products"], arr_bound))
        # Index of which arrival time to use next for each product
        a = np.zeros(self.factors["num_products"], dtype=int)

        for i in range(self.factors["num_products"]):
            arr_time[i] = np.array([arr_rng.expovariate(self.factors["lambda"][i]) for _ in range(arr_bound)])
        
        # Generate first arrivals
        for i in range(self.factors["num_products"]):
            arrival[i] = arrival[i] + arr_time[i, a[i]]
            a[i] = 1

        min_time = 0  # keeps track of minimum time of the orders not yet received
        while min_time <= self.factors["runlength"]:
            min_time = self.factors["runlength"] + 1
            for i in range(self.factors["num_products"]):
                if ((arrival[i] < min_time) and (arrival[i] <= self.factors["time_limit"][i])):
                    min_time = arrival[i]
                    min_idx = i
            if min_time > self.factors["runlength"]:
                break
            if b[min_idx] > 0:
                if min_idx % 2 == 0: # pf
                    total_revenue += sum(self.factors["pf"] * A[:,min_idx])
                else: # pd
                    total_revenue += sum(self.factors["pd"] * A[:,min_idx])
                for i in range(self.factors["num_products"]):
                    if np.dot(A[:,i].T, A[:,min_idx]) >= 1:
                        if b[i] != 0:
                            b[i] -= 1
            arrival[min_idx] += arr_time[min_idx, a[min_idx]]
            a[min_idx] = a[min_idx] + 1

        # Compose responses and gradients.
        responses = {
          'revenue': total_revenue
        }
        gradients = {}
        return responses, gradients

"""
Summary
-------
Maximize the expected revenue.
"""


class HotelRevenue(Problem):
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
    optimal_value : float
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
    rng_list : list of rng.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="HOTEL-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "unconstrained"
        self.variable_type = "continuous"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"b"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution.",
                "datatype": tuple,
                "default": tuple([100 for _ in range(56)])
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
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = Hotel(self.model_fixed_factors)
        self.dim = self.model.factors["num_products"]
        self.lower_bounds = tuple(np.zeros(self.dim))
        self.upper_bounds = tuple(self.model.factors["C"]*np.ones(self.dim))

    def check_initial_solution(self):
        return len(self.factors["initial_solution"]) == self.dim

    def check_budget(self):
        return self.factors["budget"] > 0

    def check_simulatable_factors(self):
        if len(self.lower_bounds) != self.dim:
            return False
        elif len(self.upper_bounds) != self.dim:
            return False
        else:
            return True


    def vector_to_factor_dict(self, vector):
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
        factor_dict = {
            "b": vector[:]
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
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
        vector = tuple(factor_dict["b"])
        return vector

    def response_dict_to_objectives(self, response_dict):
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
        objectives = (response_dict["revenue"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = None
        return stoch_constraints

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = (0) # tuple of tuples – of sizes self.dim by self.dim, full of zeros
        return det_stoch_constraints, det_stoch_constraints_gradients

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

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        return True

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = tuple([(rand_sol_rng.random())*self.model.factors["C"] for _ in range(self.dim)]) # U(0, 100)
        return x