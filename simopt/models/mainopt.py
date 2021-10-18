"""
Summary
-------
Simulate maintenances of a facility.
"""
import numpy as np

import math

from base import Model, Problem


class MainOpt(Model):
    """
    An model that simulates multiple periods of maintenances of
    a facility.
    Returns total cost.

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
    fixed_factors : nested dict
        fixed factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors={}):
        self.name = "MAINOPT"
        self.n_rngs = 2
        self.n_responses = 2
        self.factors = fixed_factors
        self.specifications = {

            "C_pm": {
                "description": "Preventive maintenance cost.",
                "datatype": float,
                "default": 18998.0
            },
            "C_cm": {
                "description": "Corrective maintenance cost.",
                "datatype": float,
                "default": 97997.0
            },
            "C_trip": {
                "description": "Cost given trip occurs.",
                "datatype": float,
                "default": 4349781.0
            },
            "p_trip": {
                "description": "Probability larger failure (trip) occurs.",
                "datatype": float,
                "default": 0.0073
            },
            "time_horizon": {
                "description": "Time horizon.",
                "datatype": int,
                "default": 10
            },
            "num_PMs": {
                "description": "Number of preventive maintenances over the time horizon",
                "datatype": int,
                "default": 2
            },
            "PM_times": {
                "description": "Time of preventive maintenances.",
                "datatype": list,
                "default": [5, 5]
            },
        }

        self.check_factor_list = {
            "C_pm": self.check_C_pm,
            "C_cm": self.check_C_cm,
            "C_trip": self.check_C_trip,
            "p_trip": self.check_p_trip,
            "num_PMs": self.check_num_PMs,
            "time_horizon": self.check_time_horizon,
            "PM_times": self.check_PM_times,
        }
        # Set factors of the simulation model
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_C_pm(self):
        return self.factors["C_pm"] > 0

    def check_C_cm(self):
        return self.factors["C_cm"] > 0

    def check_C_trip(self):
        return self.factors["C_trip"] >= 0

    def check_p_trip(self):
        return (self.factors["p_trip"] >= 0 or self.factors["p_trip"] <= 1)

    def check_time_horizon(self):
        return self.factors["time_horizon"] > 0

    def check_num_PMs(self):
        return self.factors["num_PMs"] > 0

    def check_PM_times(self):
        return all((PM_time > 0) & (PM_time <= self.factors["time_horizon"]) for PM_time in self.factors["PM_times"])

    def check_simulatable_factors(self):
        return (sum(PM_time for PM_time in self.factors["PM_times"]) == self.factors["time_horizon"]) & (len(self.factors["PM_times"]) == self.factors["num_PMs"])

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
            "total_cost" = total cost
            "total_failure" = total failure
        """
        # Designate random number generators.
        fail_rng = rng_list[0]
        ptrip_rng = rng_list[1]
        # Initialize interval times between PM time and the number of stretches
        inter_PMtimes = self.factors["PM_times"]
        num_stretch = len(inter_PMtimes)
        # Initialize quantities to track:
        #   - Cost in each period.
        #   - Time state in each period.
        #   - Whether the current period has failure.
        #   - Whether the current period has a "trip".
        #   - Stretch the current period belongs to.
        cost = np.zeros(self.factors["time_horizon"])
        time = np.zeros(self.factors["time_horizon"])
        failure = np.zeros(self.factors["time_horizon"])
        trip = np.zeros(self.factors["time_horizon"])
        stretch = np.zeros(self.factors["time_horizon"])
        #Run simulation over time horizon.
        for period in range(self.factors["time_horizon"]):
            # Calculate the failure rate z(t)
            z = 1 - np.exp(-time[period])
            # Generate random failure. 
            u = fail_rng.random()
            if u < z:
                failure[period] = 1
            # If failure occurs:
            if failure[period] == 1:
                # If previous failure is before maintenance, determine cost of failure
                if time[period] < inter_PMtimes[int(stretch[period])]:
                    # Generate random "trip". If a trip occurs, cost increase by C_trip; 
                    # otherwise, cost increases by C_cm
                    p = ptrip_rng.random()
                    if p < self.factors["p_trip"]:
                        trip[period] = 1
                        cost[period] += self.factors["C_trip"]
                    else:
                        cost[period] += self.factors["C_cm"]  
                        time[period] = 0
                # elif time[period] == inter_PMtimes[stretch[period]]:
                #     time[period] = 0
                #     stretch[period] += 1
                #     cost[period] += self.factors["C_pm"] 
            # Perform PM when it is the scheuduled time
            if period == np.cumsum(inter_PMtimes)[int(stretch[period])] - 1:
                time[period] = 0
                stretch[period] += 1
                cost[period] += self.factors["C_pm"]    
            # Calculate starting quantities for next period
            if period < self.factors["time_horizon"] - 1:
                cost[period + 1] = cost[period]
                time[period + 1] = time[period] + 1
                stretch[period + 1] = stretch[period]


        # Calculate responses from simulation data.
        responses = {"total_cost": cost[self.factors["time_horizon"] - 1],
                    "total_failure":stretch[self.factors["time_horizon"] - 1]
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Maximize the expected total revenue for iron ore inventory system.
"""


class MainOptMinCost(Problem):
    """
    Class to make facility maintenance simulation-optimization problems.

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
    def __init__(self, name="MAINOPT-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.dim = 4
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "discrete"
        self.lowerbound = (0)
        self.upperbound = (np.inf)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (80, 7000, 40, 100)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
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
        self.model = MainOpt(self.model_fixed_factors)

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
            "price_prod": vector[0],
            "inven_stop": vector[1],
            "price_stop": vector[2],
            "price_sell": vector[3],
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
        vector = (factor_dict["price_prod"], factor_dict["inven_stop"], factor_dict["price_stop"], factor_dict["price_sell"])
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
        objectives = (response_dict["total_revenue"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

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
        det_objectives_gradients = ((0,),)
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
        return (x[0] >= 0 and x[1] >= 0 and x[2] >= 0 and x[3] >= 0)

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
        x = (rand_sol_rng.randint(70, 90), rand_sol_rng.randint(2000, 8000), rand_sol_rng.randint(30, 50), rand_sol_rng.randint(90, 110))
        return x
