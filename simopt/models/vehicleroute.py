"""
Summary
-------
Simulate vehicle routes with stochastic demands and travel times.
"""
import numpy as np

from base import Model, Problem


class VehicleRoute(Model):
    """
    A model that simulates vehicle routiing problem with 
    stochastic demands and travel times.
    Returns the set of routes for the vehicles

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
        self.name = "VEHROUTE"
        self.n_rngs = 2
        self.n_responses = 4
        self.specifications = {
            "n_cus": {
                "description": "The number of customers.",
                "datatype": int,
                "default": 5
            },
            "n_veh": {
                "description": "The number of vehicles.",
                "datatype": int,
                "default": 5
            },
            "d_lim": {
                "description": "The demand capacity of each vehicle",
                "datatype": float,
                "default": 350.0
            },
            "t_lim": {
                "description": "The travel time limit along each route",
                "datatype": float,
                "default": 240.0
            },
            "alpha": {
                "description": "The desired service level of capacity.",
                "datatype": float,
                "default": 0.9
            },
            "beta": {
                "description": "The desire service level of time.",
                "datatype": float,
                "default": 0.9
            },
            "dist_mat": {
                "description": "The matrix representing the distance between vertices.",
                "datatype": list,
                "default": [[0, 35, 78, 76, 98, 55], [35, 0, 60, 59, 91, 81], [78, 60, 0, 3, 37, 87], [76, 59, 3, 0, 36, 83], [98, 91, 37, 36, 0, 84], [55, 81, 87, 83, 84, 0]]
            },
            "routes": {
                "description": "The routes for the vehicles.",
                "datatype": list,
                "default": [[1, 0, 0, 0, 0], [2, 0, 0 ,0 ,0], [3, 0, 0, 0, 0], [4, 0, 0, 0, 0],[5, 0, 0, 0, 0]]
            },
        }
        self.check_factor_list = {
            "n_cus": self.check_n_cus,
            "n_veh": self.check_n_veh,
            "d_lim": self.check_d_lim,
            "t_lim": self.check_t_lim,
            "alpha": self.check_alpha,
            "beta": self.check_beta,
            "dist_mat": self.check_dist_mat,
            "routes": self.check_routes
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_n_cus(self):
        return self.factors["n_cus"] > 0

    def check_n_veh(self):
        return self.factors["n_veh"] > 0

    def check_d_lim(self):
        return self.factors["d_lim"] >= 0

    def check_t_lim(self):
        return self.factors["t_lim"] >= 0

    def check_alpha(self):
        return self.factors["alpha"] >= 0 and self.factors["alpha"] <= 1
    
    def check_beta(self):
        return self.factors["beta"] >= 0 and self.factors["beta"] <= 1

    def check_dist_mat(self):
        return np.all(self.factors["dist_mat"]) > 0
    
    def check_routes(self):
        return np.all(self.factors["routes"]) > 0

    def check_simulatable_factors(self):
        if len(self.factors["dist_mat"]) != self.factors["n_cus"] + 1:
            return False
        elif len(self.factors["dist_mat"][0]) != self.factors["n_cus"] + 1:
            return False
        elif len(self.factors["routes"]) != self.factors["n_veh"]:
            return False
        elif len(self.factors["routes"][0]) != self.factors["n_cus"]:
            return False
        else:
            return True

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
            "total_dist" = total distance traveled by the vehicles
            "num_veh_used" = the number of vehicles that are used in current set of routes
            "demand_routes" = the demand along the route for each vehicle
            "time_routes" = the travel time along the route for each vehicle
        gradients : dict of dicts
            gradient estimates for each response
        """

        # Generate random travel time with uniform distribution
        time_rng = rng_list[0]
        t_travel = np.zeros((self.factors["n_cus"] + 1, self.factors["n_cus"] + 1))
        for i in range(self.factors["n_cus"] + 1):
            for j in range(self.factors["n_cus"] + 1):
                t_travel[i, j] = time_rng.uniform(0.5 * self.factors["dist_mat"][i][j], 1.5 * self.factors["dist_mat"][i][j])
        
        # Generate random customer demand with uniform distribution (110,190)
        demand_rng = rng_list[1]
        demand = np.zeros(self.factors["n_cus"] + 1)
        for i in range(self.factors["n_cus"] + 1):
            demand[i] = demand_rng.uniform(110, 190)
        
        # Initialize quantities to track
        num_veh_used = 0
        total_dist = 0
        time_routes = np.zeros(self.factors["n_veh"])
        demand_routes = np.zeros(self.factors["n_veh"])

        for i in range(self.factors["n_veh"]):
            total_dist += self.factors["dist_mat"][0][self.factors["routes"][i][0]]
            time_routes[i] += t_travel[0, self.factors["routes"][i][0]]
            # Calculate the number of vehicles that are used in the current set of routes
            if np.sum(np.array(self.factors["routes"])[i, :] > 0) > 0:
                num_veh_used += 1
            for j in range(1, self.factors["n_cus"]):
                total_dist += self.factors["dist_mat"][self.factors["routes"][i][j - 1]][self.factors["routes"][i][j]]
                time_routes[i] += t_travel[self.factors["routes"][i][j - 1], self.factors["routes"][i][j]]
            # Goes to every customer, so last entry does not end with zero - still need to add final leg (back to depot)
            if self.factors["routes"][i][self.factors["n_cus"] - 1] != 0:
                total_dist += self.factors["dist_mat"][self.factors["routes"][i][self.factors["n_cus"] - 1]][0]
                time_routes[i] += t_travel[self.factors["routes"][i][self.factors["n_cus"] - 1], 0]
            # Calculate total demand of route i
            demand_routes[i] = np.sum(demand[j] for j in np.array(self.factors["routes"])[i, :])


        # Compose responses and gradients.
        responses = {"total_dist": total_dist,
                     "num_veh_used": num_veh_used,
                     "demand_routes": demand_routes,
                     "time_routes": time_routes}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Minimize the total distance traveled by the vehicle while
maintaining desired service levels.
"""


class VehicleRouteTotalDist(Problem):
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
            initial_solution : tuple
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
    def __init__(self, name="VEHROUTE-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.minmax = (-1,)
        self.constraint_type = "stochastic"
        self.variable_type = "continuous"
        self.lower_bounds = (0, 0, 0)
        self.upper_bounds = (np.inf, np.inf, np.inf)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None  
        self.model_default_factors = {}
        self.model_decision_factors = {"routes"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (300, 300, 300)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = VehicleRoute(self.model_fixed_factors)
        self.dim = self.model.factors["n_veh"]

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
            "routes": vector[:]
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
        vector = tuple(factor_dict["routes"])
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
        objectives = (0,)
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
        stoch_constraints = (-response_dict["stockout_flag"],)
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
        det_stoch_constraints = (self.factors["epsilon"],)
        det_stoch_constraints_gradients = ((0,),)
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
        det_objectives = (np.dot(self.factors["installation_costs"], x),)
        det_objectives_gradients = ((self.factors["installation_costs"],),)
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
        return np.all(x > 0)

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
        x = tuple([300*rand_sol_rng.random() for _ in range(self.dim)])
        return x