"""
Summary
-------
Simulate the distribution of emergency medical service volunteer.
"""
import numpy as np

from base import Model, Problem


class Volunteer(Model):
    """
    A model that simulates the distribution of emergency medical service volunteer
    through a Poisson point process in a city.

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
        self.name = "VOLUNTEER"
        self.n_rngs = 7
        self.n_responses = 6
        self.specifications = {
            "mean_vol": {
                "description": "Mean number of available volunteers.",
                "datatype": int,
                "default": 1600
            },
            "thre_dist": {
                "description": "The distance within which a volunteer can reach a call within the time threshold in meters.",
                "datatype": float,
                "default": 200.0
            },
            "num_squares": {
                "description": "Number of squares (regions) the city is divided into.",
                "datatype": int,
                "default": 400
            },
            "square_length": {
                "description": "Length (or width) of the square in meters.",
                "datatype": int,
                "default": 500
            },
            "p_OHCA": {
                "description": "Probability of an OHCA occurs in each square.",
                "datatype": list,
                "default": np.genfromtxt('p_OHCA.csv', delimiter=',').tolist()
            },
            "p_vol": {
                "description": "Probability of an available volunteer is in each square.",
                "datatype": tuple,
                "default": tuple((1/400 * np.ones(400)).tolist())
            }
        }
        self.check_factor_list = {
            "mean_vol": self.check_mean_vol,
            "thre_dist": self.check_thre_dist,
            "num_squares": self.check_num_squares,
            "p_OHCA": self.check_p_OHCA,
            "p_vol": self.check_p_vol,
            "square_length": self.check_square_length
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_mean_vol(self):
        return self.factors["mean_vol"] > 0

    def check_thre_dist(self):
        return self.factors["thre_dist"] > 0

    def check_num_squares(self):
        return self.factors["num_squares"] > 0

    def check_p_OHCA(self):
        return (len(self.factors["p_OHCA"]) * len(self.factors["p_OHCA"][0])) == self.factors["num_squares"]

    def check_p_vol(self):
        return (len(self.factors["p_vol"]) * len(self.factors["p_vol"][0])) == self.factors["num_squares"]
    
    def check_square_length(self):
        return self.factors["square_length"] > 0

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
            "thre_dist_flag" = whether the distance of the closest volunteer exceeds the threshold distance
            "p_survival" = probability of survial
            "OHCA_loc" = location of the OHCA
            "closest_loc" = the closest volunteer location
            "closest_dist" = the distance of the closest volunteer in meters.
            "num_vol" = total number of volunteers available
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        num_vol_rng = rng_list[0]
        vol_loc_lat_rng = rng_list[1]
        vol_loc_lon_rng = rng_list[2]
        vol_loc_rng = rng_list[3]
        OHCA_loc_lat_rng = rng_list[4]
        OHCA_loc_lon_rng = rng_list[5]
        OHCA_loc_rng = rng_list[6]

        # Reshape p_vol.
        p_vol = np.array(list(self.factors["p_vol"])).reshape((20, 20))

        # Initialize quantities to track:
        # - Location of an OHCA.
        # - Locations of the volunteers.
        # - Whether the distance of the closet volunteer exceeds the threshold distance.
        OHCA_loc = None
        closest_loc = None
        vol_locs = []
        thre_dist_flag = 0

        # Generate number of volunteers through a Poisson random variable.
        num_vol = num_vol_rng.poissonvariate(self.factors["mean_vol"])

        # Generate the square location of an OHCA by acceptance-rejection.
        done = False
        x = None
        y = None
        while not done:
            u1 = OHCA_loc_lat_rng.uniform(0, 1)
            u2 = OHCA_loc_lon_rng.uniform(0, 1)
            x_temp = u1 * (np.sqrt(self.factors["num_squares"]) - 1)
            y_temp = u2 * (np.sqrt(self.factors["num_squares"]) - 1)
            u3 = OHCA_loc_rng.uniform(0, 1)
            if u3 <= self.factors["p_OHCA"][int(x_temp)][int(y_temp)]:
                x = x_temp
                y = y_temp
                done = True
        # Find a random location in that square.
        OHCA_loc = (x * self.factors["square_length"], 
                    y * self.factors["square_length"])
        
        # Generate the locations of the volunteers by a Poisson point process through acceptance-rejection.
        for _ in range(num_vol):
            # Generate the coordinates of the square the volunteer is located in.
            done = False
            x = None
            y = None
            while not done:
                u4 = vol_loc_lat_rng.uniform(0, 1)
                u5 = vol_loc_lon_rng.uniform(0, 1)
                x_temp = u4 * (np.sqrt(self.factors["num_squares"]) - 1)
                y_temp = u5 * (np.sqrt(self.factors["num_squares"]) - 1)
                u6 = vol_loc_rng.uniform(0, 1)
                if u6 <= p_vol[int(x_temp)][int(y_temp)]:
                    x = x_temp
                    y = y_temp
                    done = True
            # Find a random location in that square.
            vol_locs.append((x * self.factors["square_length"], 
                            y * self.factors["square_length"]))

        # Calculate the distance of the closest volunteer to the location of an OHCA.
        dists = []
        for i in range(num_vol):
            dist = np.sqrt((vol_locs[i][0] - OHCA_loc[0])**2 + (vol_locs[i][1] - OHCA_loc[1])**2)
            dists.append(dist)
        min_dist = np.min(dists)
        closest_loc = vol_locs[np.argmin(dists)]

        # Check the minimum distance against the threshold distance.
        if min_dist > self.factors["thre_dist"]:
            thre_dist_flag = 1
        # Use the survival function to calculate the probability of survival.
        # Convert distance to time: 3 min + D/(6km/hr).
        t = 3 + min_dist / (6000 / 60)
        p_survival = (1 + np.exp(0.679 + 0.262*t))**(-1)

        # Compose responses and gradients.
        responses = {"thre_dist_flag": thre_dist_flag,
                    "p_survival": p_survival,
                    "OHCA_loc": OHCA_loc,
                    "closest_loc":closest_loc,
                    "closest_dist": min_dist,
                    "num_vol": num_vol}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients

"""
Summary
-------
Minimize the probability of exceeding the threshold distance.
"""


class VolunteerDist(Problem):
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
    def __init__(self, name="VOLUNTEER-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"p_vol"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution.",
                "datatype": tuple,
                "default": tuple((1/400 * np.ones(400)).tolist())
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
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = Volunteer(self.model_fixed_factors)
        self.dim = self.model.factors["num_squares"]
        self.lower_bounds = tuple(np.zeros(self.dim))
        self.upper_bounds = tuple(np.ones(self.dim))

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
            "p_vol": vector[:]
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
        vector = tuple(factor_dict["p_vol"])
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
        objectives = (response_dict["thre_dist_flag"],)
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
        det_stoch_constraints_gradients = None
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
        det_objectives_gradients = ((0,) * self.dim,)
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
        return (np.sum(x) <= 1 + 10**(-5)) or (np.sum(x) >= 1 - 10**(-5))

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
        x = tuple(rand_sol_rng.unitsimplexvariate(self.dim))
        return x

"""
Summary
-------
Maximize the probability of survival.
"""


class VolunteerSurvival(Problem):
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
    def __init__(self, name="VOLUNTEER-2", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"p_vol"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution.",
                "datatype": tuple,
                "default": tuple((1/400 * np.ones((20, 20))).tolist())
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
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = Volunteer(self.model_fixed_factors)
        self.dim = self.model.factors["num_squares"]
        self.lower_bounds = tuple(np.zeros(self.dim))
        self.upper_bounds = tuple(np.ones(self.dim))

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
            "p_vol": vector[:]
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
        vector = tuple(factor_dict["p_vol"])
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
        objectives = (response_dict["p_survival"],)
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
        det_stoch_constraints_gradients = None
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
        det_objectives_gradients = ((0,) * self.dim,)
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
        return (np.sum(x) <= 1 + 10**(-5)) or (np.sum(x) >= 1 - 10**(-5))

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
        x = tuple(rand_sol_rng.unitsimplexvariate(self.dim))
        return x
