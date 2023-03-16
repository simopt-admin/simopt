"""
Summary
-------
Simulate the distribution of emergency medical service volunteer.
"""
import numpy as np
from itertools import chain
from math import sqrt, sin, cos, pi
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
                # "default": 100
            },
            "thre_dist": {
                "description": "The distance within which a volunteer can reach a call within the time threshold in meters.",
                "datatype": float,
                "default": 100.0
            },
            "num_squares": {
                "description": "Number of squares (regions) the city is divided into.",
                "datatype": int,
                # "default": 400
                "default": 4
            },
            "square_length": {
                "description": "Length (or width) of the square in meters.",
                "datatype": int,
                "default": 500
            },
            "p_OHCA": {
                "description": "Probability of an OHCA occurs in each square.",
                "datatype": list,
                # "default": np.genfromtxt('p_OHCA.csv', delimiter=',').tolist() # TODO: for high dimension case -either copy the entire matrix here or incorporate generation of the matrix here
                "default": [[0.1, 0.1],[0.1, 0.7]]
            },
            "p_vol": {
                "description": "Probability of an available volunteer is in each square.",
                "datatype": tuple,
                # "default": tuple((1/400 * np.ones(400)).tolist())
                "default": tuple((1/4 * np.ones(4)).tolist())
            },
            "num_OHCA": {
                "description": "Number of OHCAs to generate.",
                "datatype": int,
                "default": 30
            },
        }
        self.check_factor_list = {
            "mean_vol": self.check_mean_vol,
            "thre_dist": self.check_thre_dist,
            "num_squares": self.check_num_squares,
            "p_OHCA": self.check_p_OHCA,
            "p_vol": self.check_p_vol,
            "square_length": self.check_square_length,
            "num_OHCA": self.check_num_OHCA
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
        return len(self.factors["p_vol"]) == self.factors["num_squares"]
    
    def check_square_length(self):
        return self.factors["square_length"] > 0
    
    def check_num_OHCA(self):
        return self.factors["num_OHCA"] > 0

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
        vol_sq_rng = rng_list[1]
        vol_loc_lon_rng = rng_list[2]
        vol_loc_lat_rng = rng_list[3]
        OHCA_loc_lat_rng = rng_list[4]
        OHCA_loc_lon_rng = rng_list[5]
        OHCA_sq_rng = rng_list[6]

        # Initialize quantities to track:
        # - Location of an OHCA.
        # - Locations of the volunteers.
        # - Number of points in each square.
        OHCA_loc = None
        vol_locs = []
        vol_locs_idx = []
        
        # Generate number of volunteers through a Poisson random variable.
        num_vol = num_vol_rng.poissonvariate(self.factors["mean_vol"])

        u = vol_sq_rng.uniform()
        x = int(u * 1e+6 // 1000) /1000
        y = int(u * 1e+6 % 1000) / 1000

        sort_p_vol = np.sort(self.factors["p_vol"])[::-1]
        sort_p_idx = np.argsort(self.factors["p_vol"])[::-1]

        temp_x = x
        temp_y = y
        sort_x_idx = -1
        sort_y_idx = -1
        for i in range(len(sort_p_vol)):
            if (temp_x - sort_p_vol[i]) > 0:
                temp_x -= sort_p_vol[i]
            else:
                sort_x_idx = i
                break
        for i in range(len(sort_p_vol)):
            if (temp_y - sort_p_vol[i]) > 0:
                temp_y -= sort_p_vol[i]
            else:
                sort_y_idx = i
                break
        x_idx = sort_p_idx[sort_x_idx]
        y_idx = sort_p_idx[sort_y_idx]       
            


        


        # pts = np.zeros(self.factors["num_squares"])

        # # Generate points that are equally spaced within a circle of radius "thre_dist" using sunflow seed arrangment.
        # phi = (1 + sqrt(5)) / 2  # golden ratio
        # def sunflower(n, alpha=0):
        #     xs = []
        #     ys = []
        #     angle_stride = 2 * pi / phi ** 2
        #     b = round(alpha * sqrt(n))  # number of boundary points
        #     for k in range(1, n + 1):
        #         r = radius(k, n, b)
        #         theta = k * angle_stride
        #         xs.append(r * cos(theta))
        #         ys.append(r * sin(theta))
        #     return xs, ys
        # def radius(k, n, b):
        #     if k > n - b:
        #         return self.factors["thre_dist"]
        #     else:
        #         return sqrt(k - 0.5) / sqrt(n - (b + 1) / 2) * self.factors["thre_dist"]     
        # # Store coordinates of 200 points.
        # xs, ys = sunflower(200, alpha=2)


        # Generate the locations of the volunteers following a Poisson point process.
        prob2, alias2, value_list2 = vol_sq_rng.alias_init(dict(enumerate(list(self.factors["p_vol"]), 1)))
        for _ in range(num_vol):
            # Generate the coordinates of the square the volunteer is located in.
            u4 = vol_sq_rng.alias(prob2, alias2, value_list2)
            x = u4 // np.sqrt(self.factors["num_squares"])
            y = u4 % np.sqrt(self.factors["num_squares"])
            vol_locs_idx.append((x, y))
            # Find a random location in the square.
            u5 = vol_loc_lat_rng.uniform(0, 1)
            u6 = vol_loc_lon_rng.uniform(0, 1)
            vol_locs.append(((x + u5) * self.factors["square_length"], 
                        (y + u6) * self.factors["square_length"]))

        flat_p_OHCA = list(chain.from_iterable(self.factors["p_OHCA"]))
        prob1, alias1, value_list1 = OHCA_sq_rng.alias_init(dict(enumerate(flat_p_OHCA, 1)))    
        sum_flag = 0
        flag_grads = []
        sum_p = 0
        p_grads = []
        # Generate the location of an OHCA in each iteration.
        for i in range(self.factors["num_OHCA"]):
            # Generate the square containing the OHCA.
            u1 = OHCA_sq_rng.alias(prob1, alias1, value_list1)
            x = u1 // np.sqrt(self.factors["num_squares"])
            y = u1 % np.sqrt(self.factors["num_squares"])
            # Find a random location in the square.
            u2 = OHCA_loc_lat_rng.uniform(0, 1)
            u3 = OHCA_loc_lon_rng.uniform(0, 1)
            OHCA_loc = ((x + u3) * self.factors["square_length"], 
                        (y + u2) * self.factors["square_length"])
            # Calculate the distance of the closest volunteer to the location of an OHCA.
            dists = []
            vol_cnt = {}
            for i in range(num_vol):
                dist = np.sqrt((vol_locs[i][0] - OHCA_loc[0])**2 + (vol_locs[i][1] - OHCA_loc[1])**2)
                dists.append(dist)
                vol_coord = (vol_locs_idx[i][0], vol_locs_idx[i][1])
                if vol_coord in vol_cnt:
                    vol_cnt[vol_coord] += 1
                else:
                    vol_cnt[vol_coord] = 1
            min_dist = np.min(dists)
            # closest_loc = vol_locs[np.argmin(dists)]
            thre_dist_flag = 0
            # Check the minimum distance against the threshold distance.
            if min_dist > self.factors["thre_dist"]:
                thre_dist_flag = 1
            sum_flag += thre_dist_flag

            # # Compute gradient estimator for thre_dist_flag
            # for j in range(200):
            #     # Get actual coordinate of the point
            #     temp_x = OHCA_loc[0] + xs[j]
            #     temp_y = OHCA_loc[1] + ys[j]
            #     if temp_x > 0 and temp_y > 0 and temp_x < self.factors["square_length"] * np.sqrt(self.factors["num_squares"]) and temp_y < self.factors["square_length"] * np.sqrt(self.factors["num_squares"]):
            #         # Get index of the square
            #         idx_sq = int(np.sqrt(self.factors["num_squares"]) * int(temp_x / self.factors["square_length"]) + int(temp_y / self.factors["square_length"]))
            #         # Update pts
            #         pts[idx_sq] += 1
            # frac_sq = np.zeros(self.factors["num_squares"])
            # # Find max possible points in a square.
            # count = 0
            # for x,y in zip(xs, ys):
            #     if x > -self.factors["thre_dist"]/4 and y > -self.factors["thre_dist"]/4 and x < self.factors["thre_dist"]/4 and y < self.factors["thre_dist"]/4:
            #         count += 1
            # max_possible = (self.factors["square_length"] / self.factors["thre_dist"] * 4)**2 * count
            # for i in range(self.factors["num_squares"]):
            #     # Fraction of square i covered by circle can be approximated by pts[i]/max possible.
            #     frac_sq[i] = pts[i] / max_possible
            # # Estimated mean number of volunteers in the circle with radius "thre_dist".
            # vol_in_circle = np.sum(np.multiply(self.factors["p_vol"], frac_sq))
            # flag_grad = np.zeros(self.factors["num_squares"])
            # for i in range(self.factors["num_squares"]):
            #     flag_grad[i] = -np.exp(-self.factors["mean_vol"] * vol_in_circle) * self.factors["mean_vol"] * frac_sq[i]
            # flag_grads.append(flag_grad)
            
            # Use the survival function to calculate the probability of survival.
            # Convert distance to time: 3 min + D/(6km/hr).
            t = 3 + min_dist / (6000 / 60)
            p_survival = (1 + np.exp(0.679 + 0.262*t))**(-1)
            sum_p += p_survival

            # Compute gradient estimator for p_survival.
            p_grad = np.zeros(self.factors["num_squares"])
            for i in range(self.factors["num_squares"]):
                x = i // np.sqrt(self.factors["num_squares"])
                y = i % np.sqrt(self.factors["num_squares"])
                if self.factors["p_vol"][i] > 0:
                    p_grad[i] = p_survival * vol_cnt.get((x, y), 0) / self.factors["p_vol"][i]
            p_grads.append(p_grad)

        # Compose responses and gradients.
        responses = {"thre_dist_flag": sum_flag / self.factors["num_OHCA"],
                    "p_survival": sum_p / self.factors["num_OHCA"],
                    "num_vol": num_vol}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        gradients["thre_dist_flag"]["p_vol"] = np.average(flag_grads, axis=0)
        gradients["p_survival"]["p_vol"] = np.average(p_grads, axis=0)

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
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_available = True
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
                # "default": tuple((1/4 * np.ones(4)).tolist())
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000
            },
            "p_OHCA": {
                "description": "Probability of an OHCA occurs in each square.",
                "datatype": list,
                "default": np.genfromtxt('p_OHCA.csv', delimiter=',').tolist()
                # "default": [[0.1, 0.1],[0.1, 0.7]]
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
        self.set_linear_constraints()

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
        if np.isnan(factor_dict["p_vol"]).any():
            vector = np.nan
        else:
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
        return np.sum(x) <= 1
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
        if rand_sol_rng.random() < 0.25:
            x = tuple(rand_sol_rng.unitsimplexvariate(self.dim))
        else:
            x = tuple(list(chain.from_iterable(self.factors["p_OHCA"])))
        return x
    
    def set_linear_constraints(self):
        # Initialize linear constraint matrices.
        self.Ci = None
        self.Ce = None
        self.di = None
        self.de = None
        if self.constraint_type != "deterministic": # maybe create a new type of constraint named "linear"
            return
        else:
            self.Ce = np.ones(self.dim)
            self.de = np.array([1])


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
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"p_vol"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution.",
                "datatype": tuple,
                # "default": tuple((1/400 * np.ones(400)).tolist())
                "default": tuple((1/4 * np.ones(4)).tolist())

            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 1000
            },
            "p_OHCA": {
                "description": "Probability of an OHCA occurs in each square.",
                "datatype": list,
                # "default": np.genfromtxt('p_OHCA.csv', delimiter=',').tolist()
                "default": [[0.1, 0.1],[0.1, 0.7]]
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
        self.set_linear_constraints()

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
        if np.isnan(factor_dict["p_vol"]).any():
            vector = np.nan
        else:
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
        return np.sum(x) == 1

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
        if rand_sol_rng.random() < 0.5:
            x = tuple(rand_sol_rng.unitsimplexvariate(self.dim))
        else:
            x = tuple(list(chain.from_iterable(self.factors["p_OHCA"])))
        return x
    
    def set_linear_constraints(self):
        # Initialize linear constraint matrices.
        self.Ci = None
        self.Ce = None
        self.di = None
        self.de = None
        if self.constraint_type != "deterministic": # maybe create a new type of constraint named "linear"
            return
        else:
            self.Ce = np.ones(self.dim)
            self.de = np.array([1])
