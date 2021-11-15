"""
Summary
-------
Simulate duration of stochastic activity network (SAN).
"""
import numpy as np
from scipy import special

from base import Model, Problem


class SAN(Model):
    """
    A model that simulates a stochastic activity network problem with tasks
    that have exponentially distributed durations, and the selected means
    come with a cost. 
    Returns the optimal mean duration for each task.

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
        self.name = "SAN"
        self.n_rngs = 1
        self.n_responses = 1
        self.specifications = {
            "num_arcs": {
                "description": "Number of arcs.",
                "datatype": int,
                "default": 13
            },
            "num_nodes": {
                "description": "Number of nodes.",
                "datatype": int,
                "default": 9
            },
            "runlength": {
                "description": "Number of replications to calculate expectation.",
                "datatype": int,
                "default": 100
            },
            "initial_thetas": {
                "description": "Initial solution of means.",
                "datatype": tuple,
                "default": (1,1,1,1,1,1,1,1,1,1,1,1,1)
            }
        }
        self.check_factor_list = {
            "num_arcs": self.check_num_arcs,
            "num_nodes": self.check_num_nodes,
            "initial_thetas": self.check_initial_thetas
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_num_arcs(self):
        return self.factors["num_arcs"] > 0

    def check_num_nodes(self):
        return self.factors["num_nodes"] > 0
      
    def check_initial_thetas(self):
        positive = True
        for x in list(self.factors["initial_thetas"]):
          positive = positive & x > 0
        return (len(self.factors["initial_thetas"]) != self.factors["num_arcs"]) & positive

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
            "T" = the duration of path
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        exp_rng = rng_list[0]

        means = np.zeros(self.factors["runlength"])
        meanGrad = np.zeros((self.factors["runlength"], self.factors["num_arcs"]))
        for i in range(self.factors["runlength"]):
          exp_rng.advance_substream()
          T = np.zeros(self.factors["num_nodes"])
          Tderiv = np.zeros((self.factors["num_nodes"], self.factors["num_arcs"]))
          arcs = [exp_rng.expovariate(1/x) for x in list(self.factors["initial_thetas"])]
          # thetas = [exp_rng.expovariate(1) for i in range((self.factors["num_arcs"]))]
          # arcs = [t*x for t in thetas for x in list(self.factors["initial_thetas"])]
          thetas = list(self.factors["initial_thetas"])

          # Brute force calculation like in Matlab code
          T[1] = T[0] + arcs[0]
          Tderiv[1,:] = Tderiv[0,:]
          Tderiv[1,0] = Tderiv[1,0] + arcs[0] / thetas[0]

          T[2] = max(T[0] + arcs[1], T[1] + arcs[2])
          if T[0] + arcs[1] > T[1] + arcs[2]:
            T[2] = T[0] + arcs[1]
            Tderiv[2,:] = Tderiv[0,:]
            Tderiv[2,1] = Tderiv[2,1] + arcs[1] / thetas[1]
          else:
            T[2] = T[1] + arcs[2]
            Tderiv[2,:] = Tderiv[1,:]
            Tderiv[2,2] = Tderiv[2,2] + arcs[2] / thetas[2]

          T[3] = T[1] + arcs[3]
          Tderiv[3,:] = Tderiv[1,:]
          Tderiv[3,3] = Tderiv[3,3] + arcs[3] / thetas[3]

          T[4] = T[3] + arcs[6]
          Tderiv[4,:] = Tderiv[3,:]
          Tderiv[4,6] = Tderiv[4,6] + arcs[6] / thetas[6]

          T[5] = max([T[1] + arcs[4], T[2] + arcs[5], T[4] + arcs[8]])
          ind = np.argmax([T[1] + arcs[4], T[2] + arcs[5], T[4] + arcs[8]])
          if ind == 1:
            Tderiv[5,:] = Tderiv[1,:]
            Tderiv[5,4] = Tderiv[5,4] + arcs[4] / thetas[4]
          elif ind == 2:
            Tderiv[5,:] = Tderiv[2,:]
            Tderiv[5,5] = Tderiv[5,5] + arcs[5] / thetas[5]
          else:
            Tderiv[5,:] = Tderiv[4,:]
            Tderiv[5,8] = Tderiv[5,8] + arcs[8] / thetas[8]

          T[6] = T[3] + arcs[7]
          Tderiv[6,:] = Tderiv[3,:]
          Tderiv[6,7] = Tderiv[6,7] + arcs[7] / thetas[7]

          if T[6] + arcs[11] > T[4] + arcs[9]:
            T[7] = T[6] + arcs[11]
            Tderiv[7,:] = Tderiv[6,:]
            Tderiv[7,11] = Tderiv[7,11] + arcs[11] / thetas[11]
          else:
            T[7] = T[4] + arcs[9]
            Tderiv[7,:] = Tderiv[4,:]
            Tderiv[7,9] = Tderiv[7,9] + arcs[9] / thetas[9]

          if T[5] + arcs[10] > T[7] + arcs[12]:
            T[8] = T[5] + arcs[10]
            Tderiv[8,:] = Tderiv[5,:]
            Tderiv[8,10] = Tderiv[8,10] + arcs[10] / thetas[10]
          else:
            T[8] = T[7] + arcs[12]
            Tderiv[8,:] = Tderiv[7,:]
            Tderiv[8,12] = Tderiv[8,12] + arcs[12] / thetas[12]
          
          means[i] = T[8] + sum(1/x for x in list(self.factors["initial_thetas"]))
          meanGrad[i,:] = Tderiv[8,:] - [1/x**2 for x in list(self.factors["initial_thetas"])]

        # Compose responses and gradients.
        responses = {
          'ET': np.mean(means)
        }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients

"""
Summary
-------
Minimize the duration of the longest path from a to i plus cost.
"""


class SANLongestPath(Problem):
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
    def __init__(self, name="SAN", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "unconstrained"
        self.variable_type = "continuous"
        self.lower_bounds = (0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01)
        self.upper_bounds = (100,100,100,100,100,100,100,100,100,100,100,100,100)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = set("initial_thetas")
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution.",
                "datatype": tuple,
                "default": (1,1,1,1,1,1,1,1,1,1,1,1,1)
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
        self.model = SAN(self.model_fixed_factors)
        self.dim = self.model.factors["num_arcs"]

    def check_initial_solution(self):
        if len(self.factors["initial_solution"]) != self.dim:
            return False
        else:
            return True

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
            "initial_thetas": vector[:]
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
        vector = (factor_dict["initial_thetas"],)
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
        objectives = (response_dict["ET"],)
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
        return np.all(x >= 0)

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
        x = []
        for i in range(self.dim):
          x.append(rand_sol_rng.random())
        return x