"""
Summary
-------
"""
import numpy as np

from ..base import Model, Problem


class Fake2(Model):
    """
    A fake problem

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
    def __init__(self, fixed_factors={}):
        self.name = "FAKE2"
        self.n_rngs = 0
        self.n_responses = 1
        self.factors = fixed_factors
        self.specifications = {
            "x": {
                "description": "x",
                "datatype": tuple,
                "default": (1, 0, 0, 0, 0)
                # "default": (1, 0, 0)
                # "default": (1, 0)
            },
        }
        self.check_factor_list = {
            "x": self.check_x,
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)
        
    def check_x(self):
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
        """
        
        # Compose responses and gradients.
        responses = {"sum": 0}
        gradients = {response_key:
                     {factor_key: np.nan for factor_key in self.specifications}
                     for response_key in responses
                     }
        gradients["sum"]["x"] = np.zeros(len(self.factors["x"]))
        return responses, gradients


"""
Summary
-------
Maximize the expected profit for the continuous newsvendor problem.
"""


class FakeProblem2(Problem):
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
    def __init__(self, name="FAKE2-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.optimal_value = None
        self.model_default_factors = {}
        self.model_decision_factors = {"x"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                # "default": (-5, 0, 0, 0, 0)
                "default": (1, 0, 0, 0, 0)
                # "default": (1, 0, 0)
                # "default": (1, 0)
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
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = Fake2(self.model_fixed_factors)
        self.dim = len(self.model.factors["x"])
        self.lower_bounds = (-np.inf,) * self.dim
        # self.lower_bounds = (0,) * self.dim
        self.upper_bounds = (np.inf,) * self.dim
        # self.lower_bounds = (-1, ) * self.dim
        # self.upper_bounds = (2, ) * self.dim
        self.optimal_solution = None
        # self.optimal_solution = tuple(np.ones(self.dim) / self.dim)
        self.Ci = None
        self.Ce = None
        self.di = None
        self.de = None

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
            "x": vector[:]
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
        vector = tuple(factor_dict["x"])
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
        objectives = (response_dict["sum"],)
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
        det_objectives = (np.sum(np.array(x)**2),)
        det_objectives_gradients = (2*np.array(x),)

        # det_objectives = (-2*x[0]-6*x[1]+x[0]**2 -2*x[0]*x[1]+2*x[1]**2,)
        # det_objectives_gradients = (np.array([-2+2*x[0]-2*x[1], -6-2*x[0]+4*x[1]]),)

        # # Generalized Rosenbrock function - global optimal [  1;  1;  1;  1;  1 ]
        # f = 0
        # g = np.zeros(self.dim)
        # t1 = x[0]
        # t3 = t1 - 1
        # t0 = 200*(x[1] - t1*t1)
        # g[0] = 2*(t3 - t0*t1)
        # f = 0.0025*t0*t0 + t3*t3
        # for i in range(1, self.dim-1):
        #    t1 = x[i]
        #    t2 = 200*(x[i+1] - t1*t1)
        #    t3 = t1 - 1
        #    g[i] = 2*(t3 - t2*t1) + t0
        #    f += 0.0025*t2*t2 + t3*t3
        #    t0 = t2
        # g[self.dim - 1] = 200*(x[self.dim - 1] - x[self.dim-2]*x[self.dim-2])

        # det_objectives = (f, )
        # det_objectives_gradients = (g, )

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
        # Generate an Exponential(rate = 1) r.v.
        x = tuple([rand_sol_rng.lognormalvariate(lq=0.1, uq=10) for _ in range(self.dim)])
        return x

    # def set_linear_constraints(self):
    #     # Initialize linear constraint matrices.
    #     self.Ci = None
    #     self.Ce = None
    #     self.di = None
    #     self.de = None
    #     if self.constraint_type != "deterministic": # maybe create a new type of constraint named "linear"
    #         return
    #     else:
            # self.Ci = np.array([[1, 1],[-1, 2]])
            # self.di = np.array([[2], [2]])

            # self.Ce = np.array([[1, 1, 1, 1, 1]])
            # self.de = np.array([1]) # a simple linear constraint 1 x = 1
            # self.Ci = np.array([[1, 1, 1, 1, 1]])
            # self.di = np.array([1])

            # self.Ce = np.array([[1, 1, 1]])
            # self.de = np.array([1]) # a simple linear constraint 1 x = 1
            # self.Ci = np.array([[1, 1, 1]])
            # self.di = np.array([1])

            # self.Ce = np.array([[1, 1]])
            # self.de = np.array([1]) # a simple linear constraint 1 x = 1
            # self.Ci = np.array([[1, 1]])
            # self.di = np.array([1])

            # ## Constraints for Rosenbrock
            # # inequality constraint matrix
            # self.Ci = np.array([[1, 1, 1, 1, 1],
            #             [1, 1, 1, 0, -1],
            #             [1, 0, -1, -1, 1],
            #             [1, -1, 0, 1, 0], 
            #             [-1, -1, -1, -1, -1],
            #             [-1, -1, -1, 0, 1],
            #             [-1, 0, 1, 1, -1],
            #             [-1, 1, 0, -1, 0]])
            # # inequality constraint vector
            # self.di = np.array([[5, 3, 0, 1, -3, 1, 2, 1]])

