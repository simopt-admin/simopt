"""
Summary
-------
Minimize the total cost of a facilitysize problem.
"""
import numpy as np
from base import Problem
from oracles.facilitysizing import facilitysize

class facilitysizingTotalCost(Problem):
    """
    Base class to implement simulation-optimization problems.

    Attributes
    ----------
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
    gradient_available : bool
        indicates if gradient of objective function is available
    initial_solution : tuple
        default initial solution from which solvers start
    budget : int
        max number of replications (fn evals) for a solver to take
    optimal_bound : float
        bound on optimal objective function value
    optimal_solution : tuple
        optimal solution (if known)
    oracle : Oracle object
        associated simulation oracle that generates replications

    Arguments
    ---------
    oracle_factors : dict
        subset of non-decision factors to pass through to the oracle

    See also
    --------
    base.Problem
    """
    def __init__(self, oracle_fixed_factors={}):
        self.minmax = (-1,)
        self.dim = 3
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.constraint_type = "stochastic"
        self.variable_type = "discrete"
        self.gradient_available = False
        self.budget = 10000
        self.optimal_bound = 0
        self.optimal_solution = None
        self.inital_solution = ([150, 300, 250],)
        self.oracle_default_factors = {
            "cost": [1, 1, 1],
            "epsilon": 0.05
        }
        super().__init__(oracle_fixed_factors)
        # Instantiate oracle with fixed factors and over-riden defaults
        self.oracle = facilitysize(self.oracle_fixed_factors)
    
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
            "capa": vector[:]
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
        vector = (factor_dict["capa"],)
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
        stoch_constraints = (response_dict["stock_out_flag"],)
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
        det_stoch_constraints = (-self.oracle_default_factors["epsilon"],)
        det_stoch_constraints_gradients = ((0,),)
        return det_stoch_constraints, det_stoch_constraints_gradients
    
    # c*X, c     
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
        det_objectives = (np.dot(self.oracle_default_factors["cost"],x),)
        det_objectives_gradients = ((self.oracle_default_factors["cost"],),)
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

    def get_random_solution(self):
        """
        Generate a random solution, to be used for starting or restarting solvers.

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = 100*np.random.rand(self.dim)
        return x
