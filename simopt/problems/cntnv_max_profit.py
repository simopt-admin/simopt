"""
Summary
-------
Maximize the expected profit for the continuous newsvendor problem.
"""
import numpy as np
from base import Problem
from oracles.cntnv import CntNV


class CntNVMaxProfit(Problem):
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
    ref_optimal_solution : tuple
        reference solution (in lieu of optimal)
    oracle : Oracle object
        associated simulation oracle that generates replications
    oracle_default_factors : dict
        default values for overriding oracle-level default factors
    oracle_fixed_factors : dict
        combination of overriden oracle-level factors and defaults
    rng_list : list of rng.MRG32k3a objects
        list of random number generators used to generate a random initial solution
        or a random problem instance

    Arguments
    ---------
    oracle_factors : dict
        subset of non-decision factors to pass through to the oracle

    See also
    --------
    base.Problem
    """
    def __init__(self, oracle_fixed_factors={}):
        self.dim = 1
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.budget = 1000
        self.optimal_bound = 0
        self.optimal_solution = (0.1878,)
        self.initial_solution = (0,)
        self.ref_optimal_solution = (0.1878,)
        self.oracle_default_factors = {
            "purchase_price": 5.0,
            "sales_price": 9.0,
            "salvage_price": 1.0,
            "Burr_c": 2.0,
            "Burr_k": 20.0
            }
        super().__init__(oracle_fixed_factors)
        # Instantiate oracle with fixed factors and over-riden defaults
        self.oracle = CntNV(self.oracle_fixed_factors)

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
            "order_quantity": vector[0]
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
        vector = (factor_dict["order_quantity"],)
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
        objectives = (response_dict["profit"],)
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
        det_objectives = (0,)
        det_objectives_gradients = ((0,),)
        return det_objectives, det_objectives_gradients

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
        return x[0] > 0

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution, to be used for starting or restarting solvers.

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
        x = (rand_sol_rng.expovariate(1),)
        return x