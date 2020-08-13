"""
Summary
-------
Minimize the mean sojourn time of an M/M/1 queue.
"""
import numpy as np
from base import Problem
from oracles.mm1queue import MM1Queue

class MM1MinMeanSojournTime(Problem):
    """
    Base class to implement simulation-optimization problems.

    Attributes
    ----------
    minmax : int (+/- 1)
        indicator of maximization (+1) or minimization (-1)
    dim : int
        number of decision variables
    constraint_type : string
        description of constraints types: 
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_available : bool
        indicates if gradient of objective function is available
    budget : int
        max number of replications (fn evals) for a solver to take
    optimal_bound : float
        bound on optimal objective function value
    optimal_solution : tuple
        optimal solution (if known)
    initial_solution : tuple
        default initial solution from which solvers start
    is_objective : list of bools
        indicates if response appears in objective function
    is_constraint : list of bools
        indicates if response appears in stochastic constraint
    oracle : Oracle object
        associated simulation oracle that generates replications

    Arguments
    ---------
    noise_factors : dict
        noise factors to pass through to the oracle

    See also
    --------
    base.Problem
    """
    def __init__(self, noise_factors={}):
        self.minmax = -1
        self.dim = 1
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.budget = 10000
        self.optimal_bound = 0
        self.optimal_solution = None
        self.inital_solution = [3]
        self.is_objective = [True, False]
        self.is_constraint = [False, False]
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.oracle = MM1Queue(noise_factors)

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
            "mu": vector[0]
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
        vector = (factor_dict["mu"],)
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
        objectives = (response_dict["avg_sojourn_time"],)
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
        stoch_constraints = (response_dict["frac_cust_wait"],)
        return stoch_constraints