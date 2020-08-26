#!/usr/bin/env python
"""
Summary
-------
Provide base classes for solvers, problems, and oracles.

Listing
-------
Solver : class
Problem : class
Oracle : class
Solution : class
"""

import numpy as np
from rng.mrg32k3a import MRG32k3a

class Problem(object):
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
    """
    def __init__(self):
        self.oracle = None
        super().__init__()

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        stoch_constraints = ()
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
        det_objectives = (0,)*self.n_objectives
        det_objectives_gradients = tuple([(0,)*self.dim for _ in range(self.n_objectives)])
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
        det_stoch_constraints = (0,)*self.n_stochastic_constraints
        det_stoch_constraints_gradients = tuple([(0,)*self.dim for _ in range(self.n_stochastic_constraints)])
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
        return True

    def get_random_solution(self):
        """
        Generate a random solution, to be used for starting or restarting solvers.

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        pass
    
    def attach_rng(self, rng):
        """
        Attach random number generator to the problem.

        Arguments
        ---------
        rng : rng.MRG32k3a object
            random-number generator used to generate random solutions
        """
        self.rng = rng


    def simulate(self, solution, m=1):
        """
        Simulate `m` i.i.d. replications at solution `x`.

        Arguments
        ---------
        solution : Solution object
            solution to evalaute
        m : int
            number of replications to simulate at `x`
        """
        if m < 1:
            print('--* Error: Number of replications must be at least 1. ')
            print('--* Aborting. ')
        else:
            # pad numpy arrays if necessary
            if solution.n_reps + m > solution.storage_size:
                solution.pad_storage(m)
            # set the decision factors of the oracle
            self.oracle.factors.update(solution.decision_factors) 
            for _ in range(m):
                # generate one replication at x
                responses, gradients = self.oracle.replicate()
                # convert gradient subdictionaries to vectors mapping to decision variables
                vector_gradients = {keys:self.factor_dict_to_vector(gradient_dict) for (keys, gradient_dict) in gradients.items()}
                # convert responses and gradients to objectives and gradients and add
                # to those of deterministic components of objectives
                solution.objectives[solution.n_reps] = [sum(pairs) for pairs in zip(self.response_dict_to_objectives(responses),solution.det_objectives)]            
                solution.objectives_gradients[solution.n_reps] = [[sum(pairs) for pairs in zip(stoch_obj, det_obj)] for stoch_obj, det_obj in zip(self.response_dict_to_objectives(vector_gradients),solution.det_objectives_gradients)]
                # convert responses and gradients to stochastic constraints and gradients and add
                # to those of deterministic components of stochastic constraints
                solution.stoch_constraints[solution.n_reps] = [sum(pairs) for pairs in zip(self.response_dict_to_stoch_constraints(responses),solution.det_stoch_constraints)]
                solution.stoch_constraints_gradients[solution.n_reps] = [[sum(pairs) for pairs in zip(stoch_stoch_cons, det_stoch_cons)] for stoch_stoch_cons, det_stoch_cons in zip(self.response_dict_to_stoch_constraints(vector_gradients),solution.det_stoch_constraints_gradients)]
                # increment counter 
                solution.n_reps += 1
                # advance rngs to start of next subsubstream
                for rng in self.oracle.rng_list:
                    rng.advance_subsubstream()
            # update summary statistics
            solution.recompute_summary_statistics()
            
class Oracle(object):
    """
    Base class to implement simulation oracles (models) featured in
    simulation-optimization problems.

    Attributes
    ----------
    n_rngs : int
        number of random-number generators used to run a simulation replication
    rng_list : list of rng.MRG32k3a objects
        list of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : nested dict
        details of each factor (for GUI and data validation)
    """
    def __init__(self):
        super().__init__()

    def attach_rngs(self, rng_list):
        """
        Attach a list of random number generators to the oracle.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            list of random-number generators used to run a simulation replication
        """
        self.rng_list = rng_list

    def check_simulatable_factor(self, factor_name):
        """
        Determine if a simulation replication can be run with the given factor.

        Arguments
        ---------
        factor_name : string
            name of factor for dictionary lookup (i.e., key)

        Returns
        -------
        is_simulatable : bool
            indicates if oracle specified by factors is simulatable
        """
        is_simulatable = True
        is_simulatable *= self.check_factor_datatype(factor_name)
        is_simulatable *= self.check_factor_list[factor_name]()
        return is_simulatable
        #raise NotImplementedError

    def check_simulatable_factors(self, decision_factors):
        """
        Determine if a simulation replication can be run with the given factors.

        Arguments
        ---------
        decision_factors : dict
            decision factors of the simulation model
        
        Returns
        -------
        is_simulatable : bool
            indicates if oracle specified by factors is simulatable
        """
        self.factors.update(decision_factors) 
        return True
        #raise NotImplementedError

    def check_factor_datatype(self, factor_name):
        """
        Determine if a factor's data type matches its specification.

        Returns
        -------
        is_right_type : bool
            indicates if factor is of specified data type
        """
        is_right_type = isinstance(self.factors[factor_name], self.specifications[factor_name]["datatype"])
        return is_right_type

    def replicate(self):
        """
        Simulate a single replication for the current oracle factors.

        Returns
        -------
        responses : dict
            performance measures of interest
        gradients : dict of dicts
            gradient estimate for each response
        """
        raise NotImplementedError

class Solution(object):
    """
    Base class for solutions represented as vectors of decision variables
    and dictionaries of decision factors.

    Attributes
    ----------
    x : tuple
        vector of decision variables
    dim : int
        number of decision variables describing `x`
    decision_factors : dict
        decision factor names and values
    n_reps : int
        number of replications run at the solution
    det_objectives : tuple
        deterministic components added to objectives
    det_objectives_gradients : tuple of tuples (# objectives x dimension)
        gradients of deterministic components added to objectives
    det_stoch_constraints : tuple
        deterministic components added to LHS of stochastic constraints
    det_stoch_constraints_gradients : tuple (# stochastic constraints x dimension)
        gradients of deterministics components added to LHS stochastic constraints
    storage_size : int
        max number of replications that can be recorded in current storage 
    objectives : numpy array (# replications x # objectives)
        objective(s) estimates from each replication
    objectives_gradients : numpy array (# replications x # objectives x dimension)
        gradient estimates of objective(s) from each replication
    stochastic_constraints : numpy array (# replications x # stochastic constraints)
        stochastic constraint estimates from each replication
    stochastic_constraints_gradients : numpy array (# replications x # stochastic constraints x dimension)
        gradient estimates of stochastic constraints from each replication

    Arguments
    ---------
    x : tuple
        vector of decision variables
    problem : Problem object
        problem to which x is a solution
    """
    def __init__(self, x, problem):
        super().__init__()
        self.x = x
        self.dim = len(x)
        self.decision_factors = problem.vector_to_factor_dict(x)
        self.n_reps = 0
        self.det_objectives, self.det_objectives_gradients = problem.deterministic_objectives_and_gradients(self.x)
        self.det_stoch_constraints, self.det_stoch_constraints_gradients = problem.deterministic_stochastic_constraints_and_gradients(self.x)
        init_size = 100 # initialize numpy arrays to store up to 100 replications
        self.storage_size = init_size
        # Raw data
        self.objectives = np.zeros((init_size, problem.n_objectives))
        self.objectives_gradients = np.zeros((init_size, problem.n_objectives, problem.dim))
        self.stoch_constraints = np.zeros((init_size, problem.n_stochastic_constraints))
        self.stoch_constraints_gradients = np.zeros((init_size, problem.n_stochastic_constraints, problem.dim))
        # Summary statistics
        # self.objectives_mean = np.full((problem.n_objectives), np.nan)
        # self.objectives_var = np.full((problem.n_objectives), np.nan)
        # self.objectives_stderr = np.full((problem.n_objectives), np.nan)
        # self.objectives_cov = np.full((problem.n_objectives, problem.n_objectives), np.nan)
        # self.objectives_gradients_mean = np.full((problem.n_objectives, problem.dim), np.nan)
        # self.objectives_gradients_var = np.full((problem.n_objectives, problem.dim), np.nan)
        # self.objectives_gradients_stderr = np.full((problem.n_objectives, problem.dim), np.nan)
        # self.objectives_gradients_cov = np.full((problem.n_objectives, problem.dim, problem.dim), np.nan)
        # self.stoch_constraints_mean = np.full((problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_var = np.full((problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_stderr = np.full((problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_cov = np.full((problem.n_stochastic_constraints, problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_gradients_mean = np.full((problem.n_stochastic_constraints, problem.dim), np.nan)
        # self.stoch_constraints_gradients_var = np.full((problem.n_stochastic_constraints, problem.dim), np.nan)
        # self.stoch_constraints_gradients_stderr = np.full((problem.n_stochastic_constraints, problem.dim), np.nan)
        # self.stoch_constraints_gradients_cov = np.full((problem.n_stochastic_constraints, problem.dim, problem.dim), np.nan)

    def pad_storage(self, m):
        """
        Append zeros to numpy arrays for summary statistics
        
        Arguments
        ---------
        m : int
            number of replications to simulate
        """
        # Size of data storage
        n_objectives = len(self.det_objectives)
        n_stochastic_constraints = len(self.det_stoch_constraints)
        base_pad_size = 100 # default is to append space for 100 more replications
        # if more space needed, append in multiples of 100
        pad_size = int(np.ceil(m/base_pad_size))*base_pad_size
        self.storage_size += pad_size
        self.objectives = np.concatenate((self.objectives, np.zeros((pad_size, n_objectives))))
        self.objectives_gradients = np.concatenate((self.objectives_gradients, np.zeros((pad_size, n_objectives, self.dim))))
        self.stoch_constraints = np.concatenate((self.stoch_constraints, np.zeros((pad_size, n_stochastic_constraints))))
        self.stoch_constraints_gradients = np.concatenate((self.stoch_constraints_gradients, np.zeros((pad_size, n_stochastic_constraints, self.dim))))

    def recompute_summary_statistics(self):
        """
        Recompute summary statistics of the solution.
        """
        self.objectives_mean = np.mean(self.objectives[:self.n_reps], axis=0) 
        self.objectives_var = np.var(self.objectives[:self.n_reps], axis=0, ddof=1)
        self.objectives_stderr = np.std(self.objectives[:self.n_reps], axis=0, ddof=1)/np.sqrt(self.n_reps)
        self.objectives_cov = np.cov(self.objectives[:self.n_reps], rowvar=False, ddof=1)
        self.objectives_gradients_mean = np.mean(self.objectives_gradients[:self.n_reps], axis=0) 
        self.objectives_gradients_var = np.var(self.objectives_gradients[:self.n_reps], axis=0, ddof=1)
        self.objectives_gradients_stderr = np.std(self.objectives_gradients[:self.n_reps], axis=0, ddof=1)/np.sqrt(self.n_reps)
        self.objectives_gradients_cov = np.array([np.cov(self.objectives_gradients[:self.n_reps,obj], rowvar=False, ddof=1) for obj in range(len(self.det_objectives))])
        self.stoch_constraints_mean = np.mean(self.stoch_constraints[:self.n_reps], axis=0) 
        self.stoch_constraints_var = np.var(self.stoch_constraints[:self.n_reps], axis=0, ddof=1)
        self.stoch_constraints_stderr = np.std(self.stoch_constraints[:self.n_reps], axis=0, ddof=1)/np.sqrt(self.n_reps)
        self.stoch_constraints_cov = np.cov(self.stoch_constraints[:self.n_reps], rowvar=False, ddof=1)
        self.stoch_constraints_gradients_mean = np.mean(self.stoch_constraints_gradients[:self.n_reps], axis=0) 
        self.stoch_constraints_gradients_var = np.var(self.stoch_constraints_gradients[:self.n_reps], axis=0, ddof=1)
        self.stoch_constraints_gradients_stderr = np.std(self.stoch_constraints_gradients[:self.n_reps], axis=0, ddof=1)/np.sqrt(self.n_reps)
        self.stoch_constraints_gradients_cov = np.array([np.cov(self.stoch_constraints_gradients[:self.n_reps,stcon], rowvar=False, ddof=1) for stcon in range(len(self.det_stoch_constraints))])