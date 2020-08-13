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
    """
    def __init__(self):
        #self.oracle = None
        super().__init__()

    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : list
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
        vector : list
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
        objectives : list
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
        stoch_constraints : list
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = []
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
        det_objectives : list
            vector of deterministic components of objectives
        det_objectives_gradients : list
            vector of gradients of deterministic components of objectives
        """
        det_objectives = [0]*self.n_objectives
        det_objectives_gradients = [[0]*self.dim]*self.n_objectives
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
        det_stoch_constraints : list
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : list
            vector of gradients of deterministic components of stochastic constraints
        """
        det_stoch_constraints = [0]*self.n_stochastic_constraints
        det_stoch_constraints_gradients = [[0]*self.dim]*self.n_stochastic_constraints
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
            for _ in range(m):
                # generate one replication at x
                responses, gradients = self.oracle.replicate(solution.decision_factors)
                # increment counter 
                solution.n_reps += 1
                # convert gradient subdictionaries to vectors mapping to decision variables
                vector_gradients = {keys:self.factor_dict_to_vector(gradient_dict) for (keys, gradient_dict) in gradients.items()}
                # convert responses and gradients to objectives and gradients and add
                # to those of deterministic components of objectives
                new_objectives = [sum(pairs) for pairs in zip(self.response_dict_to_objectives(responses),solution.det_objectives)]
                new_objectives_gradients = [[sum(pairs) for pairs in zip(stoch_obj, det_obj)] for stoch_obj, det_obj in zip(self.response_dict_to_objectives(vector_gradients),solution.det_objectives_gradients)]
                # convert responses and gradients to stochastic constraints and gradients and add
                # to those of deterministic components of stochastic constraints
                new_stoch_constraints = [sum(pairs) for pairs in zip(self.response_dict_to_stoch_constraints(responses),solution.det_stoch_constraints)]
                new_stoch_constraints_gradients = [[sum(pairs) for pairs in zip(stoch_stoch_cons, det_stoch_cons)] for stoch_stoch_cons, det_stoch_cons in zip(self.response_dict_to_stoch_constraints(vector_gradients),solution.det_stoch_constraints_gradients)]
                # record
                solution.objectives.append(new_objectives)
                solution.objectives_gradients.append(new_objectives_gradients)
                solution.stoch_constraints.append(new_stoch_constraints)
                solution.stoch_constraints_gradients.append(new_stoch_constraints_gradients)                
                # advance rngs to start of next subsubstream
                for rng in self.oracle.rng_list:
                    rng.advance_subsubstream()

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

    def replicate(self, decision_factors):
        """
        Simulate a single replication at solution described by `decision_factors`.

        Arguments
        ---------
        decision_factors : dict
            decision factors of the simulation model

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
    dim : tuple
        number of decision variables describing `x`
    decision_factors : dict
        decision factor names and values
    n_reps : int
        number of replications run at the solution
    det_objectives : list
        deterministic components added to objectives
    det_objectives_gradients : list of lists (# objectives x dimension)
        gradients of deterministic components added to objectives
    objectives : list of lists (# replications x # objectives)
        objective(s) estimates from each replication
    objectives_gradients : list of lists of lists (# replications x # objectives x dimension)
        gradient estimates of objective(s) from each replication
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
        self.objectives = []
        self.objectives_gradients = []
        self.stoch_constraints = []
        self.stoch_constraints_gradients = []

    def response_mean(self, which):
        """
        Compute sample mean of specified responses.

        Arguments
        ---------
        which : list of bools of length n_responses
            responses of which to compute statistics 

        Returns
        -------
        response_mean : array of rank (n_responses)
            sample means of the specified responses
        """
        response_mean = np.mean(np.array(self.responses)[:,which], axis=0)
        return response_mean

    def response_var(self, which):
        """
        Compute sample variance of specified responses.

        Arguments
        ---------
        which : list of bools of length n_responses
            responses of which to compute statistics 

        Returns
        -------
        response_var : array of rank (n_responses)
            sample variances of the specified responses
        """
        response_var = np.var(np.array(self.responses)[:,which], axis=0, ddof=1)
        return response_var

    def response_std_error(self, which):
        """"
        Compute sample standard error of specified responses.

        Arguments
        ---------
        which : list of bools of length n_responses
            responses of which to compute statistics 

        Returns
        -------
        response_std_error : array of rank (n_responses)
            sample standard errors of the specified responses
        """
        response_std_error = np.std(np.array(self.responses)[:,which], axis=0, ddof=1)/np.sqrt(self.n_reps)
        return response_std_error

    def response_cov(self, which):
        """"
        Compute sample covariance of specified responses.

        Arguments
        ---------
        which : list of bools of length n_responses
            responses of which to compute statistics 

        Returns
        -------
        response_cov : array of rank (n_responses, n_responses)
            sample covariance matrix of the specified responses
        """
        response_cov = np.cov(np.array(self.responses)[:,which], rowvar=False, ddof=1)
        return response_cov
        
    def gradient_mean(self, which):
        """
        Compute sample mean of specified gradient components.

        Arguments
        ---------
        which : list of bools of length n_responses
            responses of which to compute statistics 

        Returns
        -------
        gradient_mean : array of rank (n_responses, dim)
            sample means of the components of the gradients of the specified responses
        """
        gradient_mean = np.mean(np.array(self.gradients)[:,which], axis=0)
        return gradient_mean

    def gradient_var(self, which):
        """
        Compute sample variance of specified gradient components.

        Arguments
        ---------
        which : list of bools of length n_responses
            responses of which to compute statistics 

        Returns
        -------
        gradient_var : array of rank (n_reponses, dim)
            sample variances of the components of the gradients of the specified responses
        """
        gradient_var = np.var(np.array(self.gradients)[:,which], axis=0, ddof=1)
        return gradient_var

    def gradient_std_error(self, which):
        """"
        Compute sample standard error of all gradient components for specified responses.

        Arguments
        ---------
        which : list of bools of length n_responses
            responses of which to compute statistics 

        Returns
        -------
        gradient_std_error : array of rank (n_reponses, dim)
            sample standard errors of the components of the gradients of the specified responses    
        """
        gradient_std_error = np.std(np.array(self.gradients)[:,which], axis=0, ddof=1)/np.sqrt(self.n_reps)
        return gradient_std_error

    def gradient_cov(self, which):
        """"
        Compute sample covariance of all gradient components,
        separately for specified responses.

        Arguments
        ---------
        which : list of bools of length n_responses
            responses of which to compute statistics 

        Returns
        -------
        response_cov : array of rank (n_responses, dim, dim)
            sample covariance matrices of the gradients, for the specified responses
        """
        gradient_cov = np.zeros((sum(which),self.dim,self.dim))
        for i in range(len(which)):
            if which[i]:
                new_index = sum(which[:i])
                sliced_gradient = [sublist[i] for sublist in self.gradients]
                gradient_cov[new_index,:,:] = np.cov(sliced_gradient, rowvar = False, ddof=1)
        return gradient_cov