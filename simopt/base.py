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
                response, gradient = self.oracle.replicate(solution.decision_factors)
                # increment counter 
                solution.n_reps += 1
                # append results
                solution.responses.append(response)
                # convert list of gradient dictionaries to list of vectors
                slim_gradients = [self.factor_dict_to_vector(gradient_dict) for gradient_dict in gradient]
                solution.gradients.append(slim_gradients)
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

    def check_simulatable_factors(self):
        """
        Determine if a simulation replication can be run with the given factors.

        Returns
        -------
        is_simulatable : bool
            indicates if oracle specified by factors is simulatable
        """
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
        Simulate a single replication at solution described by decision factors.

        Arguments
        ---------
        decision_factors : dict
            decision factor names and values

        Returns
        -------
        response : list
            performance measures of interest
        gradient : list of dicts
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
    responses: list of lists (# replications x # responses)
        performance measures of interest from each replication
    gradients: list of lists of lists (# replications x # responses x dimension)
        gradient estimate for each response from each replication
    
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
        self.responses = []
        self.gradients = []

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