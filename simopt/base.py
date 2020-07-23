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
    dim : int
        number of decision variables
    n_responses : int
        number of responses (performance measures)
    params : dict
        changeable parameters of the simulation model
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

    def check_simulatable(self, x):
        """
        Determine if a simulation replication can be run at solution `x`.

        Arguments
        ---------
        x : tuple
            solution to evaluate

        Returns
        -------
        issimulatable : bool
            indicates if `x` is simulatable
        """
        raise NotImplementedError

    def replicate(self, x):
        """
        Simulate a single replication at solution `x`.

        Arguments
        ---------
        x : tuple
            solution to evaluate

        Returns
        -------
        response : list
            performance measures of interest
        gradient : list of lists
            gradient estimate for each response
        """
        raise NotImplementedError

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
                response, gradient = self.replicate(solution.x)
                # increment counter 
                solution.n_reps += 1
                # append results
                solution.responses.append(response)
                solution.gradients.append(gradient)
                # advance rngs to start of next subsubstream
                for rng in self.rng_list:
                    rng.advance_subsubstream()

class Solution(object):
    """
    Base class for solutions, i.e., vectors of decision variables.

    Attributes
    ----------
    x : tuple
        vector of decision variables
    dim : tuple
        number of decision variables describing `x`
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
    """
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.dim = len(x)
        self.n_reps = 0
        self.responses = []
        self.gradients = []

    def response_mean(self):
        """
        Compute sample mean of all responses.

        Returns
        -------
        response_mean : array of rank (n_responses)
            sample means of the responses
        """
        response_mean = np.mean(self.responses, axis=0)
        return response_mean

    def response_var(self):
        """
        Compute sample variance of all responses.

        Returns
        -------
        response_var : array of rank (n_responses)
            sample variances of the responses
        """
        response_var = np.var(self.responses, axis=0, ddof=1)
        return response_var

    def response_std_error(self):
        """"
        Compute sample standard error of all responses.

        Returns
        -------
        response_std_error : array of rank (n_responses)
            sample standard errors of the responses
        """
        response_std_error = np.std(self.responses, axis=0, ddof=1)/np.sqrt(self.n_reps)
        return response_std_error

    def response_cov(self):
        """"
        Compute sample covariance of all responses.

        Returns
        -------
        response_cov : array of rank (n_responses, n_responses)
            sample covariance matrix of the responses
        """
        response_cov = np.cov(self.responses, rowvar=False, ddof=1)
        return response_cov
        
    def gradient_mean(self):
        """
        Compute sample mean of all gradient components.

        Returns
        -------
        gradient_mean : array of rank (n_responses, dim)
            sample means of the components of the gradients of the responses
        """
        gradient_mean = np.mean(self.gradients, axis=0)
        return gradient_mean

    def gradient_var(self):
        """
        Compute sample variance of all gradient components.

        Returns
        -------
        gradient_var : array of rank (n_reponses, dim)
            sample variances of the components of the gradients of the respones
        """
        gradient_var = np.var(self.gradients, axis=0, ddof=1)
        return gradient_var

    def gradient_std_error(self):
        """"
        Compute sample standard error of all gradient components.

        Returns
        -------
        gradient_std_error : array of rank (n_reponses, dim)
            sample standard errors of the components of the gradients of the respones    
        """
        gradient_std_error = np.std(self.gradients, axis=0, ddof=1)/np.sqrt(self.n_reps)
        return gradient_std_error

    def gradient_cov(self):
        """"
        Compute sample covariance of all gradient components,
        separately for each response.

        Returns
        -------
        response_cov : array of rank (n_responses, dim, dim)
            sample covariance matrices of the gradients, for each response
        """
        n_responses = len(self.responses[0])
        #dim = len(self.gradients[0][0])
        gradient_cov = np.zeros((n_responses,self.dim,self.dim))
        for i in range(n_responses):
            gradient_cov[i,:,:] = np.cov(self.gradients[:,i,:], rowvar=False, ddof=1)
        return gradient_cov