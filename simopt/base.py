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
aggregate : function
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

    def simulate(self, x):
        """
        Simulate a single replication at solution `x`.

        Arguments
        ---------
        x : list
            solution to evaluate

        Returns
        -------
        response : list
            performance measures of interest
        gradient : list of lists
            gradient estimate for each response
        """
        raise NotImplementedError

    def batch(self, x, m):
        """
        Simulate `m` i.i.d. replications at solution `x`.

        Arguments
        ---------
        x : list
            feasible solution to evalaute
        m : int
            number of replications to simulate at `x`

        Returns
        -------
        responses : array of rank (m, n_responses)
            performance measures of interest from each replication
        gradients : array of rank (m, n_responses, dim)
            gradient estimate for each response from each replication
        """
        if m < 1:
            print('--* Error: Number of replications must be at least 1. ')
            print('--* Aborting. ')
        else:
            responses = np.zeros((m, self.n_responses))
            gradients = np.zeros((m, self.n_responses, self.dim))
            for i in range(m):
                # generate one replication at x
                response, gradient = self.simulate(x)
                # append results
                responses[i] = response
                gradients[i] = gradient
                # advance rngs to start of next subsubstream
                for rng in self.rng_list:
                    rng.advance_subsubstream()
        return responses, gradients
   
def aggregate(responses, gradients):
    """
    Compute summary statistics from multiple replications taken at a
    given solution, e.g., the mean and covariance of the estimators of 
    each response (and their gradients).
    
    Arguments
    ---------
    responses : array of rank (m, n_responses)
        performance measures of interest from each replication
    gradients : array of rank (m, n_responses, dim)
        gradient estimate for each response from each replication
    
    Returns
    -------
    response_mean : array of rank (n_responses)
        means of the responses
    response_cov : array of rank (n_responses, n_responses)
        covariances of the responses
    gradient_mean : array of rank (n_responses, dim)
        means of the gradients of the responses
    gradient_cov : array of rank (n_responses, dim, dim)
        covariances of the gradients of the responses
    """
    # extract dimensions
    #num_reps = len(responses)
    num_responses = len(responses[0])
    dim = len(gradients[0,0])
    # calculate summary statistics for responses
    response_mean = np.mean(responses, axis=0)
    response_cov = np.cov(responses, rowvar=False)
    # calculate summary statistics for gradients
    gradient_mean = np.mean(gradients, axis=0)
    gradient_cov = np.zeros((num_responses,dim,dim))
    for i in range(num_responses):
        gradient_cov[i,:,:] = np.cov(gradients[:,i,:], rowvar=False)
    return response_mean, response_cov, gradient_mean, gradient_cov