"""
Summary
-------
Simulate a GI/G/1 queue.
"""
from base import Oracle
import numpy as np

class GIG1Queue(Oracle):
    """
    An oracle that simulates a GI/G/1 queue with an Exponential(lambda) 
    interarrival time distribution and an Exponential(x) service time 
    distribution. Returns an estimate of the average steady-state sojourn time.

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
    lambd : float (nonnegative)
        interarrival rate parameter for exponential distribution

    Arguments
    ---------
    params : dict
        dictionary of changeable parameters of the simulation model

    See also
    --------
    base.Oracle
    """
    def __init__(self, params={}):
        self.n_rngs = 2
        self.dim = 1
        self.n_responses = 1
        # Default parameters
        self.default_params = {
            "lambd": 2, # rate parameter of interarrival time
            "warmup": 20, # number of people as warmup before collecting statistics
            "people": 50 # number of people from which to calculate the average sojourn time
        }
        # Set parameters of the simulation oracle -> fill in missing entries with defaults
        for key in self.default_params:
            if key not in params:
                params[key] = self.default_params[key]
        self.params = params

    def check_simulatable(self, x):
        """
        Determine if a simulation replication can be run at solution `x`.

        Arguments
        ---------
        x : list of length 1
            solution to evalaute

        Returns
        -------
        issimulatable : bool
            indicates if `x` is simulatable
        """
        if len(x) == 1 and x[0] > 0:
            issimulatable = True
        else:
            issimulatable = False
        return issimulatable

    def simulate(self, x):
        """
        Simulate a single replication at solution `x`.

        Arguments
        ---------
        x : list of length 1
            solution to evaluate

        Returns
        -------
        response : list
            performance measures of interest
            response[0] = average sojourn time
        gradient : list of lists
            gradient estimate for each response
        """
        # extract parameters
        lambd = self.params["lambd"]
        warmup = self.params["warmup"]
        people = self.params["people"]
        mu = x[0]
        # total number of arrivals to simulate
        total = warmup + people
        # designate separate random number generators
        arrival_rng = self.rng_list[0]
        service_rng = self.rng_list[1]
        # generate all interarrival times up front
        arrival_times = [arrival_rng.expovariate(lambd) for _ in range(total)]
        # generate all service times up front
        service_times = [service_rng.expovariate(mu) for _ in range(total)]
        # create matrix storing times for each customer
        cust_mat = np.zeros((total, 4))
        # column 1 : arrival time to queue
        cust_mat[:,0] = np.cumsum(arrival_times)
        # column 2 : service time
        cust_mat[:,1] = service_times
        # column 3 : service completion time
        # column 4 : sojourn time
        # input first customer times
        cust_mat[0,2] = cust_mat[0,0] + cust_mat[0,1]
        cust_mat[0,3] = cust_mat[0,1]
        # fill in times for remaining customers
        for i in range(1,total):
            cust_mat[i,2] = max(cust_mat[i,0], cust_mat[i-1,2]) + cust_mat[i,1]
            cust_mat[i,3] = cust_mat[i,2] - cust_mat[i,0]
        # compute mean sojourn time
        mean_sojourn_time = np.mean(cust_mat[warmup:,3])
        # return mean sojourn time w/o gradient estimate
        response = [mean_sojourn_time]
        gradient = [[None]]
        return response, gradient