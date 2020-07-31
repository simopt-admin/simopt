"""
Summary
-------
Simulate a M/M/1 queue.
"""
from base import Oracle
import numpy as np

class MM1Queue(Oracle):
    """
    An oracle that simulates an M/M/1 queue with an Exponential(lambda) 
    interarrival time distribution and an Exponential(x) service time 
    distribution. Returns an estimate of the average steady-state sojourn time
    and the steady-state fraction of customers who wait.

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
    specifications : dict
        details of each factor (for GUI and data validation)

    Arguments
    ---------
    noise_factors : nested dict
        noise_factors of the simulation model

    See also
    --------
    base.Oracle
    """
    def __init__(self, noise_factors={}):
        self.n_rngs = 2
        self.n_responses = 2
        self.factors = noise_factors
        self.specifications = {
            "lambda": {
                "description": "Rate parameter of interarrival time distribution.",
                "datatype": float
            },
            "mu": {
                "description": "Rate parameter of service time distribution.",
                "datatype": float
            },
            "warmup": {
                "description": "Number of people as warmup before collecting statistics",
                "datatype": int
            },
            "people": {
                "description": "Number of people from which to calculate the average sojourn time",
                "datatype": int
            }
        }
        self.check_factor_list = {
            "lambda": self.check_lambda,
            "mu": self.check_mu,
            "warmup": self.check_warmup,
            "people": self.check_people
        }
        # # Default parameters
        # self.default_params = {
        #     "lambd": 2, # rate parameter of interarrival time
        #     "warmup": 20, # number of people as warmup before collecting statistics
        #     "people": 50 # number of people from which to calculate the average sojourn time
        # }
        # # Set parameters of the simulation oracle -> fill in missing entries with defaults
        # for key in self.default_params:
        #     if key not in params:
        #         params[key] = self.default_params[key]
        # self.params = params

    def check_lambda(self):
        return self.factors["lambda"] > 0

    def check_mu(self):
        return self.factors["mu"] > 0

    def check_warmup(self):
        return self.factors["warmup"] >= 1

    def check_people(self):
        return self.factors["people"] >= 1

    def check_simulatable_factors(self):
        #demo for condition that queue must be stable
        #return self.factors["mu"] > self.factors["lambda"]
        return True

    def replicate(self, decision_factors):
        """
        Simulate a single replication at solution described by `decision_factors`.

        Arguments
        ---------
        decision_factors : dict
            decision factors of the simulation model

        Returns
        -------
        response : list
            performance measures of interest
            response[0] = average sojourn time
            response[1] = fraction of customers who wait
        gradient : list of dicts
            gradient estimates for each response
        """
        # set the decision factors of the model
        self.factors.update(decision_factors) 
        # total number of arrivals to simulate
        total = self.factors["warmup"] + self.factors["people"]
        # designate separate random number generators
        arrival_rng = self.rng_list[0]
        service_rng = self.rng_list[1]
        # generate all interarrival times up front
        arrival_times = [arrival_rng.expovariate(self.factors["lambda"]) for _ in range(total)]
        # generate all service times up front
        service_times = [service_rng.expovariate(self.factors["mu"]) for _ in range(total)]
        # create matrix storing times and metrics for each customer
        cust_mat = np.zeros((total, 6))
        # column 0 : arrival time to queue
        cust_mat[:,0] = np.cumsum(arrival_times)
        # column 1 : service time
        cust_mat[:,1] = service_times
        # column 2 : service completion time
        # column 3 : sojourn time
        # column 4 : number of customers in system at arrival
        # column 5 : gradient of sojourn time
        # input first customer times
        cust_mat[0,2] = cust_mat[0,0] + cust_mat[0,1]
        cust_mat[0,3] = cust_mat[0,1]
        cust_mat[0,4] = 0
        cust_mat[0,5] = -cust_mat[0,1]/self.factors["mu"]
        # fill in times for remaining customers
        for i in range(1,total):
            cust_mat[i,2] = max(cust_mat[i,0], cust_mat[i-1,2]) + cust_mat[i,1]
            cust_mat[i,3] = cust_mat[i,2] - cust_mat[i,0]
            cust_mat[i,4] = sum(cust_mat[i-int(cust_mat[i-1,4])-1:i,2] > cust_mat[i,0])
            cust_mat[i,5] = -sum(cust_mat[i-int(cust_mat[i,4]):i+1,1])/self.factors["mu"]
        # with np.printoptions(precision=3, suppress=True):
        #     print(cust_mat)    
        # compute mean sojourn time and its gradient
        mean_sojourn_time = np.mean(cust_mat[self.factors["warmup"]:,3])
        grad_mean_sojourn_time = np.mean(cust_mat[self.factors["warmup"]:,5])
        # compute fraction of customers who wait
        fraction_wait = np.mean(cust_mat[self.factors["warmup"]:,4] > 0)
        # return mean sojourn time w/ gradient estimate
        # return fraction who wait w/o gradient estimate
        response = [mean_sojourn_time, fraction_wait]
        mean_sojourn_time_grad = {
            "mu": grad_mean_sojourn_time,
            "lambda": np.nan, # to be derived...
            "warmup": np.nan,
            "people": np.nan 
        }
        fraction_wait_grad = {
            "mu": np.nan,
            "lambda": np.nan,
            "warmup": np.nan,
            "people": np.nan
        }
        gradient = []
        gradient.append(dict((key, mean_sojourn_time_grad[key]) for key in decision_factors))
        gradient.append(dict((key, fraction_wait_grad[key]) for key in decision_factors))
        return response, gradient