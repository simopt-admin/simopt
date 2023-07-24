"""
Summary
-------
"""
import numpy as np

#from base import Model, Problem
from simopt.base import Model, Problem

class bus_model(Model):
    """
    Modeling a bus scheduling problem where passengers are coming in Poi(lamb)
    in the interval [0,1]. Define a variable x \in (0,1) to minimize the total
    waiting time of N passengers. Arrival times are T1, T2,...,TN
    
    total wait time = sum^N_{i=1}[xI(Ti <= x) + I(Ti > x) - Ti]
    """
    
    def __init__(self, fixed_factors=None):
            
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "BUS"
        self.n_responses = 1
        self.specifications = {
            "num_buses": {
                "description": "number of buses excluding the last one",
                "datatype": int,
                "default": 1
            },           
            "arrival_rate": {
                "description": "people's arrival rate by Poisson",
                "datatype": float,
                "default": 1
            },
            "scheduled_time": {
                "description": "Scheduled Bus Time",
                "datatype": float,
                "default": (0.5,)
            }
            
        }
        self.check_factor_list = {
            "num_buses": self.check_num_buses,
            "arrival_rate": self.check_arrival_rate,
            "scheduled_time":self.check_scheduled_time
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)
        self.n_rngs = 1
        #self.x = self.factors["scheduled_time"]

     
    def check_num_buses(self):
        return self.factors['num_buses'] > 0
    
    def check_arrival_rate(self):
        return self.factors['arrival_rate'] > 0
    
    def check_scheduled_time(self):
        given_time = self.factors['scheduled_time']
    
    def indicator(self,T,x):
        '''
        an indicator I(T <= x)
        '''
        if(T <= x):
            return 1
        else:
            return 0
    
    def two_side_indicator(self,T,x,y):
        '''
        an indicator function of whether 
        T \in (x, y] where x <= y, or
        I(x < T <= y)
        '''
        if(T <= x or T > y):
            return 0
        else:
            return 1
        
    def get_waiting_time_multiple_person(self,T,x):
        '''
        get waiting for a person of arrival time T where
        we have multiple buses, i.e. len(x) > 1. Here, x
        must be a vector (array) of sorted scheduled time
        '''
        s = 0
        #num_buses = len(x)
        s += x[0]*self.indicator(T,x[0]) + (1-self.indicator(T,x[-1])) - T
        for i in range(self.factors['num_buses']-1):
            s += x[i+1]*self.two_side_indicator(T,x[i],x[i+1])
        return s
        
    def get_waiting_time_multiple_schedule(self,arrivals,x):
        '''
        x is a vector, i.e. there are len(x)+1 buses and there are 
        multiple arrivals
        '''
        #num_buses = len(x)
        time = 0
        
        #arrivals = self.get_arrival_times()
        num_arrivals = len(arrivals)
        
        for i in range(num_arrivals):
            time += self.get_waiting_time_multiple_person(arrivals[i],x)
        return time   

    def get_waiting_time_single_schedule(self,arrivals,x):
        '''
        return the objective value, i.e.
        
        obj = sum^N_{i=1}[xI(Ti <= x) + I(Ti > x) - Ti]
        
        from a single scheduled arrival time x and the arrivals
        '''
        
        #arrivals = self.get_arrival_times()
        num_arrivals = len(arrivals)
        
        s = 0
        for i in range(num_arrivals):
            s += x*self.indicator(arrivals[i],x) + (1-self.indicator(arrivals[i],x)) - arrivals[i]
            
        return s
    
    def replicate(self, rng_list):
        
        '''
        get passengers' arrival times array in [0,1]
        according to Poisson(rate) and 
        
        return the objective value, i.e.
        
        obj = sum^N_{i=1}[xI(Ti <= x) + I(Ti > x) - Ti]
        
        from the scheduled time x
        '''
        
        T = 0
        arrivals = []
        
        exp_rng = rng_list[0]
        i = 1
        #first arraival
        #t = np.random.exponential(1/self.rate)
        t = exp_rng.expovariate(1 / self.factors["arrival_rate"])
        T = T + t
        
        #generate arrivals
        while(T < 1):
            arrivals.append(T)
            #t = np.random.exponential(1/self.rate)
            #t = rng_list[i].expovariate(1 / self.factors["arrival_rate"])
            t = exp_rng.expovariate(1 / self.factors["arrival_rate"])
            T = T + t
            i = i + 1
            
        if(len(self.factors["scheduled_time"]) > 1):  
            #result = self.get_waiting_time(arrivals,np.array(self.x))
            result = self.get_waiting_time_multiple_schedule(arrivals,np.array(self.factors["scheduled_time"]))
        else:
            result = self.get_waiting_time_single_schedule(arrivals,np.array(self.factors["scheduled_time"])[0])
        
        # Calculate responses from simulation data.
        responses = {"total_waiting_time": result
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients

class bus_problem(Problem):
    
    def __init__(self, name="BUS", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"scheduled_time"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (0.5,)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 10000
            },
         
            "Ci":{
                "description": "Coefficients for inequality constraints Ce@x <= de",
                "datatype": "matrix",
                "default": -1 * np.ones(1)
            },
            "di":{
                "description": "RHS for inequality constraints Ce@x <= de",
                "datatype": "vector",
                "default": np.array([0])
            }    
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = bus_model(self.model_fixed_factors)
        self.dim = self.model.factors["num_buses"]
        #self.Ci = -1 * np.ones(13)
        self.Ci = self.factors["Ci"]
        self.Ce = None
        #self.di = -1 * np.array([self.factors["sum_lb"]])
        self.di = self.factors["di"]
        self.de = None
        
        self.lower_bounds = (0,)*self.dim
        self.upper_bounds = (1,)*self.dim
        
        
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
            "scheduled_time": vector[:]
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
        vector = tuple(factor_dict["scheduled_time"])
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
        objectives = (response_dict["total_waiting_time"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

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
        det_stoch_constraints_gradients = ((0,) * self.dim,)  # tuple of tuples – of sizes self.dim by self.dim, full of zeros
        return det_stoch_constraints, det_stoch_constraints_gradients

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
        det_objectives = (0,)#None
        det_objectives_gradients = ((0,) * self.dim,)
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
        is_satisfy = True
        
        if((self.Ci is not None) and (self.di is not None)):
            is_satisfy = is_satisfy and (self.Ci.dot(x) <= self.di)
            
        if((self.Ce is not None) and (self.de is not None)):
            is_satisfy = is_satisfy and (self.Ce.dot(x) <= self.de)
        
        return is_satisfy
        
        #return self.Ci.dot(x) <= self.di

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
 
        return rand_sol_rng.random()   
        

