"""
Summary
-------
Simulate an open jackson network 
"""
import numpy as np
import math as math
import sys

from ..base import Model, Problem

class OpenJackson(Model):
    """
    A model of an open jackson network .

    Attributes
    ----------
    name : string
        name of model
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : dict
        fixed_factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "OPENJACKSON"
        self.n_responses = 2
        self.factors = fixed_factors
        self.specifications = {
            "number_queues": {
                "description": "The number of queues in the network",
                "datatype": int,
                "default": 5
            },
            "arrival_alphas": {
                "description": "The arrival rates to each queue from outside the network",
                "datatype": list,
                "default": [1,4,1,4,0.5]
            },
            "service_mus": {
                "description": "The mu values for the exponential service times ",
                "datatype": tuple,
                "default": (8.85,9.45,8.85,11.63,10.8)
            },
            "routing_matrix": {
                "description": "The routing matrix that describes the probabilities of moving to the next queue after leaving the current one",
                "datatype": list,
                "default": [[0.1, 0.1, 0.2, 0.2, 0],
                            [0.1, 0.1, 0.2, 0.2, 0],
                            [0.2, 0.1, 0, 0.1, 0.2],
                            [0.1, 0.1, 0.1, 0, 0.2],
                            [0.1, 0.1, 0.1, 0.1, 0.2]]
            },
            "t_end": {
                "description": "A number of replications to run",
                "datatype": int,
                "default": 500
            },
            "warm_up": {
                "description": "A number of replications to use as a warm up period",
                "datatype": int,
                "default": 200
            },
            "steady_state_initialization":{
                "description": "Whether the model will be initialized with steady state values",
                "datatype": bool,
                "default": True
            }
            
            
        }
        self.check_factor_list = {
            "number_queues": self.check_number_queues,
            "arrival_alphas": self.check_arrival_alphas,
            "routing_matrix": self.check_routing_matrix,
            "service_mus": self.check_service_mus,
            "t_end": self.check_t_end,
            "warm_up": self.check_warm_up,
            "steady_state_initialization": self.check_steady_state_initialization
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)
        self.n_rngs = 3 * (self.factors["number_queues"] + 1)

    

    def check_number_queues(self):
        return self.factors["number_queues"]>=0
    def check_arrival_alphas(self):
        return all(x >= 0 for x in self.factors["arrival_alphas"])
    def check_service_mus(self):
        return all(x >= 0 for x in self.factors["service_mus"])
    def check_routing_matrix(self):
        transition_sums = list(map(sum, self.factors["routing_matrix"]))
        if all([len(row) == len(self.factors["routing_matrix"]) for row in self.factors["routing_matrix"]]) & \
                all(transition_sums[i] <= 1 for i in range(self.factors["number_queues"])):
            return True
        else:
            return False
    def check_t_end(self):
        return self.factors["t_end"] >= 0
    def check_warm_up(self):
        # Assume f(x) can be evaluated at any x in R^d.
        return self.factors["warm_up"] >= 0
    def check_steady_state_initialization(self):
        return isinstance(self.factors["steady_state_initialization"], bool)

    # function that calulates the lambdas
    def calc_lambdas(self):
        routing_matrix = np.asarray(self.factors["routing_matrix"])
        lambdas = np.linalg.inv(np.identity(self.factors['number_queues']) - routing_matrix.T) @ self.factors["arrival_alphas"]
        return lambdas
    
    def check_simulatable_factors(self):
        lambdas = self.calc_lambdas()
        return all(self.factors['service_mus'][i] > lambdas[i] for i in range(self.factors['number_queues']))
    
   

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "average_queue_length": The time-average of queue length at each station
            "expected_queue_length": The expected queue length calculated using stationary distribution
        """
        # Designate random number generators.
        arrival_rng = [rng_list[i] for i in range(self.factors["number_queues"])]
        transition_rng = [rng_list[i + self.factors["number_queues"]] for i in range(self.factors["number_queues"])]
        time_rng = [rng_list[i + 2*self.factors["number_queues"]] for i in range(self.factors["number_queues"])]
        initialization_rng = rng_list[-1]

        variance = [[] for _ in range(self.factors["number_queues"])]
        #print("mu: ", self.factors['service_mus'])
        def geometric(p):
            return math.floor(np.log(1 - initialization_rng.uniform(0,1)) / math.log(p))
        #calculate the steady state of the queues to check the simulation
        #calculate lambdas
        routing_matrix = np.asarray(self.factors["routing_matrix"])
        lambdas = np.linalg.inv(np.identity(self.factors['number_queues']) - routing_matrix.T) @ self.factors["arrival_alphas"]
        rho = lambdas/self.factors["service_mus"]
        if((rho > 1).any()):
            print("Unstable System: arrival > service rate")
            sys.exit(1)
        #calculate expected value of queue length as rho/(1-rho)
        expected_queue_length = (rho)/(1-rho)
        if self.factors["steady_state_initialization"]:
        # sample initialized queue lengths
            
            queues = [geometric(rho[i]) for i in range(self.factors["number_queues"])]
            completion_times = [math.inf for _ in range(self.factors["number_queues"])]
            # Generate all interarrival, network routes, and service times before the simulation run.
            next_arrivals = [arrival_rng[i].expovariate(self.factors["arrival_alphas"][i]) for i in range(self.factors["number_queues"])]
            for i in range(self.factors["number_queues"]):
                if queues[i] > 0:
                    completion_times[i] = time_rng[i].expovariate(self.factors["service_mus"][i])
            time_sum_queue_length = [0 for _ in range(self.factors["number_queues"])]
        else:
            queues = [0 for _ in range(self.factors["number_queues"])]
            # Generate all interarrival, network routes, and service times before the simulation run.
            next_arrivals = [arrival_rng[i].expovariate(self.factors["arrival_alphas"][i])
                            for i in range(self.factors["number_queues"])]

            # create list of each station's next completion time and initialize to infinity.
            completion_times = [math.inf for _ in range(self.factors["number_queues"])]

            # initialize list of each station's average queue length
            time_sum_queue_length = [0 for _ in range(self.factors["number_queues"])]

        # Initiate clock variables for statistics tracking and event handling.
        clock = 0
        previous_clock = 0
        #print("q: ", queues)
        if not self.factors["steady_state_initialization"]:

            while clock < self.factors['warm_up']:
                next_arrival = min(next_arrivals)
                next_completion = min(completion_times)
                clock = min(next_arrival, next_completion)
                if next_arrival < next_completion: # next event is an arrival
                    station = next_arrivals.index(next_arrival)
                    queues[station] += 1
                    next_arrivals[station] += arrival_rng[station].expovariate(self.factors["arrival_alphas"][station])
                    if queues[station] == 1:
                        completion_times[station] = clock + time_rng[station].expovariate(self.factors["service_mus"][station])
                else: # next event is a departure
                    station = completion_times.index(next_completion)
                    queues[station] -= 1
                    if queues[station] > 0:
                        completion_times[station] = clock + time_rng[station].expovariate(self.factors["service_mus"][station])
                    else:
                        completion_times[station] = math.inf
                    
                    # schedule where the customer will go next
                    prob = transition_rng[station].random()
                    
                    if prob < np.cumsum(self.factors['routing_matrix'][station])[-1]: # customer stay in system
                        next_station = np.argmax(np.cumsum(self.factors['routing_matrix'][station]) > prob)
                        queues[next_station] += 1
                        if queues[next_station] == 1:
                            completion_times[next_station] = clock + time_rng[next_station].expovariate(self.factors["service_mus"][next_station])
            next_arrivals = [next_arrivals[i] - clock for i in range(self.factors["number_queues"])]
            completion_times = [completion_times[i] - clock for i in range(self.factors["number_queues"])]
            clock = 0
            previous_clock = 0

        # Run simulation over time horizon.
        while clock < self.factors["t_end"]:

            next_arrival = min(next_arrivals)
            next_completion = min(completion_times)

            # updated response
            clock = min(next_arrival, next_completion)
            for i in range(self.factors['number_queues']):
                time_sum_queue_length[i] += queues[i] * (clock - previous_clock)
                variance[i].append(queues[i])
            

            previous_clock = clock

            if next_arrival < next_completion: # next event is an arrival
                station = next_arrivals.index(next_arrival)
                queues[station] += 1
                next_arrivals[station] += arrival_rng[station].expovariate(self.factors["arrival_alphas"][station])
                if queues[station] == 1:
                    completion_times[station] = clock + time_rng[station].expovariate(self.factors["service_mus"][station])
            else: # next event is a departure
                station = completion_times.index(next_completion)
                queues[station] -= 1
                if queues[station] > 0:
                    completion_times[station] = clock + time_rng[station].expovariate(self.factors["service_mus"][station])
                else:
                    completion_times[station] = math.inf
                
                # schedule where the customer will go next
                prob = transition_rng[station].uniform(0,1)
                
                if prob < np.cumsum(self.factors['routing_matrix'][station])[-1]: # customer stay in system
                    next_station = np.argmax(np.cumsum(self.factors['routing_matrix'][station]) > prob)
                    queues[next_station] += 1
                    if queues[next_station] == 1:
                        completion_times[next_station] = clock + time_rng[next_station].expovariate(self.factors["service_mus"][next_station])

                
        # end of simulation
        # calculate average queue length
        average_queue_length = [time_sum_queue_length[i]/clock for i in range(self.factors["number_queues"])]
        gradient = [-lambdas[i]/((self.factors["service_mus"][i] - lambdas[i])**2) for i in range(self.factors['number_queues'])]

        responses = {"average_queue_length": average_queue_length, "expected_queue_length" :expected_queue_length, "total_jobs": sum(average_queue_length)}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        gradients['total_jobs']['service_mus'] = tuple(gradient)
        
        return responses, gradients


"""
Summary(.)
-------
Minimize the expected total number of jobs in the system at a time
"""

class OpenJacksonMinQueue(Problem):
    """
    Class to Open Jackson simulation-optimization problems.

    Attributes
    ----------
    name : string
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    service_rates_budget: int
        budget for total service rates sum
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : tuple
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : Model object
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : tuple
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="OPENJACKSON-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.model_default_factors = {}
        self.model_decision_factors = {"service_mus"}
        self.factors = fixed_factors
        

        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (9,9,9,9,9)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 100
            },
            "service_rates_budget" :{
                "description": "budget for total service rates sum",
                "datatype": int,
                "default": 66 # ask later: access model factors when setting default values for budget
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "service_rates_budget": self.check_service_rates_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        self.model = OpenJackson(self.model_fixed_factors)
        self.Ci = np.array([1 for _ in range(self.model.factors["number_queues"])])
        self.di = np.array([self.factors['service_rates_budget']])
        self.Ce = None
        self.de = None
        self.dim = self.model.factors["number_queues"]
        self.lower_bounds = tuple(0 for _ in range(self.model.factors["number_queues"]))
        self.upper_bounds = tuple(self.factors['service_rates_budget'] for _ in range(self.model.factors["number_queues"]))
        # Instantiate model with fixed factors and overwritten defaults.
        self.optimal_value = None  # Change if f is changed.
        self.optimal_solution = None  # Change if f is changed.

    def check_service_rates_budget(self):
        routing_matrix = np.asarray(self.model.factors["routing_matrix"])
        lambdas = np.linalg.inv(np.identity(self.model.factors['number_queues']) - routing_matrix.T) @ self.model.factors["arrival_alphas"]
        if sum(self.factors["service_rates_budget"]) < sum(lambdas) :
            return False
        return True
            

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
            "service_mus": vector[:]
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
        vector = (factor_dict["service_mus"],)
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
        if type(response_dict['total_jobs']) == tuple:
            objectives = (response_dict['total_jobs'][0],)
        else:
            objectives = (response_dict['total_jobs'],)
            print("service mu: ", self.model.factors["service_mus"])
            print("obj: ", objectives)
            
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
        routing_matrix = np.asarray(self.model.factors["routing_matrix"])
        lambdas = np.linalg.inv(np.identity(self.model.factors['number_queues']) - routing_matrix.T) @ self.model.factors["arrival_alphas"]
        det_objectives_gradients = (-np.array(lambdas) / ((np.array(x) - np.array(lambdas))**2),)
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints
        for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic
        constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        # Superclass method will check box constraints.
        # Can add other constraints here.
        routing_matrix = np.asarray(self.model.factors["routing_matrix"])
        lambdas = np.linalg.inv(np.identity(self.model.factors['number_queues']) - routing_matrix.T) @ self.model.factors["arrival_alphas"]
        box_feasible = all(x[i] > lambdas[i] for i in range(self.model.factors['number_queues']))
        upper_feasible = (sum(x) <= self.factors['service_rates_budget'])
        return super().check_deterministic_constraints(x) * box_feasible * upper_feasible

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : vector of decision variables
        """
        if (self.model.factors["steady_state_initialization"]==True):
            x = np.zeros(self.model.factors["number_queues"])
            lambdas = self.model.calc_lambdas()
            sum_alphas = sum(self.model.factors["arrival_alphas"])
            for i in range(self.model.factors["number_queues"]):
                x[i] = lambdas[i] + rand_sol_rng.uniform(0,1) * sum_alphas
        else:
            x = rand_sol_rng.continuous_random_vector_from_simplex(n_elements=self.model.factors["number_queues"],
                                                               summation=self.factors["service_rates_budget"],
                                                               exact_sum=False
                                                               )
        return x
