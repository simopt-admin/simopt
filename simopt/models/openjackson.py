"""
Summary
-------
Simulate an open jackson network 
"""
import autograd.numpy as np
import math as math
from collections import deque

from ..base import Model, Problem

# generates an erdos renyi graph where each subgraph has an exit 
def erdos_renyi(rng, n, p, directed = True):
    graph = np.zeros((n,n+1)) 
    for i in range(n):
        for j in range(n+1):
            prob = rng.uniform(0,1)
            if prob < p:
                graph[i][j] = 1
    if not directed:
        graph = np.triu(graph)

    #check for exits in each subgraph if there are not valid exits 
    # then create a new erdos_renyi graph until one is valid
    has_exit = set()
    checked = False
    while(not checked):
        numexitable = len(has_exit)
        for i in range(n):
            if (graph[i][-1]) == 1: 
                has_exit.add(i)
                # print("add original", has_exit)
            if len(has_exit) > 0:
                has_exit2 = []
                for j in has_exit:
                    if graph[i][j] == 1 :
                        has_exit2 += [i]
                for a in has_exit2:
                    has_exit.add(a)
                    # print("add adjacent", has_exit)
        afternumexitable = len(has_exit)
        checked = (afternumexitable == n or numexitable == afternumexitable)
    # if the graph has nodes that have no path out then add a path out to those nodes
    if len(has_exit) != n:
        for x in set(range(n)).difference(has_exit):
            graph[x][-1] = 1 

    return graph


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
    def __init__(self, fixed_factors=None, random = False):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "OPENJACKSON"
        self.n_responses = 2
        self.random = random
        self.n_random = 2  # Number of rng used for the random instance
        # random instance factors: number_queues, arrival_alphas, service_mus, routing_matrix

        self.factors = fixed_factors
        self.specifications = {
            "number_queues": {
                "description": "The number of queues in the network",
                "datatype": int,
                "default": 5
            },
            "arrival_alphas": {
                "description": "The arrival rates to each queue from outside the network",
                "datatype": tuple,
                "default": (2,3,2,4,3)
            },
            "service_mus": {
                "description": "The mu values for the exponential service times ",
                "datatype": tuple,
                "default": (11,11,11,11,11)
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
                "default": 200
            },
            "warm_up": {
                "description": "A number of replications to use as a warm up period",
                "datatype": int,
                "default": 0
            },
            "steady_state_initialization":{
                "description": "Whether the model will be initialized with steady state values",
                "datatype": bool,
                "default": False
            },
            "density_p":{
                "description": "The probability of an edge existing in the graph in the random instance",
                "datatype": float,
                "default": 0.5
            },
            "random_arrival_parameter":{
                "description": "The parameter for the random arrival rate exponential distribution when creating a random instance",
                "datatype": float,
                "default": 1
            }
            
            
        }
        self.check_factor_list = {
            "number_queues": self.check_number_queues,
            "arrival_alphas": self.check_arrival_alphas,
            "routing_matrix": self.check_routing_matrix,
            "service_mus": self.check_service_mus,
            "t_end": self.check_t_end,
            "warm_up": self.check_warm_up,
            "steady_state_initialization": self.check_steady_state_initialization,
            "density_p": self.check_density_p,
            "random_arrival_parameter": self.check_random_arrival_parameter
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)
        self.n_rngs = 3 * (self.factors["number_queues"] + 1)

    

    def check_number_queues(self):
        return self.factors["number_queues"]>=0
    def check_arrival_alphas(self):
        return all(x >= 0 for x in self.factors["arrival_alphas"])
    def check_service_mus(self):
        lambdas = self.calc_lambdas()
        return all(x >= 0 for x in self.factors["service_mus"]) and all(self.factors['service_mus'][i] > lambdas[i] for i in range(self.factors["number_queues"]))
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
    def check_density_p(self):
        return 0 <= self.factors["density_p"] <= 1
    def check_random_arrival_parameter(self):
        return self.factors["random_arrival_parameter"] >= 0

    # function that calulates the lambdas
    def calc_lambdas(self):
        routing_matrix = np.asarray(self.factors["routing_matrix"])
        lambdas = np.linalg.inv(np.identity(self.factors['number_queues']) - routing_matrix.T) @ self.factors["arrival_alphas"]
        return lambdas
    
    def check_simulatable_factors(self):
        lambdas = self.calc_lambdas()
        return all(self.factors['service_mus'][i] > lambdas[i] for i in range(self.factors['number_queues']))
    
    def attach_rng(self, random_rng):
        #returns a dirichlet distribution of same shape as alpha
        def dirichlet(alpha, rng):
            gamma_vars = [rng.gammavariate(a, 1) for a in alpha]
            sum_gamma_vars = sum(gamma_vars)
            dirichlet_vars = [x / sum_gamma_vars for x in gamma_vars]
            return dirichlet_vars
        
        self.random_rng = random_rng
        random_num_queue = self.factors['number_queues']
        p = self.factors['density_p']
        random_matrix = erdos_renyi(random_rng[0], random_num_queue,p)
        prob_matrix = np.zeros((random_num_queue, random_num_queue + 1))
        for i in range(random_num_queue):
            a = int(sum(random_matrix[i]))+1
            probs = dirichlet(np.ones(a), rng = random_rng[0])
            r = 0
            for j in range(random_num_queue+1):
                if random_matrix[i][j]==1 or j == random_num_queue:
                    prob_matrix[i][j] = probs[r]
                    r += 1
        prob_matrix = np.asarray(prob_matrix)
        prob_matrix = prob_matrix[:, :-1]
        random_arrival = []
        for i in range(random_num_queue):
            random_arrival.append(random_rng[1].expovariate(self.factors['random_arrival_parameter']))

        self.factors["arrival_alphas"] = random_arrival
        self.factors['routing_matrix'] = prob_matrix.tolist()

        return

    def get_IPA(Dl, V, W, q, k, mu, self):  # D is the dictionary, St L[i][1]: ith arrive cust's 
        def I(x, k):
            if x==k:
                return 1
            else:
                return 0
        IA, IW = [[] for i in range(q)], [[-V[i][0]/mu * I(i, k)] for i in range(q)]
        for i in range(len(Dl)):
            queue = int(Dl[i][0])
            idx = Dl[i][1]
            v = V[queue][idx]
            if idx == 0:
                if Dl[i][2][0] == -1:
                    IA[queue].append(0)
                else:
                    pre_queue = Dl[i][2][0] 
                    pre_idx = Dl[i][2][1]-1
                    print('i: ', i, ', prequeue: ', pre_queue, ', pre_idx: ', pre_idx)
                    # print('iwww', IW[pre_queue], IA[pre_queue])
                    if len(IA[pre_queue]) == 0:   # Warm up bug..
                        print('warmup')
                        a = 0
                    else:
                        a = IW[pre_queue][pre_idx] + IA[pre_queue][pre_idx]
                    IA[queue].append(a)
            else:
                # Calculate IA
                if Dl[i][2][0] == -1:
                    IA[queue].append(0)
                else:
                    pre_queue = Dl[i][2][0] 
                    pre_idx = Dl[i][2][1]-1
                    print(pre_queue, pre_idx, IW[pre_queue], IA[pre_queue])
                    if len(IA[pre_queue]) == 0:   # Warm up bug..
                        print('warmup')
                        a = 0
                    else: 
                        a = IW[pre_queue][pre_idx] + IA[pre_queue][pre_idx]
                    # print('i: ', i, ', prequeue: ', pre_queue, ', pre_idx: ', pre_idx)
                    # print('a', a)
                    IA[queue].append(a)
                if W[queue][idx] <= 0:
                    v = -V[queue][idx]/mu * I(queue, k)
                    IW[queue].append(v)
                else:
                    v = -V[queue][idx]/mu * I(queue, k) + IW[queue][idx-1]
                    # print('pre: ', IA[queue][idx-1])
                    # print('it: ', IA[queue][idx])
                    u = IA[queue][idx-1] - IA[queue][idx]
                    IW[queue].append(u + v)
    
        return IA, IW

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

        def geometric(p):
            return math.floor(np.log(1 - initialization_rng.uniform(0,1)) / math.log(p))
        #calculate the steady state of the queues to check the simulation
        #calculate lambdas
        routing_matrix = np.asarray(self.factors["routing_matrix"])
        lambdas = np.linalg.inv(np.identity(self.factors['number_queues']) - routing_matrix.T) @ self.factors["arrival_alphas"]
        rho = lambdas/self.factors["service_mus"]
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

        # warm-up period
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

        # statistics needed for IPA - waiting_record, service_record, arrival_record, transfer_record, IPA_record
        # waiting_record: records the waiting time of each customer before entering service. record when scheduling new completion times
            # helper list: time_entered. records the time each customer enters the system. record when scheduling new arrival or departing to another station.
                                        # pop when scheduling new completion times
        # service_record: records the service time of each customer. record when scheduling new completion times
        # arrival_record: records the arrival time of each customer. record when scheduling new arrivals
        # transfer_record: records where the customer is transferred from, formatted as [previous station, previous index], if new : [-1]
            # record when scheduling departures & new arrivals
        # IPA_record: records the customer's index in the queue and the station it is transferred from, each element formatted as [station, index, [previous station, previous index]].
            # record at shceduling new completion times
        # collect all statistics starting from warm-up period
        waiting_record = [[] for _ in range(self.factors["number_queues"])]
        time_entered = [deque() for _ in range(self.factors['number_queues'])]
        service_record = [[] for _ in range(self.factors["number_queues"])]
        arrival_record = [[] for _ in range(self.factors["number_queues"])]
        transfer_record = [deque() for _ in range(self.factors["number_queues"])]
        IPA_record = []

        # Run simulation over time horizon.
        while clock < self.factors['t_end']:
            next_arrival = min(next_arrivals)
            next_completion = min(completion_times)
            clock = min(next_arrival, next_completion)
            for i in range(self.factors['number_queues']):
                time_sum_queue_length[i] += queues[i] * (clock - previous_clock)

            previous_clock = clock
            if next_arrival < next_completion: # next event is an arrival
                station = next_arrivals.index(next_arrival)
                queues[station] += 1
                next_arrivals[station] += arrival_rng[station].expovariate(self.factors["arrival_alphas"][station])

                time_entered[station].append(clock)
                if queues[station] == 1:
                    completion_times[station] = clock + time_rng[station].expovariate(self.factors["service_mus"][station])
                    waiting_record[station].append(clock - time_entered[station].popleft())
            else: # next event is a departure
                station = completion_times.index(next_completion)
                queues[station] -= 1
                if queues[station] > 0:
                    completion_times[station] = clock + time_rng[station].expovariate(self.factors["service_mus"][station])
                    waiting_record[station].append(clock - time_entered[station].popleft())
                else:
                    completion_times[station] = math.inf
                # schedule where the customer will go next
                prob = transition_rng[station].random()

                if prob < np.cumsum(self.factors['routing_matrix'][station])[-1]: # customer stay in system
                    next_station = np.argmax(np.cumsum(self.factors['routing_matrix'][station]) > prob)
                    queues[next_station] += 1
                    time_entered[next_station].append(clock)
                    if queues[next_station] == 1:
                        completion_times[next_station] = clock + time_rng[next_station].expovariate(self.factors["service_mus"][next_station])
                        waiting_record[next_station].append(clock - time_entered[next_station].popleft())
        # end of simulation
        # Calculate the IPA gradient
        # IPA_gradient = []
        # for j in range(self.factors['number_queues']):
        #     IPA_gradient.append(self.get_IPA(IPA_record, service_times, waiting_times, self.factors['number_queues'], j, self.factors['service_mus'][j]))

        # calculate average queue length
        average_queue_length = [time_sum_queue_length[i]/clock for i in range(self.factors["number_queues"])]
        gradient = [-lambdas[i]/(self.factors["service_mus"][i] - lambdas[i])**(2) for i in range(self.factors['number_queues'])]
        # lagrange_obj = sum(lambdas[i]/(self.factors["service_mus"][i] - lambdas[i]) for i in range(self.factors['number_queues'])) + 0.5*sum(self.factors['service_mus'])
        lagrange_obj = sum(average_queue_length) + 0.5*sum(self.factors['service_mus'])
        lagrange_grad = [-lambdas[i]/(self.factors["service_mus"][i] - lambdas[i])**(2) + 1 for i in range(self.factors['number_queues'])]

        responses = {"total_jobs": sum(average_queue_length)}
        # responses = {"average_queue_length": average_queue_length, 'lagrange_obj': lagrange_obj, "expected_queue_length" :expected_queue_length,
        #               "total_jobs": sum(average_queue_length)}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}

        # gradients['average_queue_length']['service_mus'] = tuple(gradient)
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
    def __init__(self, name="OPENJACKSON-1", fixed_factors=None, model_fixed_factors=None, random = False, random_rng = None):
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
        self.random = random
        self.n_rngs = 1

        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (11,11,11,11,11)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000
            },
            "service_rates_budget" :{
                "description": "budget for total service rates sum",
                "datatype": int,
                "default": 100 # ask later: access model factors when setting default values for budget
            },
            "gamma_mean":{
                "description": "scale of the mean of gamma distribution when generating service rates upper bound",
                "datatype": float,
                "default": 0.5
            },
            "gamma_scale":{
                "description": "shape of gamma distribution when generating service rates upper bound",
                "datatype": tuple,
                "default": 5
            }

        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "service_rates_budget": self.check_service_rates_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        self.model = OpenJackson(self.model_fixed_factors, random)
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
        if random and random_rng:
            self.model.attach_rng(random_rng)

        # lambdas = self.model.calc_lambdas()
        # r = self.factors["service_rates_budget"]/sum(lambdas)
        # self.factors['initial_solution'] = tuple([r*lambda_i for lambda_i in lambdas])

    def attach_rngs(self, random_rng):
        self.random_rng = random_rng
        lambdas = self.model.calc_lambdas()

        # generate service rates upper bound as the sum of lambdas plus a gamma random variable with parameter as an input
        mean = self.factors["gamma_mean"] * sum(lambdas)
        scale = self.factors["gamma_scale"]
        gamma = random_rng[0].gammavariate(mean/scale, scale)
        self.factors["service_rates_budget"] = sum(lambdas) + gamma

        lambdas = self.model.calc_lambdas()
        r = self.factors["service_rates_budget"]/sum(lambdas)
        self.factors['initial_solution'] = tuple([r*lambda_i for lambda_i in lambdas])
        
        return
    
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
        det_objectives_gradients = ((0,) * self.dim,)
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
            x = [0]*self.model.factors["number_queues"]
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

"""
Summary(.)
-------
Minimize the expected total number of jobs in the system at a time
"""

class OpenJacksonMinQueueLagrange(Problem):
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
    def __init__(self, name="OPENJACKSON-2", fixed_factors=None, model_fixed_factors=None, random = False, random_rng = None):
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
        self.random = random
        self.n_rngs = 1

        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (15,15,15,15,15)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 500
            },
            "service_rates_factor" :{
                "description": "weight of the service rates in the objective function",
                "datatype": int,
                "default": 0.5
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "service_rates_factor": self.check_service_rates_factor
        }
        super().__init__(fixed_factors, model_fixed_factors)
        self.model = OpenJackson(self.model_fixed_factors, random)
        self.dim = self.model.factors["number_queues"]
        lambdas = self.model.calc_lambdas()
        self.lower_bounds = tuple(lambdas)
        self.upper_bounds = (np.inf,) * self.dim
        # Instantiate model with fixed factors and overwritten defaults.
        self.optimal_value = None  # Change if f is changed.
        self.optimal_solution = None  # Change if f is changed.
        if random and random_rng:
            self.model.attach_rng(random_rng)

        self.factors['initial_solution'] = tuple([1.1*lambda_i for lambda_i in lambdas])

    def attach_rngs(self, random_rng):
        self.random_rng = random_rng
        self.factors["service_rates_factor"] = random_rng[0].uniform(0,5)
        
        return

    def check_service_rates_factor(self):
       
        return self.factors['service_rates_factor'] >= 0

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
        if type(response_dict['lagrange_obj']) == tuple:
            objectives = (response_dict['lagrange_obj'][0],)
        else:
            objectives = (response_dict['lagrange_obj'],)
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
        det_objectives_gradients = ((0,) * self.dim,)
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
        det_stoch_constraints = tuple([0]*self.dim)
        det_stoch_constraints_gradients = (0,)
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
        return super().check_deterministic_constraints(x) * box_feasible

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
            x = [0]*self.model.factors["number_queues"]
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