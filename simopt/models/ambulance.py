
"""
Summary
-------
Simulate multiple periods worth of sales for a (s,S) inventory problem
with continuous inventory. A detailed description of the problem can be found `here <https://simopt.readthedocs.io/en/latest/sscont.html>`_.
"""
import autograd.numpy as np
from base import Auto_Model, Problem
from auto_diff_util import bi_dict, replicate_wrapper, factor_dict, resp_dict_to_array





class Ambulance(Auto_Model):
    """
    
    Desc...
    
    Attributes
    ----------
    name : str
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

    Parameters
    ----------
    fixed_factors : dict
        fixed_factors of the simulation model

        ``demand_mean``
            Mean of exponentially distributed demand in each period (`flt`)
        ``lead_mean``
            Mean of Poisson distributed order lead time (`flt`)
        ``backorder_cost``
            Cost per unit of demand not met with in-stock inventory (`flt`)
        ``holding_cost``
            Holding cost per unit per period (`flt`)
        ``fixed_cost``
            Order fixed cost (`flt`)
        ``variable_cost``
            Order variable cost per unit (`flt`)
        ``s``
            Inventory position threshold for placing order (`flt`)
        ``S``
            Max inventory position (`flt`)
        ``n_days``
            Number of periods to simulate (`int`)
        ``warmup``
            Number of periods as warmup before collecting statistics (`int`)
    See also
    --------
    base.Model
    """
    
    
    
    
    
    def __init__(self, fixed_factors={}):
        self.name = "AMBULANCE"
        self.n_rngs = 4
        self.n_responses = 1
        self.factors = fixed_factors
        self.response_names = ['avg_time_in_system']
        self.specifications = {
            "interarrival_mean": {
                "description": "Mean of exponentially distributed call interarrival times.",
                "datatype": float,
                "default": 2.5
            },
            "scene_mean": {
                "description": "Mean of exponentially distributed time ambulance spends on scene.",
                "datatype": float,
                "default": 10.0
            },
            "sim_length": {
                "description": "Number of time units to simulate.",
                "datatype": int,
                "default": 1000
            },
            "warmup": {
                "description": "Number of time units as warmup before collecting statistics.",
                "datatype": int,
                "default": 20
            },
            "amb_speed": {
                "description": "Constant speed at which ambulances travel.",
                "datatype": float,
                "default": 1.0
            },
            "num_bases": {
                "description": "Number of ambulance base locations or equivalently number of ambulances.",
                "datatype": int,
                "default": 2
            },
            "x0": {
                "description": "x coordinate of ambulance base 0",
                "datatype": float,
                "default": 0
            },
            "y0": {
                "description": "y coordinate of ambulance base 0",
                "datatype": float,
                "default": 0
            },
            "x1": {
                "description": "x coordinate of ambulance base 1",
                "datatype": float,
                "default": 0
            },
            "y1": {
                "description": "y coordinate of ambulance base 1",
                "datatype": float,
                "default": 0
            },
        }
        '''
        self.check_factor_list = {
            "demand_mean": self.check_demand_mean,
            "lead_mean": self.check_lead_mean,
            "backorder_cost": self.check_backorder_cost,
            "holding_cost": self.check_holding_cost,
            "fixed_cost": self.check_fixed_cost,
            "variable_cost": self.check_variable_cost,
            "s": self.check_s,
            "S": self.check_S,
            "n_days": self.check_n_days,
            "warmup": self.check_warmup
        }
        '''
        # Set factors of the simulation model.
        super().__init__(fixed_factors)
        self.SQUARE_WIDTH = 20.0   # kilometers
        self.MAX_EVENTS = 5000     # max events the eventlist can hold
        self.EVENT_SIZE = 5         # Number of values in an event
        # CODES FOR EVENTS
        self.END_EVENT = 0
        self.ARRIVAL_EVENT = 1
        self.SERVICE_EVENT = 2
        self.AVAILABLE = 0
        self.BUSY = 1
        
    '''
    # Check for simulatable factors
    def check_demand_mean(self):
        return self.factors["demand_mean"] > 0

    def check_lead_mean(self):
        return self.factors["lead_mean"] > 0

    def check_backorder_cost(self):
        return self.factors["backorder_cost"] > 0

    def check_holding_cost(self):
        return self.factors["holding_cost"] > 0

    def check_fixed_cost(self):
        return self.factors["fixed_cost"] > 0

    def check_variable_cost(self):
        return self.factors["variable_cost"] > 0

    def check_s(self):
        return self.factors["s"] > 0

    def check_S(self):
        return self.factors["S"] > 0

    def check_n_days(self):
        return self.factors["n_days"] >= 1

    def check_warmup(self):
        return self.factors["warmup"] >= 0

    def check_simulatable_factors(self):
        return self.factors["s"] < self.factors["S"]
        
    '''
    
    
    def next_arrival(self, t, rng_gens, MEAN_INTERARRIVAL, MEAN_SCENE_TIME):
    # Event is 'arrival', time of arrival, x, y, scene_time
        the_event = [t +  -np.log(1-rng_gens[0].random())*MEAN_INTERARRIVAL,
                 self.ARRIVAL_EVENT,
                 rng_gens[1].random()*self.SQUARE_WIDTH,
                 rng_gens[2].random()*self.SQUARE_WIDTH,
                 -np.log(1-rng_gens[3].random())*MEAN_SCENE_TIME]
        return the_event


    # Event list is simply an unordered array

    # Get next event, delete it from event_list and decrease num_events by 1
    def get_next_event(self, event_list, num_events, SIM_LENGTH):
        mini = SIM_LENGTH + 2.0
        for i in range(num_events):
            if event_list[i][0] < mini:
                mini = event_list[i][0]
                next_event_index = i
                #print(next_event_index)
    #    next_event_index = np.argmin(event_list[:,0])
    #    print('event index %d' %next_event_index)
        next_event = event_list[next_event_index]
        event_list[next_event_index:num_events+1] = event_list[next_event_index+1:num_events+2]
        num_events -= 1
    #    print(event_list)
        return event_list, num_events, next_event

    def add_event(self, the_event, event_list, num_events):
        if num_events == self.MAX_EVENTS:
            print('Max events exceeded')
        else:
            event_list[num_events] = the_event
            num_events += 1
        return event_list, num_events

    def inner_replicate(self, diff_factors, rng_list, response_names):
        SQUARE_WIDTH = self.SQUARE_WIDTH
        MAX_EVENTS = self.MAX_EVENTS
        EVENT_SIZE = self.EVENT_SIZE
        END_EVENT = self.END_EVENT
        ARRIVAL_EVENT = self.ARRIVAL_EVENT
        SERVICE_EVENT = self.SERVICE_EVENT
        AVAILABLE = self.AVAILABLE
        BUSY = self.BUSY
        
        
        PRINT_DETAILS = False
        rng_gens = rng_list
        
        
        
        factors = factor_dict(self, diff_factors)
        SIM_LENGTH = factors['sim_length']
        NUM_BASES = factors['num_bases']
        AMB_SPEED = factors['amb_speed']
        x = [factors['x0'], factors['x1']]
        y = [factors['y0'], factors['y1']]
        
        
        '''
        print(SIM_LENGTH)
        print(EVENT_SIZE)
        print(MAX_EVENTS)
        '''
        
        event_list = [[SIM_LENGTH+1]*EVENT_SIZE for _ in range(MAX_EVENTS)] # Initialize with what is essentially infinity        
        event_list[0][0] = SIM_LENGTH # End simulation at this time
        event_list[0][1] = END_EVENT  # End event posted
        num_events = 1
        queued_calls = [[0]*EVENT_SIZE for _ in range(MAX_EVENTS)]
        
        
        
        arrival_event = np.zeros(EVENT_SIZE)
        arrival_event = self.next_arrival(0, rng_gens, factors['interarrival_mean'],factors['scene_mean'])
        event_list, num_events = self.add_event(arrival_event, event_list, num_events)

        ambs = [[x[i],y[i],AVAILABLE] for i in range(NUM_BASES)] # For each ambulance, store its base location and its status
        
        
        active_calls = 0

        event_list, num_events, next_event = self.get_next_event(event_list, num_events, factors['sim_length'])

        sum_responses = 0
        num_calls = 0
        while next_event[1] != END_EVENT:
        #     print('Next event')
        #     print(next_event)
            #print('active calls = ' + str(active_calls))
            #print("delta", grad_response_delta)
            #print("carried", carried_grad)
            #print("grad",grad_response)
            if next_event[1] == ARRIVAL_EVENT:
                if PRINT_DETAILS:
                    print('arrival at time %.1f' %(next_event[0]))
                active_calls += 1
                if active_calls > NUM_BASES: # All full, have to queue
                    queued_calls[active_calls-NUM_BASES-1] = next_event # Add call to the queue
                    if PRINT_DETAILS:
                        print('queued due to lack of ambs')
                else: # Have an ambulance available. Find closest one. All ambs are at base when available
                    vector_differences = np.array(ambs)[:,0:2] - np.array(next_event[2:4]) # differences in location
                    closest = 0
                    closest_time = SQUARE_WIDTH * 5 / AMB_SPEED # Effectively infinity
                    for i in range(NUM_BASES):
                        if ambs[i][2] == AVAILABLE:
                            this_time = np.sum(np.abs(vector_differences[i][:])) / AMB_SPEED # Manhattan distance
        #                    print(this_time)
                            if this_time < closest_time:
                                closest_time = this_time
                                closest = i
                    ambs[closest][2] = BUSY
                    #calls_handled_by[closest] += 1
                    service_event_completion_time = next_event[0] + 2.0 * closest_time + next_event[4]
                    if PRINT_DETAILS:
                        #print(service_event_completion_time._value)
                        print('Dispatched amb %d, completes at time %.0f' %(closest, service_event_completion_time._value))
                    service_event = [service_event_completion_time, SERVICE_EVENT, closest, 0, 0] # Don't need last two entries
                    event_list, num_events = self.add_event(service_event, event_list, num_events)
                    sum_responses += closest_time
                    
                    
                   
                    num_calls += 1
                arrival_event = self.next_arrival(next_event[0], rng_gens, factors['interarrival_mean'],factors['scene_mean']) # Add new arrival to end of event queue
                event_list, num_events = self.add_event(arrival_event, event_list, num_events)
                if PRINT_DETAILS:
                    print('Arrival added at time %.0f' %(arrival_event[0]))
            elif next_event[1] == SERVICE_EVENT:
                free_amb = int(next_event[2])
                if PRINT_DETAILS:
                    print('Service completion by amb %d at time %.0f' %(free_amb, next_event[0]._value))
                ambs[free_amb][2] = AVAILABLE
                active_calls -= 1
                
                if active_calls >= NUM_BASES: # Queued calls, so serve the first in line 
                    #calls_handled_by[free_amb] += 1
                    queued_call = queued_calls[0]
                    if PRINT_DETAILS:
                        print('Serving call received at time %.0f' %(queued_call[0]))
                    queued_calls[0:active_calls] = queued_calls[1:active_calls+1] # delete the first queued call
                    #this_time = (np.abs(ambs[free_amb][0] - queued_call[2])+ np.abs(ambs[free_amb][1] - queued_call[3]))/ AMB_SPEED # time for this amb to get to call
                    this_time = np.sum(np.abs(np.array(ambs)[free_amb, 0:2] - np.array(queued_call)[2:4])) / AMB_SPEED # time for this amb to get to call
                    service_event_completion_time = next_event[0] + 2 * this_time + queued_call[4]
                    service_event = [service_event_completion_time, SERVICE_EVENT, free_amb, 0, 0]
                    event_list, num_events = self.add_event(service_event, event_list, num_events)
                    sum_responses += this_time + (next_event[0] - queued_call[0]) # BUG HERE I THINK! Should be this_time + time in queue
                    
                        
                    num_calls += 1
        #     print('Completed event processing')
        #     print('Event list')
        #     print(event_list)
            event_list, num_events, next_event = self.get_next_event(event_list, num_events, factors['sim_length'])
        #     print('Event list after retrieving next event')
        #     print(event_list)
        #     events_processed += 1
        if PRINT_DETAILS:
           
          
            print('Simulation complete')
            print('number of calls %d' %num_calls)
            print('Average response time')
            print(sum_responses / num_calls)
            print('Estimated gradient of average response time')
        
            #print('active calls = ' + str(active_calls))
            print("calls handled by ", calls_handled_by)
        #f_values[replication] = sum_responses #/ num_calls
        #grad_x_values[replication] = grad_response[0] #/ num_calls
        #grad_y_values[replication] = grad_response[1] #/ num_calls
        
    # end for replication
    #return f_values, grad_x_values, grad_y_values, calls_handled_by
    #print(grad_response[0], grad_response[1])
            
            
        responses = {'avg_time_in_system': sum_responses/num_calls}

        return resp_dict_to_array(self, responses, response_names)

      

"""
Summary
-------
Minimize the expected total cost for (s, S) inventory system.
"""


class AmbulanceMinTIS(Problem):
    """
    Class to make (s,S) inventory simulation-optimization problems.

    Attributes
    ----------
    name : str
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : str
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : str
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : float
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : base.Model
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
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
        user-specified name of problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="AMBULANCE-1", fixed_factors={}, model_fixed_factors={}):
        self.SQUARE_WIDTH = 20.0
        self.name = name
        self.dim = 4
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.lower_bounds = (0, 0, 0, 0, )
        self.upper_bounds = (self.SQUARE_WIDTH, self.SQUARE_WIDTH, self.SQUARE_WIDTH, self.SQUARE_WIDTH,)
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {
            "interarrival_mean": 2.5,
            "scene_mean": 10.0,
            "sim_length": 1000,
            "warmup": 20,
            "amb_speed": 1.0,
            "num_bases": 2
        }
        self.model_decision_factors = {"x0", "y0", 'x1', 'y1'}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (20.0, 20.0, 0.0, 0.0)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 1000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = Ambulance(self.model_fixed_factors)

    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dict
            dictionary with factor keys and associated values
        """
        factor_dict = {
            "x0": vector[0],
            "y0": vector[1],
            "x1": vector[2],
            "y1": vector[3]
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dict
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = (factor_dict["x0"], factor_dict["y0"], factor_dict["x1"], factor_dict["y1"],)
        return vector

    def response_dict_to_objectives(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dict
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (response_dict['avg_time_in_system'],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

        Arguments
        ---------
        response_dict : dict
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
        det_objectives_gradients = ((0,),(0,),(0,),(0,),)
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
        #return (x[0] >= 0 and x[1] >= 0)
        return (np.all(x>=0) and np.all(x <= self.SQUARE_WIDTH))

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.mrg32k3a.MRG32k3a
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = (rand_sol_rng.random()*self.SQUARE_WIDTH, rand_sol_rng.random()*self.SQUARE_WIDTH, 
             rand_sol_rng.random()*self.SQUARE_WIDTH, rand_sol_rng.random()*self.SQUARE_WIDTH)
        return x
