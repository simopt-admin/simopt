"""
Summary
-------
Simulate demand at facilities.
"""
import numpy as np

from base import Model, Problem


class Voting(Model):
    """
    A model that simulates a voting problem with 
    gamma distributions, 
    Returns ...

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
        details of each factor (for GUI and data validation)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : nested dict
        fixed factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors={}):
        self.name = "Voting"
        self.n_rngs = 4
        self.n_responses = 1
        self.specifications = {
            "mach_allocation": {
                "description": "number of machines allocation for precinct i",
                "datatype": list,
                "default": [10,10,10,10,10]
            },
            "n_mach": {
                "description": "max number of machines available",
                "datatype": int,
                "default": 50
            },
            "mid_turn_per": {
                "description": "midpoint turnout percentage for precinct i",
                "datatype": list,
                "default": [.4,.2,.6,.3,.1]
            },
            "turn_ran": {
                "description": "turnout range specific to precinct i",
                "datatype": list,
                "default": [10,15,10,50,30]
            },
            "reg_vote": {
                "description": "number of registered voters in precinct i",
                "datatype": list,
                "default": [100,200,100,400,200]
            },
            "mean_time2vote": {
                "description": "The mean time for the gamma distributed time taken to vote",
                "datatype": int,
                "default": 7.5
            },
            "stdev_time2vote": {
                "description": "The standard deviation for the gamma distributed time to vote",
                "datatype": int,
                "default": 2
            },
            "mean_repair": {
                "description": "Voting machines are repaired according to a gamma distribution, this is the mean time, minutes",
                "datatype": int,
                "default": 60
            },
            "stdev_repair": {
                "description": "standard deviation for gamma distribution for time to repair a machine, minutes",
                "datatype": int,
                "default": 20
            },
            "bd_prob": {
                "description": "Probability at which the voting machines break down (bd)",
                "datatype": float,
                "default": .05
            },
            "hours": {
                "description": "number of hours open to vote",
                "datatype": float,
                "default": 13.0
            },
            "n_prec":{
                "description": "Number of precincts",
                "datatype": int,
                "default": 5
            }
        }
        self.check_factor_list = {
            "mach_allocation_amt": self.check_mach_allocation_amt,
            "mach_allocation_len": self.check_mach_allocation_len,
            "mid_turn_perc": self.check_mid_turn_perc,
            "mid_turn_perc_len": self.check_mid_turn_perc_len,
            "turn_ran_len": self.check_turn_ran_len,
            "reg_vote_len": self.check_reg_vote_len,
            "bd_prob_perc": self.check_bd_prob_perc
        }
        # Set factors of the simulation model.
                                                                                # ASK ABOUT CHECK AND NON NEGATIVITY CONTRAINTS

        super().__init__(fixed_factors)

    def check_mach_allocation_amt(self): #Making sure that all machines are allocated and equal to max available
        return sum(self.factors["mach_allocation"]) == self.factors["n_mach"] 

    def check_mach_allocation_len(self): #verifying that the length of the list matches number of precincts
        return len(self.factors["mach_allocation"]) == self.factors["n_prec"]

    def check_mid_turn_perc(self): #veifying that all are percentages
        for i in self.factors["mid_turn_perc"]:
            if i < 0 and i > 1:
                return False
        return True
        
    def check_mid_turn_perc_len(self): #verifying that the length of the list matches number of precincts
        #FILL CODE HERE
        return

    def check_turn_ran_len(self): #verifying that the length of the list matches number of precincts
        #FILL CODE HERE
        return

    def check_reg_vote_len(self): #verifying that the length of the list matches number of precincts
        #FILL CODE HERE
        return
 
    def check_bd_prob_perc(self): #veifying that all are percentages
        for i in self.factors["bd_prob_perc"]:
            if i < 0 and i > 1:
                return False
        return True    
        
    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict          
            <NEW>
            "turnout_param" = the factor that go to vote in a precinct versus voting population in that precinct, triangularly distributed
            "vote_time" = time that it takes for each voter to cast a ballot, gamma distributed
                                        ### DO WE NEED TO CALCULATE THIS FOR EACH VOTER OR JUST FOR THE MACHINE OR THE PRECINCT
            
            ##VERY CONFUSED DO WE DECIDE IF EACH IS BROKEN OR NOT OR DO PER PRECINCT, Etc.
            "mach_bd" = binary variable, probability that the           
                0 : The voting machine is broken down at start of day
                1 : The voting machine does not break down for the day 

            "repair_time" = the time that it will take for a machine to be repaired, gamma distributed
            "arrival_rate" = rate of arrival to the voting location

        gradients : dict of dicts
            gradient estimates for each response
        """
        #                      self.factors["mid_turn_per"]
        
        for m in range(len(self.factors["mach_allocation"])):          #p is num of machines in that precinct
            mach_delay = []
            for i in range(self.factors["mach_allocation"][i]):        #i is each individual machine in that precinct 
                n = 1
                p = self.factors["bd_prob"]         #Default is .05
                if np.random.binomial(n,p) == 1:    #Determining if the machine will be borken down to start day
                    t = random.gammavariate((self.factors["mean_repair"]^2)/(self.factors["stdev_repair"]^2),(self.factors["stdev_repair"]^2)/(self.factors["mean_repair"])) #Determines wait time for broken machine in minutes
                else:
                    t = 0
                mach_delay.append(t)
                    #ti = ai + bi*T
           
            t_i = self.factors["mid_turn_per"] + self.factors["turn_ran"] * random.triangular(-1,1,0)

            

            p_lamda = (self.factors["reg_vote"] * t_i) / self.factors["hours"]
            

            arr_times = []
            t = expovariate(p_lamda)              #initial arrival
            while t <= self.factors["hours"]*60:      
                arr_times.append(t)                 #appends before so that the last arrival in list will be before voting closes
                t = expovariate(p_lamda) + t

            voting_times = []
            for i in range(len(arr_times)):
                voting_times.append(random.gammavariate((self.factors["mean_time2vote"]^2)/(self.factors["stdev_time2vote"]^2),(self.factors["stdev_time2vote"]^2)/(self.factors["mean_time2vote"])))
            

            #Starting machine availablility, numbers represent at what time the machine BECOMES available
            available = []        
            for i in range(self.factors["mach_allocation"][i]):
                if mach_delay[i] == 0:
                    available.append(0)     #1 means it is available
                else:
                    available.append(mach_delay[i])     #time that machine becomes available

            t = 0   
            votes = 
            wait_time = []      #going to collect all wait times of voters
            while votes <= len(arr_times): 
                #start iterating by times, break for the smallest next time

            


        # Compose responses and gradients.
        responses = {'stockout_flag': stockout_flag,
                     'n_fac_stockout': n_fac_stockout,
                     'n_cut': n_cut}
        return responses

"""
Summary
-------
Minimize the (deterministic) total cost of installing capacity at
facilities subject to a chance constraint on stockout probability.
"""


class FacilitySizingTotalCost(Problem):
    """
    Base class to implement simulation-optimization problems.

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
    optimal_value : float
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
    rng_list : list of rng.MRG32k3a objects
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
    def __init__(self, name="FACSIZE-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.dim = 3
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.minmax = (-1,)
        self.constraint_type = "stochastic"
        self.variable_type = "continuous"
        self.lower_bounds = (0, 0, 0)
        self.upper_bounds = (np.inf, np.inf, np.inf)
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None  # (185, 185, 185)
        self.model_default_factors = {}
        self.model_decision_factors = {"capacity"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (300, 300, 300)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000
            },
            "installation_costs": {
                "description": "Cost to install a unit of capacity at each facility.",
                "datatype": tuple,
                "default": (1, 1, 1)
            },
            "epsilon": {
                "description": "Maximum allowed probability of stocking out.",
                "datatype": float,
                "default": 0.05
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "installation_costs": self.check_installation_costs,
            "epsilon": self.check_epsilon
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = FacilitySize(self.model_fixed_factors)

    def check_installation_costs(self):
        if len(self.factors["installation_costs"]) != self.model.factors["n_fac"]:
            return False
        elif any([elem < 0 for elem in self.factors["installation_costs"]]):
            return False
        else:
            return True

    def check_epsilon(self):
        return 0 <= self.factors["epsilon"] <= 1

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
            "capacity": vector[:]
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
        vector = tuple(factor_dict["capacity"])
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
        objectives = (0,)
        return objectives

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
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = (-response_dict["stockout_flag"],)
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
        det_stoch_constraints = (self.factors["epsilon"],)
        det_stoch_constraints_gradients = ((0,),)
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
        det_objectives = (np.dot(self.factors["installation_costs"], x),)
        det_objectives_gradients = ((self.factors["installation_costs"],),)
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
        return np.all(x > 0)

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = tuple([300*rand_sol_rng.random() for _ in range(self.dim)])
        return x


"""
Summary
-------
Maximize the probability of not stocking out subject to a budget
constraint on the total cost of installing capacity.
"""


class FacilitySizingMaxService(Problem):
    """
    Base class to implement simulation-optimization problems.

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
    optimal_value : float
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
    rng_list : list of rng.MRG32k3a objects
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
    def __init__(self, name="FACSIZE-2", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.dim = 3
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.lower_bounds = (0, 0, 0)
        self.upper_bounds = (np.inf, np.inf, np.inf)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None  # (175, 179, 143)
        self.model_default_factors = {}
        self.model_decision_factors = {"capacity"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (100, 100, 100)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000
            },
            "installation_costs": {
                "description": "Cost to install a unit of capacity at each facility.",
                "datatype": tuple,
                "default": (1, 1, 1)
            },
            "installation_budget": {
                "description": "Total budget for installation costs.",
                "datatype": float,
                "default": 500.0
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "installation_costs": self.check_installation_costs,
            "installation_budget": self.check_installation_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = FacilitySize(self.model_fixed_factors)

    def check_installation_costs(self):
        if len(self.factors["installation_costs"]) != self.model.factors["n_fac"]:
            return False
        elif any([elem < 0 for elem in self.factors["installation_costs"]]):
            return False
        else:
            return True

    def check_installation_budget(self):
        return self.factors["installation_budget"] > 0

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
            "capacity": vector[:]
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
        vector = tuple(factor_dict["capacity"])
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
        objectives = (1 - response_dict["stockout_flag"],)
        return objectives

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
        det_objectives_gradients = ((0, 0, 0),)
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
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
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
        return (np.dot(self.factors["installation_costs"], x) <= self.factors["installation_budget"])

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        # Generate random solution using acceptable/rejection.
        # TO DO: More efficiently sample uniformly from the simplex.
        while True:
            x = tuple([self.factors["installation_budget"]*rand_sol_rng.random() for _ in range(self.dim)])
            if self.check_deterministic_constraints(x):
                break
        return x