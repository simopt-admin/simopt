"""
Summary
-------
Simulate demand at facilities.
"""
import numpy as np

from base import Model, Problem


class EmergencyRoom(Model):
    """
    A model that simulates a facilitysize problem with a
    multi-variate normal distribution.
    Returns the probability of violating demand in each scenario.

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
        self.name = "Emergency Room"
        self.n_rngs = 7 # how many rngs?
        self.n_responses = 1
        self.specifications = {
            "employee_allocations": {
                "description": "Allocations of each type of employee: receptionists, doctors, lab technicians,"
                "treatment nurses, and ER nurses, respectively.",
                "datatype": list,
                "default": [1, 4, 3, 2, 5]
            },
            "employee_max": {
                "description": "Maximums of each type of employee: receptionists, doctors, lab technicians,"
                "treatment nurses, and ER nurses, respectively.",
                "datatype": list,
                "default": [3, 6, 5, 6, 12]
            },
            "employee_min": {
                "description": "Minimums of each type of employee: receptionists, doctors, lab technicians,"
                "treatment nurses, and ER nurses, respectively.",
                "datatype": list,
                "default": [1, 4, 2, 2, 8]
            },
            "employee_salaries": {
                "description": "Yearly salaries for each type of employee: receptionists, doctors, "
                "lab technicians, treatment nurses, and ER nurses, respectively.",
                "datatype": list,
                "default": [40000, 120000, 50000, 35000, 35000]
            },
            "st_reception": {
                "description": "Exponential distribution parameter for reception service time.",
                "datatype": float,
                "default": 1/7.5
            },
            "st_labtests_min": {
                "description": "Service time minimum for triangular distribution of lab tests.",
                "datatype": int,
                "default": 10
            },
            "st_labtests_mode": {
                "description": "Service time mode for triangular distribution of lab tests.",
                "datatype": int,
                "default": 20
            },
            "st_labtests_max": {
                "description": "Service time maximum for triangular distribution of lab tests.",
                "datatype": int,
                "default": 30
            },
            "st_er": {
                "description": "Exponential distribution parameter for emergency room service time.",
                "datatype": float,
                "default": 1/90
            },
            "st_exam": {
                "description": "Exponential distribution parameter for examination room service time.",
                "datatype": float,
                "default": 1/15
            },
            "st_reexam": {
                "description": "Exponential distribution parameter for re-examination process service time.",
                "datatype": float,
                "default": 1/9
            },
            "st_tr_min": {
                "description": "Service time minimum for triangular distribution of treatments.",
                "datatype": int,
                "default": 20
            },
            "st_tr_mode": {
                "description": "Service time mode for triangular distribution of treatments.",
                "datatype": int,
                "default": 28
            },
            "st_tr_max": {
                "description": "Service time maximum for triangular distribution of treatments.",
                "datatype": int,
                "default": 30
            },
            "walkin_rates": {
                "description": "List of arrival rates for walk ins. Each element represents the arrival"
                "rate for the next two hours in a 24 hour day starting at hour 0.",
                "datatype": list,
                "default": [5.25, 3.8, 3, 4.8, 7, 8.25, 9, 7.75, 7.75, 8, 6.5, 3.25]
            },
            "prob_extra": {
                "description": "Probability that extra tests are needed.",
                "datatype": float,
                "default": 0.5
            },
            "prob_treatmentneeded": {
                "description": "Probability that treatment is needed.",
                "datatype": float,
                "default": 0.8
            },
            "prob_majorinjury": {
                "description": "Probability that a patient has a major injury.",
                "datatype": float,
                "default": 0.5
            },
            "warm_period": {
                "description": "Warm up period for simulation, in days.",
                "datatype": int,
                "default": 4
            },
            "run_time": {
                "description": "Run time of simulation, in days",
                "datatype": int,
                "default": 100
            },
            "amb_arr": {
                "description": "The rate per hour of ambulance arrivals",
                "datatype": int,
                "default": 2
            },

        }
        self.check_factor_list = {
            "employee_allocations": self.check_employee_allocations,
            "employee_max": self.check_employee_max,
            "employee_min": self.check_employee_min,
            "employee_salaries": self.check_employee_salaries,
            "st_reception": self.check_st_reception,
            "st_labtests_min": self.check_st_labtests_min,
            "st_labtests_mode": self.check_st_labtests_mode,
            "st_labtests_max": self.check_st_labtests_max,
            "st_er": self.check_st_er,
            "st_exam": self.check_st_exam,
            "st_reexam": self.check_st_reexam,
            "st_tr_min": self.check_st_tr_min,
            "st_tr_mode": self.check_st_tr_mode,
            "st_tr_max": self.check_st_tr_max,
            "walkin_rates": self.check_walkin_rates,
            "prob_extra": self.check_prob_extra,
            "prob treatmentneeded": self.check_prob_treatmentneeded,
            "prob_majorinjury": self.check_prob_majorinjury,
            "warm_period": self.check_warm_period,
            "run_time": self.check_run_time,
            "amb_arr": self.check_amb_arr
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)
    def check_simulatable_factors(self):
        if len(self.factors["employee_allocations"]) != len(self.factors["employee_max"]):
            return False
        elif len(self.factors["employee_allocations"]) != len(self.factors["employee_min"]):
            return False
        elif len(self.factors["employee_allocations"]) != len(self.factors["employee_employee_salaries"]):
            return False
        else:
            return True

    def check_employee_allocations(self):
        for i in range(len(self.factors["employee_allocations"])):
            if self.factors["employee_allocations"][i] > self.factors["employee_max"][i]:
                return False
            elif self.factors["employee_allocations"][i] < self.factors["employee_min"][i]:
                return False
            elif i+1 == len(self.factors["employee_allocations"]):
                return np.all(self.factors["employee_allocations"]) > 0
        return False

    def check_employee_min(self):
        return np.all(self.factors["employee_min"]) > 0

    def check_employee_max(self):
        return np.all(self.factors["employee_max"]) > 0
    
    def check_employee_salaries(self):
        return np.all(self.factors["employee_salaries"]) > 0

    def check_st_reception(self):
        return self.factors["st_reception"] > 0
    
    def check_st_labtests_min(self):
        return self.factors["st_labtests_min"] > 0

    def check_st_labtests_mode(self):
        return self.factors["st_labtests_mode"] > 0

    def check_st_labtests_max(self):
        return self.factors["st_labtests_max"] > 0

    def check_st_er(self):
        return self.factors["st_er"] > 0

    def check_st_exam(self):
        return self.factors["st_labtests_exam"] > 0

    def check_n_fac(self):
        return self.factors["n_fac"] > 0
    
    def check_walkin_rates(self):
        for i in range(len(self.factors["walkin_rates"])):
            if self.factors["walkin_rates"][i] < 0:
                return False
        return len(self.factors(["walkin_rates"])) > 0

    def check_simulatable_factors(self):
        if len(self.factors["capacity"]) != self.factors["n_fac"]:
            return False
        elif len(self.factors["mean_vec"]) != self.factors["n_fac"]:
            return False
        elif len(self.factors["cov"]) != self.factors["n_fac"]:
            return False
        elif len(self.factors["cov"][0]) != self.factors["n_fac"]:
            return False
        else:
            return True
    def check_st_reexam(self):
        return self.factors["st_reexam"] > 0

    def check_st_tr_min(self):
        return self.factors["st_tr_min"] > 0
    
    def check_st_tr_mode(self):
        return self.factors["st_tr_mode"] > 0

    def check_st_tr_max(self):
        return self.factors["st_tr_max"] > 0

    def check_prob_extra(self):
        return self.factors["prob_extra"] > 0

    def check_prob_treatmentneeded(self):
        return self.factors["prob_treatmentneeded"] > 0

    def check_prob_majorinjury(self):
        return self.factors["prob_majorinjury"] > 0

    def check_warm_period(self):
        return self.factors["prob_warm_period"] > 0
    
    def check_run_time(self):
        return self.factors["run_time"] > 0

    def check_amb_arr(self):
        return self.factors["amb_arr"] > 0

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
            performance measures of interest
            "stockout_flag" = a binary variable
                 0 : all facilities satisfy the demand
                 1 : at least one of the facilities did not satisfy the demand
            "n_fac_stockout" = the number of facilities which cannot satisfy the demand
            "n_cut" = the number of toal demand which cannot be satisfied
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate RNG for demands.
        arrivals_rng = rng_list[0]
        reception_rng = rng_list[1]
        lab_rng = rng_list[2]
        exam_rng = rng_list[3]
        reExam_rng = rng_list[4]
        treatment_rng = rng_list[5]
        er_rng = rng_list[6]
        uniform_rng = rng_list[7]
        amb_rng = rng_list[8]

        arrival_times = []
        arrival_amb = []
        # Generate random arrival times at facilities from truncated multivariate normal distribution.
        t = arrivals_rng.expovariate(self.factors["walkin_rates"][0])

        while t <= (self.factors["run_time"] + self.factors["warm_period"]) * 60:
            if t <= 120:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][0])
            elif t <= 240:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][1])
            elif t <= 360:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][2])
            elif t <= 480:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][3])
            elif t <= 600:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][4])
            elif t <= 720:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][5])
            elif t <= 840:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][6])
            elif t <= 960:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][7])
            elif t <= 1080:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][8])
            elif t <= 1200:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][9])
            elif t <= 1320:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][10])
            elif t <= 1440:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][11])

        t = arrivals_rng.expovariate(self.factors["walkin_rates"][0])

        while t <= (self.factors["run_time"] + self.factors["warm_period"]) * 60:
            arrival_amb.append(t)
            t += amb_rng.expovariate(1/self.factors["amb_arr"])
        
        wait_times = []

        #wait times for receptionist
        while len(wait_times) <= len(arrival_times):


        #generating the wait times for each patient
        #amb does not go to receptionist, everything else the same 
        #wait time can not exceed two hours

        # Compose responses and gradients.
        responses = {'stockout_flag': stockout_flag,
                     'n_fac_stockout': n_fac_stockout,
                     'n_cut': n_cut}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients

        #need to generate amount of employees in a given sim

"""
Summary
-------
Minimize the (deterministic) total cost of installing capacity at
facilities subject to a chance constraint on stockout probability.
"""


class EmergencyRoom(Problem):
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
