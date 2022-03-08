"""
Summary
-------
Simulate the spread of COVID-19 over a period of time.
"""
import numpy as np

from base import Model, Problem


class COVID(Model):
    """
    A model that simulates...

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
    def __init__(self, fixed_factors={}):
        self.name = "COVID"
        self.n_rngs = 8
        self.n_responses = 1
        self.factors = fixed_factors
        self.specifications = {
            "num_groups": {
                "description": "Number of groups.",
                "datatype": int,
                "default": 3
            },
            "p_trans": {
                "description": "Probability of transmission per interaction.",
                "datatype": float,
                "default": 0.018
            },
            "inter_rate": {
                "description": "Interaction rates between two groups per day",
                "datatype": tuple,
                "default": (17.58, 17.58, 13.57)
            },
            "group_size": {
                "description": "Size of each group.",
                "datatype": tuple,
                "default": (8123, 3645, 4921)
            },
            "lamb_exp_inf": {
                "description": " Lambda for from exposed to infectious (mean 2 days).",
                "datatype": float,
                "default": 1/2
            },
            "lamb_inf_sym": {
                "description": "Lambda for from infectious to symptomamatic (mean 3 days).",
                "datatype": float,
                "default": 1/3
            },
            "lamb_sym": {
                "description": "Lambda for in symptomatic state (mean 12 days).",
                "datatype": float,
                "default": 1/12
            },
            "n": {
                "description": "Number of days to simulate.",
                "datatype": int,
                "default": 1000
            },
            "init_infect_percent": {
                "description": "Initial proportion of infected.",
                "datatype": tuple,
                "default": (0.00156, 0.00161, 0.00166)
            },
            "freq":{
                "description": "Testing frequency of each group.",
                "datatype": tuple,
                "default": (3/7, 2/7, 1/7)
            },
            "asymp_rate":{
                "description": "Percentage of asymptomatic among all the confirmed cases",
                "datatype": float,
                "default": 0.35
            },
        }
        self.check_factor_list = {
            "num_groups": self.check_num_groups,
            "p_trans": self.check_p_trans,
            "inter_rate": self.check_inter_rate,
            "group_size": self.check_group_size,
            "lamb_exp_inf": self.check_lamb_exp_inf,
            "lamb_inf_sym": self.check_lamb_inf_sym,
            "lamb_sym": self.check_lamb_sym,
            "n":self.check_n,
            "init_infect_percent": self.check_init_infect_percent,
            "freq": self.check_freq,
            "asymp_rate": self.check_asymp_rate

        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_num_groups(self):
        return self.factors["num_groups"] > 0

    def check_p_trans(self):
        return self.factors["p_trans"] > 0
    
    def check_inter_rate(self):
        return all(np.array(self.factors["inter_rate"]) >= 0) & (len(self.factors["inter_rate"]) == self.factors["num_groups"])

    def check_group_size(self):
        return all(np.array(self.factors["group_size"]) >= 0) & (len(self.factors["group_size"]) == self.factors["num_groups"])

    def check_lamb_exp_inf(self):
        return self.factors["lamb_exp_inf"] > 0

    def check_lamb_inf_sym(self):
        return self.factors["lamb_inf_sym"] > 0

    def check_lamb_sym(self):
        return self.factors["lamb_sym"] > 0

    def check_n(self):
        return self.factors["n"] > 0
    
    def check_init_infect_percent(self):
        return all(np.array(self.factors["init_infect_percent"]) >= 0) & (len(self.factors["init_infect_percent"]) == self.factors["num_groups"])

    def check_freq(self):
        return all(np.array(self.factors["freq"]) >= 0) & (len(self.factors["freq"]) == self.factors["num_groups"])

    def check_asymp_rate(self):
        return self.factors["asymp_rate"] > 0

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
            "num_infected" = average number of infected individuals per day
        """
        # Designate random number generator for generating Poisson random variables.
        poisson_exp_inf_rng = rng_list[0]
        poisson_inf_sym_rng = rng_list[1]
        poisson_sym_rng = rng_list[2]
        binom_rng1 = rng_list[3]
        binom_rng2 = rng_list[4]
        binom_rng3 = rng_list[5]
        binom_rng4 = rng_list[6]
        binom_rng5 = rng_list[7]
        # Calculate the transmission rate
        t_rate = np.array(self.factors["inter_rate"]) * self.factors["p_trans"]
        # Initialize states, each row is one day, each column is one group
        susceptible = np.zeros((self.factors["n"], self.factors["num_groups"]))
        quarantine = np.zeros((self.factors["n"], self.factors["num_groups"]))
        exposed = np.zeros((self.factors["n"], self.factors["num_groups"]))
        infectious = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation = np.zeros((self.factors["n"], self.factors["num_groups"]))
        asymptomatic = np.zeros((self.factors["n"], self.factors["num_groups"]))
        symptomatic = np.zeros((self.factors["n"], self.factors["num_groups"]))
        recovered = np.zeros((self.factors["n"], self.factors["num_groups"]))
        # Initialize the performance measures of interest
        num_infected = []
        # Add day 0 num infections
        infectious[0,:] = np.dot(list(self.factors["group_size"]), list(self.factors["init_infect_percent"]))
        susceptible[0,:] = np.subtract(list(self.factors["group_size"]), infectious[0, :])
        num_infected.append(np.sum(infectious[0,:]))

        # Loop through day 1 - day n-1
        for day in range(1, self.factors["n"]):
            # generate number of exposed from the transmission matrix and update exposed and susceptible
            num_exp = np.ceil(np.matmul(np.transpose(infectious[day-1]), t_rate))
            exposed[day,:] = np.add(exposed[day,:], num_exp)
            susceptible[day, :] = np.subtract(susceptible[day, :], num_exp)
            # update num_infected
            num_infected += np.sum(num_exp)
            # generate number of days remaining in exposed and update exposed and infectious
            exp_days = min(poisson_exp_inf_rng.poissonvariate(self.factors["lamb_exp_inf"]), 7)
            if day + exp_days <= self.factors["n"]: 
                infectious[day+exp_days,:] = np.add(infectious[day+exp_days,:], exposed[day,:])
                exposed[day+exp_days,:] = np.subtract(exposed[day+exp_days,:], exposed[day,:])
            # generate number of days remaining in infectious and update asymptomatic, symptomatic, and infectious
            inf_days = min(poisson_inf_sym_rng.poissonvariate(self.factors["lamb_inf_sym"]), 8)   
            if day + inf_days <= self.factors["n"]: 
                num_asymp = []
                for g in range(self.factors["num_groups"]):
                    num_asymp.append(binom_rng5.binomialvariate(infectious[day][g].astype(int), self.factors["asymp_rate"]))
                num_symp = np.subtract(infectious[day, :], np.array(num_asymp))
                symptomatic[day + inf_days, :] = np.add(symptomatic[day + inf_days, :], num_symp)
                asymptomatic[day + inf_days, :] = np.add(asymptomatic[day + inf_days, :], num_asymp)
                infectious[day + inf_days, :] = np.subtract(infectious[day + inf_days, :], infectious[day, :])
            # generate number of days remaining in symptomatic or asymtomatic state, update recovered, symptomatic and asymptomatic
            sym_days = min(poisson_sym_rng.poissonvariate(self.factors["lamb_sym"]), 20)
            asym_days = min(poisson_sym_rng.poissonvariate(self.factors["lamb_sym"]), 20)
            if day + sym_days <= self.factors["n"]:
                recovered[day + sym_days, :] = np.add(recovered[day + sym_days, :], symptomatic[day, :])
                symptomatic[day + sym_days, :] = np.subtract(symptomatic[day + sym_days, :], symptomatic[day, :])
            if day + asym_days <= self.factors["n"]:
                recovered[day + asym_days, :] = np.add(recovered[day + asym_days, :], asymptomatic[day, :])
                symptomatic[day + asym_days, :] = np.subtract(asymptomatic[day + asym_days, :], asymptomatic[day, :])
            # Generate number of isolations, update exposed, infectious, asymptomatic, symptomatic, and isolation
            tested_out_exp = []
            tested_out_inf = []
            tested_out_asym = []
            tested_out_sym = []
            for g in range(self.factors["num_groups"]):
                tested_out_exp.append(binom_rng1.binomialvariate(exposed[day-1][g].astype(int), self.factors["freq"][g]))
                tested_out_inf.append(binom_rng2.binomialvariate(infectious[day-1][g].astype(int), self.factors["freq"][g]))
                tested_out_asym.append(binom_rng3.binomialvariate(asymptomatic[day-1][g].astype(int), self.factors["freq"][g]))
                tested_out_sym.append(binom_rng4.binomialvariate(symptomatic[day-1][g].astype(int), self.factors["freq"][g]))
            isolation[day,:] = [tested_out_exp[i] + tested_out_inf[i] + tested_out_asym[i] + tested_out_sym[i] for i in range(self.factors["num_groups"])]
            exposed[day, :] = [tested_out_exp[i] for i in range(self.factors["num_groups"])]
            infectious[day, :] = [tested_out_inf[i] for i in range(self.factors["num_groups"])]
            asymptomatic[day, :] = [tested_out_asym[i] for i in range(self.factors["num_groups"])]
            symptomatic[day, :] = [tested_out_sym[i] for i in range(self.factors["num_groups"])]


        # Compose responses and gradients.
        responses = {"num_infected": np.mean(num_infected)}
        gradients = {response_key:
                     {factor_key: np.nan for factor_key in self.specifications}
                     for response_key in responses
                     }
        return responses, gradients


"""
Summary
-------
Maximize the expected profit for the continuous newsvendor problem.
"""


class DynamNewsMaxProfit(Problem):
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
    def __init__(self, name="DYNAMNEWS-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "discrete"
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_fixed_factors = {}
        self.model_decision_factors = {"init_level"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (2, 3)
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
        self.model = DynamNews(self.model_fixed_factors)
        self.dim = self.model.factors["num_prod"]
        self.lower_bounds = (0,) * self.dim
        self.upper_bounds = (np.inf,) * self.dim

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
            "init_level": vector[:]
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
        vector = tuple(factor_dict["init_level"])
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
        objectives = (response_dict["profit"],)
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
        det_objectives_gradients = ((0,),)
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
        x = tuple([rand_sol_rng.uniform(0, 10) for _ in range(self.dim)])
        return x