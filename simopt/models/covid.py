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
        self.n_rngs = 6
        self.n_responses = 4
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
                "default": 2
            },
            "lamb_inf_sym": {
                "description": "Lambda for from infectious to symptomamatic (mean 3 days).",
                "datatype": float,
                "default": 3
            },
            "lamb_sym": {
                "description": "Lambda for in symptomatic state (mean 12 days).",
                "datatype": float,
                "default": 12
            },
            "n": {
                "description": "Number of days to simulate.",
                "datatype": int,
                "default": 100
            },
            "init_infect_percent": {
                "description": "Initial proportion of infected.",
                "datatype": tuple,
                "default": (0.00156, 0.00161, 0.00166)
            },
            "freq":{
                "description": "Testing frequency of each group.",
                "datatype": tuple,
                "default": (3/7, 3/7, 3/7)
            },
            "asymp_rate":{
                "description": "Percentage of asymptomatic among all the confirmed cases",
                "datatype": float,
                "default": 0.35
            },
            "false_pos":{
                "description": "Percentage of asymptomatic among all the confirmed cases",
                "datatype": float,
                "default": 0.23
            },
            "false_neg":{
                "description": "Percentage of asymptomatic among all the confirmed cases",
                "datatype": float,
                "default": 0.12
            },
            "iso_day":{
                "description": "Number of days in isolation",
                "datatype": int,
                "default": 7
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
            "asymp_rate": self.check_asymp_rate,
            "false_pos": self.check_false_pos,
            "false_neg": self.check_false_neg,
            "iso_day": self.check_iso_day,
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

    def check_false_pos(self):
        return (self.factors["false_pos"] > 0) & (self.factors["false_pos"] < 1)

    def check_false_neg(self):
        return (self.factors["false_neg"] > 0) & (self.factors["false_neg"] < 1)
    def check_iso_day(self):
        return self.factors["iso_day"] > 0

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
            "num_infected" = number of infected individuals per day
            "num_susceptible" = number of susceptible individuals per day
            "num_exposed" = number of exposed individuals per day
            "num_recovered" = number of infected individuals per day
        """
        # Designate random number generator for generating Poisson random variables.
        poisson_numexp_rng = rng_list[0]
        poisson_exp_inf_rng = rng_list[1]
        poisson_inf_sym_rng = rng_list[2]
        binom_asymp_rng = rng_list[3]
        poisson_sym_rng = rng_list[4]
        binom_test_rng = rng_list[5]

        # Calculate the transmission rate
        t_rate = np.array(self.factors["inter_rate"]) * self.factors["p_trans"]
        # Initialize states, each row is one day, each column is one group
        susceptible = np.zeros((self.factors["n"], self.factors["num_groups"]))
        # quarantine = np.zeros((self.factors["n"], self.factors["num_groups"]))
        exposed = np.zeros((self.factors["n"], self.factors["num_groups"]))
        infectious = np.zeros((self.factors["n"], self.factors["num_groups"]))
        asymptomatic = np.zeros((self.factors["n"], self.factors["num_groups"]))
        symptomatic = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation_exp = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation_inf = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation_symp = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation_asymp = np.zeros((self.factors["n"], self.factors["num_groups"]))

        recovered = np.zeros((self.factors["n"], self.factors["num_groups"]))

        # Initialize the performance measures of interest
        new_infectious = np.zeros(self.factors["n"])
        num_infected = np.zeros(self.factors["n"])
        num_exposed = np.zeros(self.factors["n"])
        num_recovered = np.zeros(self.factors["n"])
        num_susceptible = np.zeros(self.factors["n"])
        # Add day 0 num infections
        infectious[0,:] = np.ceil(np.multiply(list(self.factors["group_size"]), list(self.factors["init_infect_percent"])))
        susceptible[0,:] = np.subtract(list(self.factors["group_size"]), infectious[0, :])
        num_infected[0] = np.sum(infectious[0,:])
        num_susceptible[0] = np.sum(susceptible[0,:])

        # Loop through day 1 - day n-1
        for day in range(1, self.factors['n']):
            # update the states from the day before
            if day < self.factors["n"]:
                susceptible[day, :] += susceptible[day - 1, :]
                exposed[day, :] += exposed[day- 1, :]
                infectious[day, :] += infectious[day- 1, :]
                isolation_exp[day, :] += isolation_exp[day- 1, :]
                isolation_asymp[day, :] += isolation_asymp[day- 1, :]
                isolation_symp[day, :] += isolation_symp[day- 1, :]
                isolation_inf[day, :] += isolation_inf[day- 1, :]
                asymptomatic[day, :] += asymptomatic[day- 1, :]
                symptomatic[day, :] += symptomatic[day- 1, :]
                recovered[day, :] += recovered[day- 1, :]
        
            # Generate number of tested out
            tested_out_free_exp = []
            tested_out_free_inf = []
            tested_out_free_symp = []
            tested_out_free_asymp = []
            for g in range(self.factors["num_groups"]):
                tested_out_free_exp.append(binom_test_rng.binomialvariate((exposed[day][g] - isolation_exp[day][g]).astype(int), self.factors["freq"][g]))
                tested_out_free_inf.append(binom_test_rng.binomialvariate((infectious[day][g] - isolation_inf[day][g]).astype(int), self.factors["freq"][g]))
                tested_out_free_symp.append(binom_test_rng.binomialvariate((symptomatic[day][g] -isolation_symp[day][g]).astype(int), self.factors["freq"][g]))
                tested_out_free_asymp.append(binom_test_rng.binomialvariate((asymptomatic[day][g] - isolation_asymp[day][g]).astype(int), self.factors["freq"][g]))

            # update people in isolation
            isolation_exp[day, :] += np.array(tested_out_free_exp)
            isolation_inf[day, :] += np.array(tested_out_free_inf)
            isolation_symp[day, :] += np.array(tested_out_free_symp)
            isolation_asymp[day, :] += np.array(tested_out_free_asymp)

            # generate number of exposed from the transmission matrix and update exposed and susceptible
            num_exp = np.multiply(np.multiply(t_rate, (infectious[day, :] + symptomatic[day, :] + asymptomatic[day, :])),(susceptible[day, :]/(susceptible[day, :] + exposed[day, :] + infectious[day, :] + symptomatic[day, :] + asymptomatic[day, :] + recovered[day, :])))
            num_exp = [min(poisson_numexp_rng.poissonvariate(num_exp[i]), susceptible[day, i]) for i in range(self.factors["num_groups"])]

            exposed[day, :] = np.add(exposed[day, :], num_exp)
            susceptible[day, :] = np.subtract(susceptible[day, :], num_exp)
            # update new_infectious
            new_infectious[day] = np.sum(num_exp)

            # generate number of days remaining in exposed and update exposed and infectious (by new infectious of the day)
            exp_days = min(poisson_exp_inf_rng.poissonvariate(self.factors["lamb_exp_inf"]), 7)
            if day + exp_days < self.factors["n"]: 
                infectious[day+exp_days,:] = np.add(infectious[day+exp_days,:], num_exp)
                exposed[day+exp_days,:] = np.subtract(exposed[day+exp_days,:], num_exp)
         
            # generate number of days remaining in infectious and update asymptomatic, symptomatic, and infectious
            inf_days = min(poisson_inf_sym_rng.poissonvariate(self.factors["lamb_inf_sym"]), 8)   
            if day + inf_days + exp_days < self.factors["n"]: 
                num_asymp = []
                for g in range(self.factors["num_groups"]):
                    num_asymp.append(binom_asymp_rng.binomialvariate(num_exp[g], self.factors["asymp_rate"]))
                num_symp = np.subtract(num_exp, np.array(num_asymp))
                symptomatic[day + inf_days+exp_days , :] = np.add(symptomatic[day + inf_days+exp_days, :], num_symp)
                asymptomatic[day + inf_days+exp_days, :] = np.add(asymptomatic[day + inf_days+exp_days, :], num_asymp)
                infectious[day + inf_days+exp_days, :] = np.subtract(infectious[day + inf_days+ exp_days, :], num_exp)

            # generate number of days remaining in symptomatic or asymtomatic state, update recovered, symptomatic and asymptomatic
            asym_sym_days = min(poisson_sym_rng.poissonvariate(self.factors["lamb_sym"]), 20)
            if day + asym_sym_days + inf_days + exp_days < self.factors["n"]:
                recovered[day + asym_sym_days+ inf_days+exp_days, :] = np.add(recovered[day + asym_sym_days+ inf_days+exp_days, :], num_exp)
                symptomatic[day + asym_sym_days+ inf_days+exp_days, :] = np.subtract(symptomatic[day + asym_sym_days+ inf_days+exp_days, :], num_symp)
                asymptomatic[day + asym_sym_days+ inf_days+exp_days, :] = np.subtract(asymptomatic[day + asym_sym_days+ inf_days+exp_days, :], num_asymp)

            # Update isolation to recovered
            if day + self.factors["iso_day"] < self.factors["n"]: 
                # recovered[day + self.factors["iso_day"], :] += np.array(tested_out_free_exp) + np.array(tested_out_free_inf) + np.array(tested_out_free_symp) + np.array(tested_out_free_asymp)
                isolation_exp[day + self.factors["iso_day"], :] -= np.array(tested_out_free_exp)
                isolation_inf[day + self.factors["iso_day"], :] -= np.array(tested_out_free_inf)
                isolation_symp[day + self.factors["iso_day"], :] -= np.array(tested_out_free_symp) 
                isolation_asymp[day + self.factors["iso_day"], :] -= np.array(tested_out_free_asymp)
                
            
            print('day', day)
            # print('tested_out_exp', tested_out_free_exp)
            print('new exposed', num_exp)
            print('exposed', exposed[day, :])
            print('day + inf_days', day + inf_days)
            print('infectious', infectious[day, :])
            print('num_symp', num_symp)
            print('num_asymp', num_asymp)
            print('asymptomatic', asymptomatic[day, :])
            print('symptomatic', symptomatic[day, :])
            print('isolation', isolation_asymp[day, :] + isolation_exp[day,:]+isolation_inf[day,:]+isolation_symp[day,:])
            print('susceptible', susceptible[day, :])
            print('recovered', recovered[day, :])

            # update performance measures
            num_exposed[day] = np.sum(exposed[day, :] - isolation_exp[day,:])
            num_susceptible[day] =np.sum(susceptible[day, :])
            num_recovered[day] =np.sum(recovered[day, :])
            num_infected[day] = np.sum(infectious[day, :] - isolation_inf[day,:]) + np.sum(asymptomatic[day, :]-isolation_asymp[day, :]) + np.sum(symptomatic[day, :] - isolation_symp[day,:])

        # Compose responses and gradients.
        responses = {"num_infected": num_infected, "num_exposed": num_exposed, "num_susceptible": num_susceptible, "num_recovered": num_recovered}
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


class CovidMinInfect(Problem):
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
    def __init__(self, name="COVID-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
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
                "default": (3/7, 1/7, 1/7)
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