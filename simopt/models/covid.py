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
        self.n_responses = 1
        self.factors = fixed_factors
        self.specifications = {
            "num_groups": {
                "description": "Number of groups.",
                "datatype": int,
                "default": 3
            },
            "transmission_rate": {
                "description": "Rate of transmission.",
                "datatype": tuple,
                "default": (0.22, 0.072, 0.0018, 0.0018, 0.061, 0.15, 0.0018, 0.0034, 0.0039, 0.072)
            },
            "group_size": {
                "description": "Size of each group.",
                "datatype": tuple,
                "default": (8123, 3645, 4921)
            },
            "lamb_exp_inf": {
                "description": "Mean time from exposure to infectious.",
                "datatype": float,
                "default": 2.
            },
            "lamb_inf_sym": {
                "description": "Mean time from infectious to symptom onset.",
                "datatype": float,
                "default": 3.
            },
            "lamb_sym": {
                "description": "Mean time in symptomatic state.",
                "datatype": float,
                "default": 12.
            },
            "lamb_iso": {
                "description": "Mean number of isolations.",
                "datatype": float,
                "default": 0.85
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
                "default": (3/7, 2/7, 1/7)
            }
        }
        self.check_factor_list = {
            "num_groups": self.check_num_groups,
            "p_trans": self.check_p_trans,
            "group_size": self.check_group_size,
            "lamb_exp_inf": self.check_lamb_exp_inf,
            "lamb_inf_sym": self.check_lamb_inf_sym,
            "lamb_sym": self.check_lamb_sym,
            "lamb_iso": self.check_lamb_iso,
            "n":self.check_n,
            "init_infect_percent": self.check_init_infect
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_num_groups(self):
        return self.factors["num_groups"] > 0

    def check_p_trans(self):
        return all(np.array(self.factors["p_trans"]) >= 0) & (len(self.factors["p_trans"]) == self.factors["num_groups"])

    def check_group_size(self):
        return all(np.array(self.factors["group_size"]) >= 0) & (len(self.factors["group_size"]) == self.factors["num_groups"])

    def check_lamb_exp_inf(self):
        return self.factors["lamb_exp_inf"] > 0

    def check_lamb_inf_sym(self):
        return self.factors["lamb_inf_sym"] > 0

    def check_lamb_sym(self):
        return self.factors["lamb_sym"] > 0

    def check_lamb_iso(self):
        return self.factors["lamb_iso"] > 0

    def check_n(self):
        return self.factors["n"] > 0
    
    def check_init_infect(self):
        return all(np.array(self.factors["init_infect_percent"]) >= 0) & (len(self.factors["init_infect_percent"]) == self.factors["num_groups"])

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
            "num_infected" = number of infected individuals
        """
        # Designate random number generator for generating Poisson random variables.
        poisson_exp_inf_rng = rng_list[0]
        poisson_inf_sym_rng = rng_list[1]
        poisson_sym_rng = rng_list[2]
        poisson_iso_rng = rng_list[3]
        poisson_trace_rng = rng_list[4]
        binom_rng = rng_list[5]
        # Initialize states, each row is one day, each column is one group
        susceptible = []
        quarantine = []
        exposed = []
        infectious = []
        isolation = []
        asymptomatic = []
        symptomatic = []
        recovered = np.zeros(self.factors["num_groups"])
        # Add day 0 num infections
        infectious.append(np.dot(list(self.factors["group_size"]), list(self.factors["init_infect_percent"])))
        susceptible.append(np.subtract(list(self.factors["group_size"]), infectious[0]))
        for s in [quarantine, exposed, isolation, asymptomatic, symptomatic, recovered]:
            s.append(np.zeros(self.factors["num_groups"]))

        # Go through day 1 - day n
        for day in range(1, self.factors["n"]+1):
            #the ppl that get tested out
            tested_out = []
            for g in range(len(self.factors["num_groups"])):
                tested_out.append(binom_rng.binomialvariate(infectious[day-1][g], self.factors["freq"][g]))
            isolation.append(tested_out)
            infectious.append()

        # Compute Poisson rvs...
        gumbel = np.zeros(((self.factors["num_customer"], self.factors["num_prod"])))
        for t in range(self.factors["num_customer"]):
            for j in range(self.factors["num_prod"]):
                gumbel[t][j] = Gumbel_rng.gumbelvariate(-self.factors["mu"] * np.euler_gamma, self.factors["mu"])
        # Compute utility for each product and each customer.
        utility = np.zeros((self.factors["num_customer"], self.factors["num_prod"] + 1))
        for t in range(self.factors["num_customer"]):
            for j in range(self.factors["num_prod"] + 1):
                if j == 0:
                    utility[t][j] = 0
                else:
                    utility[t][j] = self.factors["c_utility"][j - 1] + gumbel[t][j - 1]

        # Initialize inventory.
        inventory = np.copy(self.factors["init_level"])
        itembought = np.zeros(self.factors["num_customer"])

        # Loop through customers
        for t in range(self.factors["num_customer"]):
            instock = np.where(inventory > 0)[0]
            # Initialize the purchase option to be no-purchase.
            itembought[t] = 0
            # Assign the purchase option to be the product that maximizes the utility.
            for j in instock:
                if utility[t][j + 1] > utility[t][int(itembought[t])]:
                    itembought[t] = j + 1
            if itembought[t] != 0:
                inventory[int(itembought[t] - 1)] -= 1

        # Calculate profit.
        numsold = self.factors["init_level"] - inventory
        revenue = numsold * np.array(self.factors["price"])
        cost = self.factors["init_level"] * np.array(self.factors["cost"])
        profit = revenue - cost

        # Compose responses and gradients.
        responses = {"profit": np.sum(profit), "n_prod_stockout": np.sum(inventory == 0)}
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
