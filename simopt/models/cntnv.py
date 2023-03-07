"""
Summary
-------
Simulate a day's worth of sales for a newsvendor.
A detailed description of the model/problem can be found `here <https://simopt.readthedocs.io/en/latest/cntnv.html>`_.
"""
import numpy as np
import pulp
from ..base import Model, Problem

class FactNV(Model):
    """
    A model that simulates a day's worth of sales for a newsvendor
    with a Burr Type XII demand distribution. Returns the profit, after
    accounting for order costs and salvage.

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
        self.name = "FactNews"
        self.n_rngs = 1
        self.n_responses = 1
        self.factors = fixed_factors
        self.specifications = {
            "num_customer": {
                "description": "number of customers. Only input if simulating dynamic NV",
                "datatype": int,
                "default": None
            },
            "num_resource": {
                "description": "number of resources",
                "datatype": int,
                "default": 5
            },
            "num_product": {
                "description": "number of products",
                "datatype": int,
                "default": 5
            },
            "res_to_prod": {
                "description": "P by R matrix mapping R resources to P products",
                "datatype": list,
                "default": list(np.identity(5))
            },
            "purchase_cost": {
                "description": "purchasing cost per resource unit",
                "datatype": list,
                "default": list(5 * np.ones(5))
            },
            "recourse_cost": {
                "description": "recourse purchasing cost per resource unit",
                "datatype": list,
                "default": list(10 * np.ones(5))
            },
            "processing_cost": {
                "description": "processing cost per product",
                "datatype": list,
                "default": list(0.1 * np.ones(5))
            },
            "order_cost": {
                "description": "fixed one-time ordering cost",
                "datatype": float,
                "default": 0
            },
            "purchase_yield": {
                "description": "yield rate of purchased materials (in FactNV) or products (in RetaNV)",
                "datatype": list,
                "default": list(np.ones(5))
            },
            "sales_price": {
                "description": "sales price per product unit",
                "datatype": list,
                "default": list(9 * np.ones(5))
            },
            "salvage_price": {
                "description": "salvage cost per product unit",
                "datatype": float,
                "default": list(np.ones(5))
            },
            "order_quantity": {
                "description": "order quantity per resource (in FactNV) or product (in RetaNV)",
                "datatype": list,  # or int
                "default": list(20* np.ones(5))
            },
            "recourse_quantity": {
                "description": "recourse order quantity per resource",
                "datatype": list,  # or int
                "default": list(1* np.ones(5))
            },
            "Burr_c": {
                "description": "Burr Type XII cdf shape parameter",
                "datatype": float,
                "default": 2.0
            },
            "Burr_k": {
                "description": "Burr Type XII cdf shape parameter",
                "datatype": float,
                "default": 20.0
            },
            "mu": {
                "description": "mu for calculating Gumbel random variable",
                "datatype": float,
                "default": None
            },
            "c_utility": {
                "description": "constant of each product's utility",
                "datatype": list,
                "default": None
            }
        }
        self.check_factor_list = {
            "num_customer": self.check_customer,
            "num_resource": self.check_num_resource,
            "num_product": self.check_num_product,
            "res_to_product": self.check_res_to_prod,
            "purchase_cost": self.check_purchase_cost,
            "processing_cost": self.check_processing_cost,
            "order_cost": self.check_order_cost,
            "purchase_yield": self.check_purchase_yield,
            "sales_price": self.check_sales_price,
            "salvage_price": self.check_salvage_price,
            "order_quantity": self.check_order_quantity,
            "recource_quantity": self.check_recourse_quantity,
            "Burr_c": self.check_Burr_c,
            "Burr_k": self.check_Burr_k,
            "mu": self.check_mu,
            "c_utility": self.check_utility
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_customer(self):
        return (self.factors["num_customer"] is None)
    
    def check_num_resource(self):
        return (self.factors["num_resource"] > 0)
    
    def check_num_product(self):
        return (self.factors["num_product"] > 0)
    
    def check_res_to_prod(self):
        return (np.shape(self.factors["res_to_prod"])[0] == self.factors["num_product"]) and (np.shape(self.factors["res_to_prod"])[1] == self.factors["num_resource"])

    def check_purchase_cost(self):
        return all(np.array(self.factors["purchase_cost"]) > 0) and (len(self.factors["purchase_cost"]) == self.factors["num_resource"])

    def check_recourse_cost(self):
        if self.factors["recourse_cost"] != None:
            return all(np.array(self.factors["recourse_cost"]) > 0) and (len(self.factors["recourse_cost"]) == self.factors["num_resource"])
        else:
            return True
    
    def check_processing_cost(self):
        return all(np.array(self.factors["processing_cost"]) >= 0) and (len(self.factors["processing_cost"]) == self.factors["num_product"])
        
    def check_order_cost(self):
        return (self.factors["order_cost"] >= 0)
        
    def check_purchase_yield(self):
        return all(self.factors["purchase_yield"] <= 1) and (len(self.factors["purchase_yield"]) == self.factors["num_resource"])

    def check_sales_price(self):
        return all(np.array(self.factors["sales_price"]) > 0)

    def check_salvage_price(self):
        return all(np.array(self.factors["salvage_price"]) > 0)

    def check_order_quantity(self):
        return all(np.array(self.factors["order_quantity"]) > 0) and (len(self.factors["order_quantity"]) == self.factors["num_resource"])

    def check_recourse_quantity(self):
        return all(np.array(self.factors["recourse_quantity"]) >= 0) and (len(self.factors["recourse_quantity"]) == self.factors["num_resource"])

    def check_Burr_c(self):
        return self.factors["Burr_c"] > 0

    def check_Burr_k(self):
        return self.factors["Burr_k"] > 0
    
    def check_mu(self):
        return True
    
    def check_utility(self):
        return True
    
    def check_simulatable_factors(self):
        return all((self.factors["sales_price"] - np.dot(self.factors["purchase_cost"], np.array(self.factors["res_to_prod"]).T)) > 0)
    
    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "profit" = profit in this scenario
            "stockout_qty" = amount by which demand exceeded supply
            "stockout" = was there unmet demand? (Y/N)
        """
        # Designate random number generator for demand variability.
        demand_rng = rng_list[0]
        # Generate random demand according to Burr Type XII distribution.
        # If U ~ Uniform(0,1) and the Burr Type XII has parameters c and k,
        #   X = ((1-U)**(-1/k - 1))**(1/c) has the desired distribution.
        demand = []
        for i in range(self.factors["num_product"]):
            base = ((1 - demand_rng.random())**(-1 / self.factors["Burr_k"]) - 1)
            exponent = (1 / self.factors["Burr_c"])
            unit_demand = base**exponent
            demand.append(unit_demand)
   
        # define IP problem for production of goods to fulfill realized demand
        X = pulp.LpVariable.dicts("X", range(self.factors["num_product"]), lowBound=0, cat='Integer')
        Y = pulp.LpVariable.dicts("Y", range(self.factors["num_resource"]), lowBound=0, cat='Integer')
        prob = pulp.LpProblem("Integer Program", pulp.LpMaximize)
        prob += pulp.lpSum([self.factors["sales_price"][i]*X[i] for i in range(self.factors["num_product"])]) - \
        pulp.lpSum([X[i]*np.dot(self.factors["res_to_prod"], self.factors["purchase_cost"])[i] for i in range(self.factors["num_product"])]) - \
        pulp.lpSum([Y[i]*self.factors["recourse_cost"][i] for i in range(self.factors["num_resource"])])

        # Define constraints
        prob += pulp.lpSum([X[i]*np.dot(self.factors["res_to_prod"], self.factors["purchase_cost"])[i] for i in range(self.factors["num_product"])]) <= self.factors["prod_budget"]
        for j in range(self.factors["num_resource"]):
            prob += pulp.lpSum([self.factors["res_to_prod"][i][j]*X[i] for i in range(self.factors["num_product"])]) <= self.factors["order_quantity"][j] + Y[j]
        for i in range(self.factors["num_product"]):
            prob += X[i] <= demand[i]
        prob.solve()

        # Results of IP
        Finish_Goods = np.array([pulp.value(X[i]) for i in range(self.factors["num_product"])])
        Recourse = np.array([pulp.value(Y[i]) for i in range(self.factors["num_resource"])])

        # Calculate profit.
        total_cost = (self.factors["purchase_cost"] * self.factors["order_quantity"] + \
                        Recourse * self.factors["recourse_cost"] + \
                        Finish_Goods * self.factors["processing_cost"] + self.factors["order_cost"])
        sales_revenue = (np.array(list(map(min, demand, Finish_Goods)))
                         * self.factors["sales_price"])
        salvage_revenue = ((np.where((Finish_Goods-demand)<0, 0, (Finish_Goods-demand)))
                           * self.factors["salvage_price"])
        profit = sales_revenue + salvage_revenue - total_cost

        """HOW TO MODIFY BELOW?"""
        stockout_qty = max(demand - self.factors["order_quantity"], 0)
        stockout = int(stockout_qty > 0)
        # Calculate gradient of profit w.r.t. order quantity.
        if demand > self.factors["order_quantity"]:
            grad_profit_order_quantity = (self.factors["sales_price"]
                                          - self.factors["purchase_price"])
        elif demand < self.factors["order_quantity"]:
            grad_profit_order_quantity = (self.factors["salvage_price"]
                                          - self.factors["purchase_price"])
        else:
            grad_profit_order_quantity = np.nan
        # Compose responses and gradients.
        responses = {"profit": profit, "stockout_qty": stockout_qty, "stockout": stockout}
        gradients = {response_key:
                     {factor_key: np.nan for factor_key in self.specifications}
                     for response_key in responses
                     }
        gradients["profit"]["order_quantity"] = grad_profit_order_quantity
        return responses, gradients

class RetaNV(FactNV):
    """
    Child of FactNV class. Several parameters are modified accordingly, along with the replicate method. 
    """
    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "RetaNews"
        self.factors = fixed_factors
        self.specifications = {
            "num_resource": {
                "description": "number of resources",
                "datatype": int,
                "default": None
            },
            "res_to_prod": {
                "description": "P by R matrix mapping R resources to P products",
                "datatype": list,
                "default": None
            },
            "processing_cost": {
                "description": "processing cost per product",
                "datatype": list,
                "default": None
            },
        }
        super().__init__(fixed_factors)

    def check_num_resource(self):
        return (self.factors["num_resource"] is None)
      
    def check_res_to_prod(self):
        return (self.factors["res_to_prod"] is None)

    def check_purchase_cost(self):
        return all(np.array(self.factors["purchase_cost"]) > 0) and (len(self.factors["purchase_cost"]) == self.factors["num_product"])

    def check_recourse_cost(self):
        return all(np.array(self.factors["recourse_cost"]) > 0) and (len(self.factors["recourse_cost"]) == self.factors["num_product"])
    
    def check_processing_cost(self):
        return (self.factors["processing_cost"] is None)
        
    def check_purchase_yield(self):
        return all(self.factors["purchase_yield"] <= 1) and (len(self.factors["purchase_yield"]) == self.factors["num_product"])

    def check_order_quantity(self):
        return all(np.array(self.factors["order_quantity"]) > 0) and (len(self.factors["order_quantity"]) == self.factors["num_product"])

    def check_recourse_quantity(self):
        return all(np.array(self.factors["recourse_quantity"]) >= 0) and (len(self.factors["recourse_quantity"]) == self.factors["num_product"])

    def check_simulatable_factors(self):
        return all((self.factors["sales_price"] - self.factors["purchase_cost"]) > 0)
    
    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.
        Arguments
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication
        Returns
        -------
        responses : dict
            performance measures of interest
            "profit" = profit in this scenario
            "stockout_qty" = amount by which demand exceeded supply
            "stockout" = was there unmet demand? (Y/N)
        """
        # Designate random number generator for demand variability.
        demand_rng = rng_list[0]
        # Generate random demand according to Burr Type XII distribution.
        # If U ~ Uniform(0,1) and the Burr Type XII has parameters c and k,
        #   X = ((1-U)**(-1/k - 1))**(1/c) has the desired distribution.
        demand = []
        for i in range(self.factors["num_product"]):
            base = ((1 - demand_rng.random())**(-1 / self.factors["Burr_k"]) - 1)
            exponent = (1 / self.factors["Burr_c"])
            unit_demand = base**exponent
            demand.append(unit_demand)

        # Calculate profit.
        total_cost = self.factors["purchase_cost"] * self.factors["order_quantity"] + \
                        self.factors["recourse_cost"] * self.factors["recourse_quantity"] +\
                        self.factors["order_cost"]
        sales_revenue = (np.array(list(map(min, demand, (self.factors["order_quantity"]+self.factors["recourse_quantity"]))))
                         * self.factors["sales_price"])
        salvage_revenue = ((np.where(((self.factors["order_quantity"]+self.factors["recourse_quantity"])-demand)<0, 0, (self.factors["order_quantity"]+self.factors["recourse_quantity"]-demand)))
                           * self.factors["salvage_price"])
        profit = sales_revenue + salvage_revenue - total_cost

        """HOW TO MODIFY BELOW?"""
        stockout_qty = max(demand - self.factors["order_quantity"], 0)
        stockout = int(stockout_qty > 0)
        # Calculate gradient of profit w.r.t. order quantity.
        if demand > self.factors["order_quantity"]:
            grad_profit_order_quantity = (self.factors["sales_price"]
                                          - self.factors["purchase_price"])
        elif demand < self.factors["order_quantity"]:
            grad_profit_order_quantity = (self.factors["salvage_price"]
                                          - self.factors["purchase_price"])
        else:
            grad_profit_order_quantity = np.nan
        # Compose responses and gradients.
        responses = {"profit": profit, "stockout_qty": stockout_qty, "stockout": stockout}
        gradients = {response_key:
                     {factor_key: np.nan for factor_key in self.specifications}
                     for response_key in responses
                     }
        gradients["profit"]["order_quantity"] = grad_profit_order_quantity
        return responses, gradients
    
class DynamNV(RetaNV):

    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "DynamNews"
        self.factors = fixed_factors
        self.specifications = {
            "num_customer": {
                "description": "number of customers. Only input if simulating dynamic NV",
                "datatype": int,
                "default": 5
            },
            "recourse_quantity": {
                "description": "recourse order quantity per resource",
                "datatype": list,  # or int
                "default": None
            },
            "mu": {
                "description": "mu for calculating Gumbel random variable",
                "datatype": float,
                "default": 1.0
            },
            "c_utility": {
                "description": "constant of each product's utility",
                "datatype": list,
                "default": [5 + j for j in range(1, 6)]
            }
        }
        super().__init__(fixed_factors)

    def check_customer(self):
        return (self.factors["num_customer"] > 0)
    
    def check_recourse_quantity(self):
        return (self.factors["recourse_quantity"] is None)
    
    def check_mu(self):
        return (self.factors["mu"] is not None)
    
    def check_utility(self):
        return (len(self.factors["c_utility"]) == self.factors["num_product"])
    
    def replicate(self, rng_list):
        """
        A method for dynamic newsvendor model. Customers purchase products according to their utility level. 
        """
        # Designate random number generator for generating a Gumbel random variable.
        Gumbel_rng = rng_list[0]
        # Compute Gumbel rvs for the utility of the products.
        gumbel = np.zeros(((self.factors["num_customer"], self.factors["num_product"])))
        for t in range(self.factors["num_customer"]):
            for j in range(self.factors["num_product"]):
                gumbel[t][j] = Gumbel_rng.gumbelvariate(-self.factors["mu"] * np.euler_gamma, self.factors["mu"])
        # Compute utility for each product and each customer.
        utility = np.zeros((self.factors["num_customer"], self.factors["num_product"] + 1))
        for t in range(self.factors["num_customer"]):
            for j in range(self.factors["num_product"] + 1):
                if j == 0:
                    utility[t][j] = 0
                else:
                    utility[t][j] = self.factors["c_utility"][j - 1] + gumbel[t][j - 1]

        # Initialize inventory.
        inventory = np.copy(self.factors["order_quantity"])
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

        numsold = self.factors["order_quantity"] - inventory
        revenue = numsold * np.array(self.factors["sales_price"])
        salvage = inventory * self.factors["salvage_price"]
        cost = self.factors["order_quantity"] * np.array(self.factors["pruchase_cost"]) + self.factors["order_cost"]
        profit = revenue - cost + salvage

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


class CntNVMaxProfit(Problem):
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
    def __init__(self, name="CNTNEWS-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.dim = 1
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.lower_bounds = (0,)
        self.upper_bounds = (np.inf,)
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None  # (0.1878,)  # TO DO: Generalize to function of factors.
        self.model_default_factors = {
            "purchase_price": 5.0,
            "sales_price": 9.0,
            "salvage_price": 1.0,
            "Burr_c": 2.0,
            "Burr_k": 20.0
            }
        self.model_decision_factors = {"order_quantity"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (0,)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
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
        self.model = Problem(self.model_fixed_factors)

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
            "order_quantity": vector[0]
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
        vector = (factor_dict["order_quantity"],)
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
        return x[0] > 0

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
        # Generate an Exponential(rate = 1) r.v.
        x = (rand_sol_rng.expovariate(1),)
        return x
