"""
Summary
-------
Simulate a day's worth of sales for a newsvendor.
A detailed description of the model/problem can be found `here
<https://simopt.readthedocs.io/en/latest/cntnv.html>`_.
"""
import numpy as np
import pulp
import scipy.stats as stats
import sympy as sp
from scipy.integrate import quad
from ..base import Model, Problem


class CntNV(Model):
    """
    A model that simulates a day's worth of sales for a newsvendor
    with a exponential demand distribution. Returns the profit, after
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
        self.name = "CNTNV"
        self.n_rngs = 1
        self.n_responses = 1
        self.factors = fixed_factors
        self.specifications = {
            "num_material": {
                "description": "number of material types",
                "datatype": int,
                "default": 4
            },
            "num_product": {
                "description": "number of product types",
                "datatype": int,
                "default": 3
            },
            "mat_to_prod": {
                "description": "PxR matrix, mapping M material to P products",
                "datatype": list,
                "default": [[1, 2, 1, 3],
                            [1, 1, 3, 1],
                            [2, 0, 4, 1]]
            },
            "material_cost": {
                "description": "purchasing cost per material unit",
                "datatype": list,
                "default": list(1 * np.ones(4))
            },
            "recourse_cost": {
                "description": "recourse purchasing cost per material unit",
                "datatype": list,
                "default": list(2 * np.ones(4))
            },
            "process_cost": {
                "description": "processing cost per product",
                "datatype": list,
                "default": list(0.1 * np.ones(3))
            },
            "order_cost": {
                "description": "fixed one-time ordering cost",
                "datatype": int,
                "default": 20
            },
            "purchase_yield": {
                "description": "yield rate of purchased materials",
                "datatype": list,
                "default": list(0.9 * np.ones(4))
            },
            "total_budget": {
                "description": "total budget for newsvendor's operation",
                "datatype": int,
                "default": 600
            },
            "sales_price": {
                "description": "sales price per product unit",
                "datatype": list,
                "default": list(12 * np.ones(3))
            },
            "salvage_price": {
                "description": "salvage price per material unit",
                "datatype": list,
                "default": list(0.6 * np.ones(4))
            },
            "order_quantity": {
                "description": "initial order quantity per material",
                "datatype": list,
                "default": list(20 * np.ones(4))
            },
            "poi_mean": {
                "description": "parameter for poisson demand distribution for each time break-point",
                "datatype": list(),
                "default": list([15 * np.ones(3)])
            },
            "t_intervals":{
                "description": "time break-points for change in demand",
                "datatype": list,
                "default": list(np.zeros(1))
            },
            "rank_corr":{
                "description": "rank_correlation between demands",
                "datatype": list(),
                "default": list([np.zeros(3),]*3)
            }
            }
        self.check_factor_list = {
            "num_material": self.check_num_material,
            "num_product": self.check_num_product,
            "mat_to_prod": self.check_mat_to_prod,
            "material_cost": self.check_material_cost,
            "process_cost": self.check_process_cost,
            "order_cost": self.check_order_cost,
            "recourse_cost": self.check_recourse_cost,
            "purchase_yield": self.check_purchase_yield,
            "total_budget": self.check_total_budget,
            "sales_price": self.check_sales_price,
            "salvage_price": self.check_salvage_price,
            "order_quantity": self.check_order_quantity,
            "poi_mean": self.check_poi_mean
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_num_material(self):
        return (self.factors["num_material"] > 0)

    def check_num_product(self):
        return (self.factors["num_product"] > 0)

    def check_mat_to_prod(self):
        return (np.shape(self.factors["mat_to_prod"])[0] ==
                self.factors["num_product"]) \
            and (np.shape(self.factors["mat_to_prod"])[1] ==
                 self.factors["num_material"])

    def check_material_cost(self):
        return all(np.array(self.factors["material_cost"]) > 0) \
            and (len(self.factors["material_cost"]) ==
                 self.factors["num_material"])

    def check_recourse_cost(self):
        if self.factors["recourse_cost"] is not None:
            return all(np.array(self.factors["recourse_cost"]) > 0) \
                and (len(self.factors["recourse_cost"]) ==
                     self.factors["num_material"])
        else:
            return True

    def check_process_cost(self):
        return all(np.array(self.factors["process_cost"]) >= 0) \
            and (len(self.factors["process_cost"]) ==
                 self.factors["num_product"])

    def check_order_cost(self):
        return (self.factors["order_cost"] >= 0)

    def check_purchase_yield(self):
        return all(np.array(self.factors["purchase_yield"]) <= 1) \
            and (len(self.factors["purchase_yield"]) ==
                 self.factors["num_material"])

    def check_total_budget(self):
        return (np.dot(self.factors["order_quantity"], self.factors["material_cost"])
                <= self.factors["total_budget"])

    def check_sales_price(self):
        return all(np.array(self.factors["sales_price"]) > 0) \
            and (len(self.factors["sales_price"]) ==
                 self.factors["num_product"])

    def check_salvage_price(self):
        return all(np.array(self.factors["salvage_price"]) >= 0) \
            and (len(self.factors["salvage_price"]) ==
                 self.factors["num_material"])

    def check_order_quantity(self):
        return True
        # return all(np.array(self.factors["order_quantity"]) > 0) \
        #     and (len(self.factors["order_quantity"]) ==
        #          self.factors["num_material"]) \
        #     and (np.issubdtype(self.factors["order_quantity"].dtype, np.integer))

    def check_poi_mean(self):
        return True
        # return all(np.array(self.factors["poi_mean"]) > 0) \
        #     and (len(self.factors["poi_mean"]) ==
        #          self.factors["num_product"]) \
        #     and (np.issubdtype(self.factors["poi_mean"].dtype, np.integer))

    def check_simulatable_factors(self):
        return all((self.factors["sales_price"] -
                    np.dot(self.factors["mat_to_prod"],
                           self.factors["material_cost"])) > 0)

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
        demand = []
        # new code starts here
        # if correlation between demands, find C from rank correlation
        if not all(all(x == 0 for x in inner) for inner in self.factors["rank_corr"]):
            # find roe star from rank correlation 
            corr = []
            for row in self.factors["rank_corr"]:
                corr_row = [2*np.sin((np.pi*rank)/6) for rank in row]
                corr.append(corr_row)
            # generate dependent multivariate normals
            C = np.linalg.cholesky(corr)
        
        # find spp breakpoints for each t interval
        # assume constants for now
        if self.factors["t_intervals"] != [0]:
            all_spp_breakpoints = []
            t_lb = 0
            t = sp.symbols("t")
            for i, t_interval in enumerate(self.factors["t_intervals"]):
                t_ub = t_interval
                add_const = [0,]*self.factors["num_product"]
                rate_functions= [sp.lambdify(t,sp.sympify(rate), modules=["math"]) for rate in self.factors["poi_mean"][i]]
                max_spp = [quad(rate,t_lb,t_ub)[0]+add_const[j] for j, rate in enumerate(rate_functions) ]
                #max_spp = [rate*(t_ub-t_lb) + add_const[j] for j, rate in enumerate(self.factors["poi_mean"][i])]
                add_const = max_spp
                all_spp_breakpoints.append(max_spp)
                t_lb = t_ub
            # determine spp at end of 24 hours (stopping condition)
            rate_functions = [sp.lambdify(t,sp.sympify(rate), modules=["math"]) for rate in self.factors["poi_mean"][-1]]
            stop = [quad(rate,t_lb,t_ub)[0]+add_const[j] for j, rate in enumerate(rate_functions)]
            all_spp_breakpoints.append(stop)
        else: # rate function does not change over t
            t = sp.symbols("t")
            rate_functions= [sp.lambdify(t,sp.sympify(rate), modules=["math"]) for rate in self.factors["poi_mean"][-1]]
            stop = [quad(rate,0,24)[0] for rate in rate_functions ]

            
        
        # generate demand for constant arrival rate
        demand = [0,]*self.factors["num_product"]
        s = [0,]*self.factors["num_product"]
        if self.factors["t_intervals"] == [0]: # constant rate for poisson process
            while any(s_i < stop[i] for i,s_i in enumerate(s)):
                # generate uniforms 
                # if correlation between demands, generate dependent uniforms
                if not all(all(x == 0 for x in inner) for inner in self.factors["rank_corr"]):
                    z = [demand_rng.normalvariate(),]*self.factors["num_product"]
                    z_corr = np.dot(C, np.array(z))
                    u = stats.norm.cdf(z_corr)
                else: # no correlation between demands, return uncorrelated uniforms
                    u = [demand_rng.random(),]*self.factors["num_product"]
                # inverse rate
                for i, u_i in enumerate(u):
                    s[i] -= np.log(u_i)
                # update demand numbers
                for i, s_i in enumerate(s):
                    if s_i < stop[i]:
                        demand[i]+=1 
        else: # non-constant demand rates
            while any(s_i < stop[i] for i,s_i in enumerate(s)):
                # generate uniforms 
                # if correlation between demands, generate dependent uniforms
                if not all(all(x == 0 for x in inner) for inner in self.factors["rank_corr"]):
                    z = [demand_rng.normalvariate(),]*self.factors["num_product"]
                    z_corr = np.dot(C, np.array(z))
                    u = stats.norm.cdf(z_corr)
                else: # no correlation between demands, return uncorrelated uniforms
                    u = demand_rng.random(self.factors["num_product"])
                for i,u_i in enumerate(u):
                    s[i] -= np.log(u_i)
                    for j, breakpoint_rates in enumerate(self.factors["poi_mean"]):
                        max_spp = all_spp_breakpoints[j][i]
                        if s[i] <= max_spp:
                           demand[i]+=1

        stock_material = self.factors['order_quantity'] # use this for continuous 

        # # Generate binomial r.v for material levels based on yield rates
        # stock_material = [sum([1 for i in range(int(n))
        #                        if demand_rng.random() < p])
        #                   for p, n in zip(self.factors["purchase_yield"],
        #                                   self.factors["order_quantity"])]
        # If recourse cost is not specified, use an array of very large numbers
        # This heuristically makes it impossible to buy any recourse material
        if self.factors["recourse_cost"] is None:
            self.factors["recourse_cost"] = [element * 10000 for
                                             element in self.factors["material_cost"]]
        # define IP problem for production of goods to fulfill realized demand
        """
        X: amount of products produced for each product type
        Y: recourse quantity for each material type
        Objective: maximize profit => sum(sales) - sum(process cost) - sum(recourse cost)
        """
        X = pulp.LpVariable.dicts("X", range(self.factors["num_product"]),
                                  lowBound=0, cat='Continuous')
        Y = pulp.LpVariable.dicts("Y", range(self.factors["num_material"]),
                                  lowBound=0, cat='Continuous')
        prob = pulp.LpProblem("Continuous Problem", pulp.LpMaximize)
        prob += pulp.lpSum(self.factors["sales_price"][i] * X[i]
                           for i in range(self.factors["num_product"])) - \
            pulp.lpSum([X[i] * self.factors["process_cost"][i]
                        for i in range(self.factors["num_product"])]) - \
            pulp.lpSum([Y[i] * self.factors["recourse_cost"][i]
                        for i in range(self.factors["num_material"])])
        # Define constraints
        """
        C1: sum(recourse cost) + sum(process cost) <= total budget - sum(material cost)
        C2: for each material, amount used <= (initial material + recourse quantity)
        """

        prob += (pulp.lpSum([Y[i]*self.factors["recourse_cost"][i]
                            for i in range(self.factors["num_material"])]) +
                 pulp.lpSum([X[i]*self.factors["process_cost"][i]
                            for i in range(self.factors["num_product"])])) <= \
            self.factors["total_budget"] - np.dot(self.factors["order_quantity"],
                                                  self.factors["material_cost"])
        for j in range(self.factors["num_material"]):
            prob += pulp.lpSum([self.factors["mat_to_prod"][i][j] * X[i]
                                for i in range(self.factors["num_product"])]) <= \
                                    stock_material[j] + Y[j]
        for i in range(self.factors["num_product"]):
            prob += X[i] <= demand[i]
        prob.solve(pulp.apis.PULP_CBC_CMD(msg=False))

        # Results of IP
        Finish_Goods = np.array([pulp.value(X[i])
                                 for i in range(self.factors["num_product"])])
        Recourse = np.array([pulp.value(Y[i])
                             for i in range(self.factors["num_material"])])
        Inventory = np.array(self.factors["order_quantity"]) + Recourse - \
            np.dot(np.linalg.pinv(self.factors["mat_to_prod"]), Finish_Goods)

        # Calculate profit.
        total_cost = (np.dot(self.factors["order_quantity"], self.factors["material_cost"]) +
                      np.dot(Recourse, self.factors["recourse_cost"]) +
                      np.dot(Finish_Goods, self.factors["process_cost"]) +
                      self.factors["order_cost"])
        sales_revenue = np.dot([min(s, d) for s, d in zip(Finish_Goods, demand)],
                               self.factors["sales_price"])
        salvage_revenue = np.dot(Inventory, self.factors["salvage_price"])
        profit = sales_revenue + salvage_revenue - total_cost

        stockout_qty = [max(d - s, 0) for d, s in zip(demand, Finish_Goods)]
        stockout = [1 if a > 0 else 0 for a in stockout_qty]
        # Compose responses and gradients.
        responses = {"profit": profit, "stockout_qty": stockout_qty, "stockout": stockout}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications}
                     for response_key in responses}
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
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"order_quantity"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": [40, 40, 100, 60]
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 3000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = CntNV(self.model_fixed_factors)
        self.dim = self.model.factors["num_material"]
        self.lower_bounds = (0,)*self.dim
        self.upper_bounds = (np.inf,)*self.dim

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
            "order_quantity": vector[:]
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
        vector = tuple(factor_dict["order_quantity"])
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
        return super().check_deterministic_constraints(x) and (
            np.dot(x, self.model.factors["material_cost"]) <
            self.model.factors["total_budget"])

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
        # Generate an r.v. vector.
        # The result is rounded down to get discrete values.
        x = (rand_sol_rng.continuous_random_vector_from_simplex(
            n_elements=self.model.factors["num_material"],
            summation=self.model.factors["total_budget"],
            exact_sum=False))
            #weights=self.model.factors["material_cost"]))
            
        x = [int(np.floor(i)) for i in x]
        return x
