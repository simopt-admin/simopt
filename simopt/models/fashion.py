"""
Summary
-------
Simulate multiple periods worth of sales for a fashion retailer.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/fashion.html>`_.
"""
import numpy as np

from ..base import Model, Problem


class Fashion(Model):
    """

    A model that simulates a redorder quantity dyring a sales season for an inventory
    replenishment problem with current inventory position, backlogged demand,
    forecasted future demand, customer's willingness to backorder, and returns taken
    into account. Returns an optimal time t and an inventory reorder quantity Q2.

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
    fixed_factors : dict (anything that is not a decision variable, not a random variable)
        fixed_factors of the simulation model

        ``leadtime``
            the time between the ordering and recieving of products (11) (int)

        ``product_price``
            fixed price of products, does NOT change (int)

        ``n_weeks``
            number of weeks to simulate (int)

        ``second_order_time``
            initial periods before the actual simulation starts, t

         ``initial_order_quantity``
            initial order quantity, Q1

         ``backorder_d``
            backorder function constant d

         ``backorder_B``
            backorder function B

        just include in code somewhere: ``return_limit``
            the time given to return merchandise (after 1 week, items considered as sold) (int)

        add: probability that a customer is willing to wait for a backorder, factors in demand and return model
            probability to customer returns their product (bino?), salvage price for leftover stock (muda)

    See also
    --------
    base.Model
    """

    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "Fashion"
        self.n_rngs = 3  # change later (3 seperate random number generators, returns, customer willingness to wait, demand)
        self.n_responses = 7  # ???? (random variable outputs: profit(+ or -), cost and revenue, unmet demand, orders placed after q2 placed)
        self.factors = fixed_factors
        self.specifications = {
            "leadtime": {
                "description": " the time between the ordering and recieving of products (t+11)",
                "datatype": int,
                "default": 11
            },
            "product_price": {
                "description": "fixed price of products, does NOT change",
                "datatype": float,
                "default": 20.0  # 20 dolllars a piece?
            },
            "n_weeks": {
                "description": "number of weeks to simulate",
                "datatype": int,
                "default": 22
            },
            "initial_order_quantity": {  # q1
                "description": "initial order quantity",
                "datatype": int,
                "default": 1000
            },
            "second_order_time": {  # t
                "description": "time when the second order is placed",
                "datatype": int,
                "default": 7
            },
            "return_chance": {
                "description": "chance of customer returns",
                "datatype": float,
                "default": 0.35
            },
            "return_refund": {
                "description": "how much the customer gets refuneded",
                "datatype": float,
                "default": 0.5
            },
            "backorder_d": {
                "description": "backorder constant d",
                "datatype": int,
                "default": 3
            },
            "backorder_B": {
                "description": "backorder constant B",
                "datatype": float,
                "default": 1.5
            },

        }

        self.check_factor_list = {  # functions will check if the provided factors meet certain conditions.
            "leadtime": self.check_leadtime,
            "product_price": self.check_product_price,
            "n_weeks": self.check_n_weeks,
            "initial_order_quantity": self.check_initial_order_quantity,
            "second_order_time": self.check_second_order_time,
            "return_chance": self.check_return_chance,
            "return_refund": self.check_return_refund,
            "backorder_d": self.check_backorder_d,
            "backorder_B": self.check_backorder_B,

        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)  # initialize the object with the given factors.

    # Check for simulatable factors
    def check_leadtime(self):
        return self.factors["leadtime"] > 0

    def check_product_price(self):
        return self.factors["product_price"] > 0  # should put a set defualt price?

    def check_n_weeks(self):
        return self.factors["n_weeks"] > 1  # start at 1 week

    def check_initial_order_quantity(self):
        return self.factors["initial_order_quantity"] > 0

    def check_second_order_time(self):
        return self.factors["second_order_time"] > 0

    def check_return_chance(self):
        return 0 < self.factors["return_chance"] < 1

    def check_return_refund(self):
        return self.factors["return_refund"] < self.factors["product_price"]

    def check_backorder_d(self):
        return self.factors["backorder_d"] > 0

    def check_backorder_B(self):
        return self.factors["backorder_B"] > 0

    def replicate(self, rng_list):

        '''

        Simulate a single sales season for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [mrg32k3a.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest


            ``Profit``
               amount of profit generated during the sales season
            ``Potential Lost Sales``
                the number of lost sales incurred from stockouts
            ``Optimal Q2``
                The best q2 to maximize profit
            ``salvage cost``
               amount of money needed in order for leftovers to be disposed of in landfill

'''
        # Designate random number generators. have 3
        demand_rng = rng_list[0]
        return_rng = rng_list[1]  # binomial, with n and p
        backorder_rng = rng_list[2]

    # Generate demand for simulation:
        import numpy as np

        def generate_weekly_demand(mean_demand, autoregressive_coefficient, error_std, n_weeks):
            demand = np.zeros(n_weeks)
            demand[0] = 100  # Initial demand for Week 0
            for t in range(1, n_weeks):
                demand[t] = mean_demand + autoregressive_coefficient * (demand[t - 1] - mean_demand) + demand_rng.binomialvariate(0, error_std)
            return demand

        mean_demand = 10
        autoregressive_coefficient = 0.3
        error_std = 10
        n_weeks = 22

        weekly_demand = generate_weekly_demand(mean_demand, autoregressive_coefficient, error_std, n_weeks)
        # print("Weekly Demand:", weekly_demand)
        # Initialize starting and ending inventories for each period.
        demands = weekly_demand

        def calculate_backorders(B, t, d):
            B = self.factors["backorder_B"]
            d = self.factors["backorder_d"]
            t = self.factors["second_order_time"]
            exponent = -(t + self.factors["leadtime"] - d)
            backorders = B * np.exp(exponent)
            return backorders

        # Run simulation over time horizon.
        n_weeks = self.factors["n_weeks"]

        # initialize
        reorder_quantity = 4000  # Q2
        # return_chance = .35 #  chance that a customer will return, in the dict.

        # Initialize
        start_inv = np.zeros(n_weeks, dtype=int)  # Inventory at the start of each week as integers

        end_inv = np.zeros(n_weeks, dtype=int)     # Inventory at the end of each week
        orders_received = np.zeros(n_weeks)  # Orders received at the beginning of each week
        backorders = np.zeros(n_weeks, dtype=int)  # Track backorders
        returns = np.zeros(n_weeks)
        lost_sales = np.zeros(n_weeks)

        start_inv[0] = self.factors["initial_order_quantity"]
        total_cost = start_inv[0] * self.factors["product_price"]
        revenue_per_unit = self.factors["product_price"] / 4
        total_revenue = 0

        for week in range(n_weeks):
            if week > 0:
                #  Handle returns: calculate based on previous week's demand and return chance
                returns[week] = return_rng.binomialvariate(n=int(demands[week - 1]), p=self.factors["return_chance"])
                start_inv[week] = start_inv[week - 1] + returns[week] - demands[week - 1] + orders_received[week]

            #  Calculate projected inventory for the week including future restocks
            projected_inventory = start_inv[week] + np.sum(orders_received[week:])

            #  Fulfill demand and calculate lost sales or backorders
            if projected_inventory >= demands[week]:
                end_inv[week] = start_inv[week] - demands[week]  # Calculate the ending inventory after meeting the demand
                if end_inv[week] < 0:
                    # If the actual starting inventory wasn't enough to meet the demand
                    unsatisfied_demand = -end_inv[week]  # Calculate how much demand was not satisfied
                    backorder_prob = calculate_backorders(1.5, self.factors["second_order_time"], 3)  # using binomial and the function
                    backorders[week] = backorder_rng.binomialvariate(n=unsatisfied_demand, p=backorder_prob)
                    end_inv[week] = 0  # Set end inventory to zero since all available inventory has been used up
            else:
                #  When the projected inventory is less than the week's demand
                lost_sales[week] = demands[week] - start_inv[week]  # number of sales lost due to insufficient inventory
                end_inv[week] = 0

            total_revenue += (weekly_demand[week] - lost_sales[week]) * revenue_per_unit - returns[week] * self.factors["product_price"] * self.factors["return_refund"]

            #  Place reorder if it is the specified second order time
            if week == self.factors["second_order_time"]:
                orders_received[week + self.factors["leadtime"]] = reorder_quantity
                total_cost += reorder_quantity * self.factors["return_refund"]

            # Fulfill backorders if there is inventory available
            if week > 0 and end_inv[week] > 0 and backorders[week - 1] > 0:
                available_for_backorders = min(end_inv[week], backorders[week - 1])
                end_inv[week] -= available_for_backorders  # Reduce the ending inventory of the current week by the amount that will be used to fulfill some of the backorders.
                backorders[week - 1] -= available_for_backorders

        profit = total_revenue - total_cost

        # final summary
        responses = {
            "final_end_inventory: ": end_inv[-1],
            "backorders: ": backorders[-1],
            "lost_sales": np.sum(lost_sales),
            "profit": profit

        }

        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


class Fashionopt(Problem):

    def __init__(self, name="FASHION-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.dim = 2
        self.n_objectives = 1  # maxmimizing profit
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "discrete"  # because q1 and t

        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}  # empty because our defaults are the same

        self.model_decision_factors = {"intitial_order_quantity", "second_order_time"}  # q1 snd t?
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution from which solvers start",
                "datatype": tuple,
                "default": (100, 9)
            },
            "budget": {  # how many sales season the algorithm gets to find an optimal q2
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
        #  Instantiate model with fixed factors and overwritten defaults.
        self.model = Fashion(self.model_fixed_factors)

        self.lower_bounds = (0, 0)  # 0 for q1 and 0 for t
        self.upper_bounds = (np.inf, self.model.factors["n_weeks"] - self.model.factors["leadtime"])  # 11 for t

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
        factor_dict = {  # so q1 is the first value, and t is the second one
            "initial_order_quanitity": vector[0],
            "second_order_time": vector[1]
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
        vector = (factor_dict["initial_order_quantity"], factor_dict["second_order_time"])
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
        objectives = (response_dict["profit"],)  # our objective is to maximize profit
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

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
        return True

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """

        x = ((np.max(np.round(rand_sol_rng.normalvariate(mu=1000, sigma=np.sqrt(200)))), 0), rand_sol_rng.randint(1, self.upper_bounds[1]))  # will generate random numbers for q1 and t
        return x
