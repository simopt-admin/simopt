"""
Summary
-------
Simulate a day of voting operations in multiple precincts.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/voting.html>`_.
"""

import numpy as np
import math as math

from base import Model, Problem


class Voting(Model):

    """
    A model that simulates a day of voting operations in multiple precincts.

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
    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "VOTING"
        self.n_rngs = 5
        self.n_responses = 2
        self.specifications = {
            "mach_allocation": {
                "description": "number of machines allocation for precinct i",
                "datatype": list,
                "default": [10, 10, 10, 10, 10]
            },
            "n_mach": {
                "description": "max number of machines available",
                "datatype": int,
                "default": 50
            },
            "mid_turn_per": {
                "description": "midpoint turnout percentage for precinct i",
                "datatype": list,
                "default": [10, 15, 10, 20, 30]
            },
            "turn_ran": {
                "description": "turnout range specific to precinct i",
                "datatype": list,
                "default": [.4, .2, .3, .3, .1]
            },
            "reg_vote": {
                "description": "number of registered voters in precinct i",
                "datatype": list,
                "default": [100, 200, 100, 200, 200]
            },
            "mean_time2vote": {
                "description": "the mean time for the gamma distributed time taken to vote",
                "datatype": int,
                "default": 7.5
            },
            "stdev_time2vote": {
                "description": "the standard deviation for the gamma distributed time to vote",
                "datatype": int,
                "default": 2
            },
            "mean_repair": {
                "description": "mean for gamma distribution for time to repair a machine, minutes",
                "datatype": int,
                "default": 60
            },
            "stdev_repair": {
                "description": "standard deviation for gamma distribution for time to repair a machine, minutes",
                "datatype": int,
                "default": 20
            },
            "bd_prob": {
                "description": "probability at which the voting machines break down (bd)",
                "datatype": float,
                "default": .05
            },
            "hours": {
                "description": "number of hours open to vote",
                "datatype": float,
                "default": 13.0
            },
            "n_prec": {
                "description": "number of precincts",
                "datatype": int,
                "default": 5
            }
        }
        self.check_factor_list = {
            "mach_allocation": self.check_mach_allocation,
            "n_mach": self.check_n_mach,
            "mid_turn_per": self.check_mid_turn_per,
            "turn_ran": self.check_turn_ran,
            "reg_vote": self.check_reg_vote,
            "mean_time2vote": self.check_mean_time2vote,
            "stdev_time2vote": self.check_stdev_time2vote,
            "mean_repair": self.check_mean_repair,
            "stdev_repair": self.check_stdev_repair,
            "bd_prob": self.check_bd_prob,
            "hours": self.check_hours,
            "n_prec": self.check_n_prec
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_turn_ran(self):
        return len(self.factors["turn_ran"]) > 0

    def check_reg_vote(self):
        return len(self.factors["reg_vote"]) > 0

    def check_mach_allocation(self):
        # Make sure that all available machines are allocated.
        return all([n_machine_alloc > 0 for n_machine_alloc in self.factors["mach_allocation"]]) and sum(self.factors["mach_allocation"]) == self.factors["n_mach"]

    def check_n_mach(self):
        return self.factors["n_mach"] > 0

    def check_n_prec(self):
        return self.factors["n_prec"] > 0

    def check_mid_turn_per(self):
        # Make sure that min/max of triangular distributions are between zero
        # and one, corresponding to percentages.
        all_min_greater_than_zero = all([self.factors["mid_turn_per"][i] - self.factors["turn_ran"][i] >= 0 for i in range(self.factors["n_prec"])])
        all_max_less_than_one = all([self.factors["mid_turn_per"][i] + self.factors["turn_ran"][i] <= 1 for i in range(self.factors["n_prec"])])
        return all_min_greater_than_zero and all_max_less_than_one

    def check_mean_time2vote(self):
        return self.factors["mean_time2vote"] > 0

    def check_stdev_time2vote(self):
        return self.factors["stdev_time2vote"] > 0

    def check_mean_repair(self):
        return self.factors["mean_repair"] > 0

    def check_stdev_repair(self):
        return self.factors["stdev_repair"] > 0

    def check_hours(self):
        return self.factors["hours"] > 0

    def check_simulatable_factors(self):
        # Make sure that all lists have the correct length.
        if len(self.factors["turn_ran"]) != self.factors["n_prec"]:
            return False
        elif len(self.factors["reg_vote"]) != self.factors["n_prec"]:
            return False
        elif len(self.factors["mid_turn_per"]) != self.factors["n_prec"]:
            return False
        elif len(self.factors["mach_allocation"]) != self.factors["n_prec"]:
            return False
        else:
            return True

    def check_bd_prob(self):
        return 0 <= self.factors["bd_prob"] < 1

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
            "perc_avg_waittime": list of all precint waittimes
            "perc_no_waittime": percentage of voters that did not have to wait at the location
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        breakdown_rng = rng_list[0]
        turnout_rng = rng_list[1]
        arrival_rng = rng_list[2]
        voting_rng = rng_list[3]
        choices_rng = rng_list[4]

        # Generate variable dictating county-wide (base) turnout.
        turnout_T = turnout_rng.triangular(-1, 1, 0)

        # Simulate each precinct separately.
        for m in range(self.factors["n_prec"]):  # m is index of precinct.
            # Determining how many machines in the precint will be broken down at start of the day.
            # mach_list contains time at which each machine would finish service and become operational.
            mach_list = []
            for i in range(self.factors["mach_allocation"][m]):  # i is index of each machine in that precinct.
                p = self.factors["bd_prob"]
                if choices_rng.choices([0, 1], [1 - p, p]) == [1]:
                    # Machine is broken; schedule the repair time.
                    t = breakdown_rng.gammavariate((self.factors["mean_repair"] ^ 2) / (self.factors["stdev_repair"] ^ 2), (self.factors["stdev_repair"] ^ 2) / (self.factors["mean_repair"]))  # Determines wait time for broken machine in minutes
                else:
                    # Machine is available at start of the day.
                    t = math.inf
                mach_list.append(t)

            # Calculate precinct level turnout and arrival rate based on voter rolls.
            t_i = self.factors["mid_turn_per"][m] + self.factors["turn_ran"][m] * turnout_T
            p_lamda = (self.factors["reg_vote"][m] * t_i) / (self.factors["hours"])

            # Generate all arrival times in advance.
            arr_times = []
            t = arrival_rng.expovariate(p_lamda)
            while t <= (self.factors["hours"] * 60):
                arr_times.append(t)  # Appends before so that the last arrival in list will be before voting closes.
                t += arrival_rng.expovariate(p_lamda)

            # Generate all voting times in advance.
            voting_times = []
            for i in range(len(arr_times)):
                voting_times.append(voting_rng.gammavariate((self.factors["mean_time2vote"] ** 2) / (self.factors["stdev_time2vote"] ** 2), (self.factors["stdev_time2vote"] ** 2) / (self.factors["mean_time2vote"])))

            # Initialize statistics.
            prec_avg_waittime = []
            perc_no_waittime = []

            # Initialize state variables for simulation.
            queue = []  # Contains arrival times of voters in the queue.
            wait_times = []
            clock = 0
            vote_ind = 0
            arr_ind = 0
            mach_ind = 0

            # Simulate a day at the precinct.
            while arr_ind < len(arr_times):

                if min(mach_list) <= arr_times[arr_ind]:  # Next event is that a machine becomes available.
                    clock = min(mach_list)
                    mach_ind = mach_list.index(min(mach_list))
                    if len(queue) > 0:  # If people in queue, take one out and put into a machine.
                        next_queue = queue.pop(0)
                        mach_list[mach_ind] = clock + voting_times[vote_ind]
                        vote_ind += 1
                        wait_times.append(clock - next_queue)
                    elif len(queue) == 0:  # If queue is empty, set the machine's next available time to infinity.
                        mach_list[mach_ind] = math.inf

                elif arr_times[arr_ind] < min(mach_list):  # Next event is that a voter arrives.
                    clock = arr_times[arr_ind]
                    if len(queue) == 0:  # If queue is empty, check to see if a machine is available.
                        for i in range(len(mach_list)):
                            if mach_list[i] == math.inf:
                                mach_ind = i
                                break
                            elif mach_list[i] != math.inf:
                                mach_ind = -1
                        if mach_ind >= 0:  # machine is open and place in machine
                            mach_list[mach_ind] = clock + voting_times[vote_ind]
                            wait_times.append(0)
                            vote_ind += 1
                        elif mach_ind == -1:  # No machines are available, so voter joins the queue.
                            queue.append(clock)
                    elif len(queue) > 0:
                        queue.append(clock)
                    arr_ind += 1

            # After all voters arriving before polls close have arrived,
            # simulate long enough to empty the polling station.
            while len(queue) > 0:
                clock = min(mach_list)
                mach_ind = mach_list.index(min(mach_list))
                next_queue = queue.pop(0)
                mach_list[mach_ind] = clock + voting_times[vote_ind]
                vote_ind += 1
                wait_times.append(clock - next_queue)

            # Calculate summary statistics for the precinct:
            #     average waiting time
            #     percentage of voters who did not wait
            prec_avg_waittime.append(sum(wait_times) / len(wait_times))
            perc_no_waittime.append(wait_times.count(0) / len(wait_times))
        # Compose responses and gradients.
        responses = {
            "prec_avg_waittime": prec_avg_waittime,
            "perc_no_waittime": perc_no_waittime
        }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Minimize the maximum average waiting time across all precincts.

"""


class MinVotingMaxWaitTime(Problem):
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
    def __init__(self, name="VOTING-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "deterministic"
        self.variable_type = "discrete"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"mach_allocation"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (10, 10, 10, 10, 10)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 1000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

        super().__init__(fixed_factors, model_fixed_factors)
        self.model = Voting(self.model_fixed_factors)
        self.dim = self.model.factors["n_prec"]
        self.lower_bounds = (1,) * self.model.factors["n_prec"]
        self.upper_bounds = (np.inf,) * self.model.factors["n_prec"]

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
            "mach_allocation": vector[:]
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
        vector = tuple(factor_dict["mach_allocation"])
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

        objectives = (max(response_dict["prec_avg_waittime"]), )
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
        det_objectives = (0,)  # can set to none, if there was a cost penalty then this could be use
        det_objectives_gradients = (0, ) * self.model.factors['n_prec']
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
        # Check machine allocation constraint.
        allocation_feasible = (np.sum(x) == self.model.factors["n_mach"])
        # Check box constraints.
        box_feasible = super().check_deterministic_constraints(x)
        return allocation_feasible * box_feasible

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
        x = rand_sol_rng.integer_random_vector_from_simplex(self,
                                                             n_elements=self.model.factors["n_prec"],
                                                             summation=self.model.factors["n_mach"],
                                                             with_zero=False
                                                             )
        return x
