"""
Summary
-------
Simulate demand at facilities.
"""

from tkinter import END

import numpy as np

from base import Model, Problem
import math as math


class Voting(Model):

    """
    A model that simulates a day of voting operations in multiple precincts

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
        self.n_rngs = 5
        self.n_responses = 1
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
                "default": [.4, .2, .6, .3, .1]
            },
            "turn_ran": {
                "description": "turnout range specific to precinct i",
                "datatype": list,
                "default": [10, 15, 10, 50, 30]
            },
            "reg_vote": {
                "description": "number of registered voters in precinct i",
                "datatype": list,
                "default": [100, 200, 100, 400, 200]
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
                "description": "voting machines are repaired according to a gamma distribution, this is the mean time, minutes",
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

    def check_mach_allocation(self):  # Making sure that all machines are allocated and equal to max available
        for i in range(len(self.factors["mach_allocation"])):
            if self.factors["mach_allocation"][i] < 0:
                return False
        return sum(self.factors["mach_allocation"]) == self.factors["n_mach"]

    def check_n_mach(self):  # verifying that the machines are positive values
        return self.factors["n_mach"] > 0

    def check_n_prec(self):  # verifying that precinct number is positive
        return self.factors["n_prec"] > 0

    def check_mid_turn_per(self):  # veifying that all are percentages
        for i in self.factors["mid_turn_per"]:
            if i < 0 or i > 1:
                return False
        return True

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

    def check_simulatable_factors(self):  # all lists have indeces for number of precincts
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

    def check_bd_prob(self):  # veifying breakdown is a probability
        if self.factors["bd_prob"] < 0 or self.factors["bd_prob"] > 1:
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

            "mach_bd" = binary variable, probability that the
                0 : The voting machine is broken down at start of day
                1 : The voting machine does not break down for the day

            "repair_time" = the time that it will take for a machine to be repaired, gamma distributed
            "arrival_rate" = rate of arrival to the voting location

        gradients : dict of dicts
            gradient estimates for each response
        """

        breakdown_rng = rng_list[0]
        turnout_rng = rng_list[1]
        arrival_rng = rng_list[2]
        voting_rng = rng_list[3]
        choices_rng = rng_list[4]
        prec_avg_waittime = []
        perc_no_waittime = []

        for m in range(self.factors["n_prec"]):  # p is num of machines in that precinct
            mach_list = []
            for i in range(len(self.factors["mach_allocation"])):  # i is each individual machine in that precinct
                p = self.factors["bd_prob"]  # Default is .05
                if choices_rng.choices([0, 1], [1 - p, p]) == 1:  # Determining if the machine will be borken down to start day
                    t = breakdown_rng.gammavariate((self.factors["mean_repair"] ^ 2) / (self.factors["stdev_repair"] ^ 2), (self.factors["stdev_repair"] ^ 2) / (self.factors["mean_repair"]))  # Determines wait time for broken machine in minutes
                else:
                    t = math.inf
                mach_list.append(t)

            t_i = self.factors["mid_turn_per"][m] + abs(self.factors["turn_ran"][m] * turnout_rng.triangular(-1, 1, 0))  # ask Dr. Eckman about this!!

            p_lamda = (self.factors["reg_vote"][m] * t_i) / self.factors["hours"]

            arr_times = []
            t = arrival_rng.expovariate(p_lamda)  # initial arrival
            print(p_lamda)
            while t <= (self.factors["hours"] * 60):
                arr_times.append(t)  # appends before so that the last arrival in list will be before voting closes
                t += arrival_rng.expovariate(p_lamda)  # list is time at which each person arrives
            voting_times = []
            for p in range(self.factors["n_prec"]):
                for i in range(len(arr_times)):
                    voting_times.append(voting_rng.gammavariate((self.factors["mean_time2vote"] ** 2) / (self.factors["stdev_time2vote"] ** 2), (self.factors["stdev_time2vote"] ** 2) / (self.factors["mean_time2vote"])))
            queue = []
            wait_times = []
            clock = 0
            vote_ind = 0
            arr_ind = 0
            mach_ind = 0
            print("before while loop")
            while len(wait_times) <= len(arr_times):
                if min(mach_list) <= arr_times[arr_ind]:
                    clock = min(mach_list)
                    if len(queue) > 0:  # logic works here since the only next event can be an arrival as if mahcines finish there are no entities to enter them
                        clock = arr_times[arr_ind]  # updates since we are also moving to the next event here to
                        mach_ind = mach_list.index(min(mach_list))
                        mach_list[mach_ind] = clock + voting_times[vote_ind]
                        vote_ind += 1
                        arr_ind += 1
                        wait_times.append(clock - queue.pop(0))
                    elif len(queue) == 0:
                        mach_ind = mach_list.index(min(mach_list))
                        mach_list[mach_ind] = math.inf
                    else:
                        print("error in replicate simulation loop 1")
                        END
                elif arr_times[arr_ind] < min(mach_list):
                    clock = arr_times[arr_ind]
                    if len(queue) == 0:
                        for i in range(len(mach_list)):
                            if mach_list[i] == math.inf:
                                mach_ind = i
                                break
                            elif mach_list[i] != math.inf:
                                mach_ind = -1
                        if mach_ind >= 0:
                            mach_list[mach_ind] = clock + voting_times[vote_ind]
                            wait_times.append(0)
                            vote_ind += 1
                            arr_ind += 1
                        elif mach_ind == -1:  # no infinity values in list
                            queue.append(clock)
                        else:
                            print("error in loop queue is empty arrival times less than machine list")
                            END
                    elif len(queue) > 0:
                        queue.append(clock)
                    else:
                        print("error in simulation loop 1, arrival times less than machine list")
                        END

                else:
                    print('error in replicate simulation loop 2')
                    END
        prec_avg_waittime.append.mean(wait_times)
        perc_no_waittime.append(wait_times.count(0) / len(wait_times))

        responses = {
            'avg_wait_time': prec_avg_waittime,
            'perc_no_waittime': perc_no_waittime
        }

        return responses
"""
Summary
-------
Minimize the (deterministic) total cost of installing capacity at
facilities subject to a chance constraint on stockout probability.
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
    def __init__(self, name="voting", fixed_factors={}, model_fixed_factors={}):
        self.name = name  # refer to the model factor of number of precincts, move below the initialization of the models   #self.model.factors["n_prec"]
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.minmax = (-1,)
        self.constraint_type = "deterministic"
        self.variable_type = "discrete"
        self.upper_bounds = (math.inf, math.inf, math.inf, math.inf, math.inf)
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
                "default": 10000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

        super().__init__(fixed_factors, model_fixed_factors)
        self.model = MinVotingMaxWaitTime(self.model_fixed_factors)
        self.dim = self.model.factors["n_prec"]
        self.lower_bounds = ()
        for i in range(self.dim):  # can we do this??
            self.upper_bounds.append(math.inf)
            self.lower_bounds.append(1)

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

        objectives = (max(response_dict["avg_wait_time"]), )  # need to take the max average waiting time, in a tuple with a comma at the end.  = np.max(response_dict[avg_waitingtime])
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
        stoch_constraints = None  # can set to none
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
        det_stoch_constraints = None  # can set to none
        det_stoch_constraints_gradients = None
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
        det_objectives = None  # can set to none, if there was a cost penalty then this could be use
        det_objectives_gradients = None
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
        return np.all(sum(x) >= self.model.factors["n_machines"])  # self.model.factors["n_machines"] >= sum(x)

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
        x = tuple([300 * rand_sol_rng.random() for _ in range(self.dim)])  # natalia will have the code for this, a little more tricky
        return x
