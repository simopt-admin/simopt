"""
Summary
-------
Simulate matching of Valorant players on an online platform.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/chessmm.html>`_.
"""
from itertools import combinations
import numpy as np

from simopt.base import Model, Problem


class ValorantMatchmaking(Model):
    """
    A model that simulates a matchmaking problem with  a
    Elo (non-stationary triangular distribution) of players and Poisson arrivals.
    Returns the average difference between matched players.

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
        self.name = "VALORANT"
        self.n_rngs = 4
        self.n_responses = 2
        self.specifications = {
            "poisson_rate": {
                "description": "rate of Poisson process for player arrivals according to the time of day",
                "datatype": list,
                "default": [1 / 20, 1 / 15, 1 / 18, 1/ 20, 1/ 20, 1/ 23, 1/ 30, 1/ 25,
                            1/30, 1 / 20, 1 / 15, 1 / 18, 1/ 20, 1/ 20, 1/ 23, 1/ 30, 1/ 25,
                            1/30, 1 / 20, 1 / 15, 1 / 18, 1/ 20, 1/ 20, 1/ 23, 1/ 30, 1/ 25, 1/30]
            },
            "allowable_diff": {
                "description": "maximum allowable difference between Elo ratings",
                "datatype": float,
                "default": 200.0
            },
            "team_num": {
                "description": "number of players on a team",
                "datatype": int,
                "default": 5
            },
            "kd_mean": {
                "description": "The mean kill/death ratio for multivariate normal distribution of k/d, elo and win rate of players",
                "datatype": float,
                "default": 1.0
            },
            "wr_mean": {
                "description": "The mean win percentage for multivariate normal distribution of k/d, elo, and win rate of players",
                "datatype": float,
                "default": 50.0
            },
            "cov_matrix": {
                "description": "The covariance matrix, with the relationships between elo, kill/death ratio, and win rates.",
                "datatype": list,
                "default": [[40000, 28, 950], [28, .04, .8], [950, .8, 100]]
            },
            "elo_mean": {
                "description": "The mean of the elo of players in a multivariate normal distribution of k/d, elo, and win rate of players",
                "datatype": float,
                "default": 1300.0
            },
            "team_dif": {
                "description": "Allowable difference in average player rating between teams matched against each other",
                "datatype": float,
                "default": 400.0
            },
            "wait_thresh": {
                "description": "The amount of time a player waits before they get prioritized for matchmaking, in seconds",
                "datatype": float,
                "default": 1.05,
            },
            "slack": {
                "description": "If someone is waiting more than the wait threshold, the allowable difference increases by this amount",
                "datatype": float,
                "default": 200.0,
            },
            "kd_multiplier": {
                "description": "The factor kd is multiplied by to calibrate the rating of a player with elo and win rate.",
                "datatype": float,
                "default": 50.0,
            },
            "wr_multiplier": {
                "description": "The factor wr is multiplied by to calibrate the rating of a player with elo and kd.",
                "datatype": float,
                "default": 1.5,
            }
        }
        self.check_factor_list = {
            "poisson_rate": self.check_poisson_rate,
            "team_num": self.check_team_num,
            "kd_mean": self.check_kd_mean,
            "wr_mean": self.check_wr_mean,
            "allowable_diff": self.check_allowable_diff,
            "elo_mean": self.check_elo_mean,
            "team_dif": self.check_team_dif,
            "wait_thresh": self.check_wait_thresh,
            "slack": self.check_slack,
            "cov_matrix": self.check_cov_matrix,
            "kd_multiplier": self.check_kd_multiplier,
            "wr_multiplier": self.check_wr_multiplier
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_poisson_rate(self):
        for i in range(len(self.factors["poisson_rate"])):
            if self.factors["poisson_rate"][i] < 0:
                return False
        return True

    def check_team_num(self):
        return self.factors["team_num"] > 0

    def check_kd_multiplier(self):
        return self.factors["kd_multiplier"] > 0

    def check_wr_multiplier(self):
        return self.factors["wr_multiplier"] > 0

    def check_kd_mean(self):
        return self.factors["kd_mean"] > 0

    def check_wr_mean(self):
        return self.factors["wr_mean"] > 0

    def check_allowable_diff(self):
        return self.factors["allowable_diff"] > 0

    def check_elo_mean(self):
        return self.factors["elo_mean"] > 0

    def check_wait_thresh(self):
        return self.factors["wait_thresh"] > 0

    def check_slack(self):
        return self.factors["slack"] > 0

    def check_team_dif(self):
        return self.factors["team_dif"] > 0

    def check_cov_matrix(self):
        for i in range(len(self.factors["cov_matrix"])):
            for j in range(len(self.factors["cov_matrix"])):
                if self.factors["cov_matrix"][i][j] < 0:
                    return False
        return True

    def matchups(self, players):
        team_comps = list(combinations(players, self.factors["team_num"]))
        matches = []
        for i in range(len(team_comps)): # iterate through all combos
            new_list = [players[j] for j in range(len(players)) if players[j] not in team_comps[i]]
            opp_teams = list(combinations(new_list, self.factors["team_num"]))
            for l in range(len(opp_teams)):
                team = []
                team.append(opp_teams[l])
                team.append(team_comps[i])
                matches.append(team)
        return matches

    def prio(self, matchup, prios):
        count = 0
        for i in range(len(matchup)):
            for j in range(len(matchup[i])):
                for k in range(len(prios)):
                    if prios[k] == matchup[i][j]:
                        count += 1
        return count
    def sufficient(self, matchup): # need to change if priority queue is large
        if abs(np.mean(matchup[0]) - np.mean(matchup[1])) > self.factors["team_dif"]:
            return False
        for i in range(len(matchup)):
            difference = max(matchup[i]) - min(matchup[i])
            if difference > self.factors["allowable_diff"]:
                return False
        return True

    def prio_sufficient(self, matchup): # need to change if priority queue is large
        if abs(np.mean(matchup[0]) - np.mean(matchup[1])) > self.factors["team_dif"]:
            return False
        for i in range(len(matchup)):
            difference = max(matchup[i]) - min(matchup[i])
            if difference > (self.factors["allowable_diff"] + self.factors["slack"]):
                return False
        return True

    def wait_count(self, arrival_times, players, matchup):
        wait_time = 0
        for i in range(2):
            for j in range(self.factors["team_num"]):
                temp_ind = players.index(matchup[i][j])
                wait_time += abs(arrival_times[temp_ind] - arrival_times[-1])
        return wait_time

    def replicate(self, rng_list):
        # """
        # Simulate a single replication for the current model factors.
        #
        # Arguments
        # ---------
        # rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        #     rngs for model to use when simulating a replication
        #
        # Returns
        # -------
        # responses : dict
        #     performance measures of interest
        #     "avg_diff" = the average Elo difference between all pairs
        #     "avg_wait_time" = the average waiting time
        # gradients : dict of dicts
        #     gradient estimates for each response
        # """
        # # Designate separate random number generators.
        elo_rng = rng_list[0]
        arrival_rng = rng_list[1]
        uni_rng = rng_list[2]
        # # Initialize statistics.
        # # Incoming players are initialized with a wait time of 0.
        waiting_players = []
        elo_diffs = []
        arr_times = []
        # # Simulate arrival and matching and players.
        #
        # """
        # NSPP: Generate arrival times at the maximum rate at which they arrive
        # Multiply by probability for each hour to see if you accept or reject
        # """
        # # thinning the non-stationary poisson process
        max_lambda = max(self.factors["poisson_rate"])
        t_star = []  # gen arrivals
        times = []  # accepted arrivals
        new_t = []
        u = []
        for i in range(3000):
            new_t.append(arrival_rng.expovariate(1 / max_lambda))

        t_star.append(new_t[0])
        for i in range(1, len(new_t)):
            t_star.append(t_star[i - 1] + new_t[i])

        t_star = [i for i in t_star if i <= 24]

        for i in range(len(t_star)):
            u.append(uni_rng.uniform(0, 1))

        for i in range(len(t_star)):
            ind = int(np.floor(t_star[i]))
            if self.factors["poisson_rate"][ind] / max_lambda >= u[i]:
                times.append(t_star[i])

        wait_times = []
        for i in range(len(times)):
            wait_times.append(0)

        count = len(times)
        mean = [self.factors["elo_mean"], self.factors["kd_mean"], self.factors["wr_mean"]]
        cov = self.factors["cov_matrix"]
        time_list = []
        matched = []
        time_waited = []
        total_team_dif = 0
        for player in range(len(times)):  # iterating through each player as they come in to create teams
            print(player, "hello")
            teams = []
            time = times[player]  # update time to new player
            elo, kd, wr = elo_rng.mvnormalvariate(mean, cov, factorized = False)  # create elo, wr, and kd
            while kd < 0 or wr < 0 or elo < 0:
                kd, wr, elo = elo_rng.mvnormalvariate(mean, cov, factorized = False)
            player_rating = kd + wr + elo  # compound into one value (change to account for different weight after ***)
            waiting_players.append(player_rating)  # place player in queue
            time_list.append(time) # place time of player in queue in separate list
            prio = np.any(abs(time - wait_time) > self.factors["wait_thresh"] for wait_time in time_list) # has anyone been waiting longer than the threshold? True/False
            prio_queue = [] # priority player queue
            team_comps = list(combinations(waiting_players, self.factors["team_num"]))  # makes combinations of teams of x players
            matches = self.matchups(waiting_players) # returns a list of 5v5 possible matchups
            if not prio:
                matches = [match for match in matches if self.sufficient(match)]
            else:
                matches = [match for match in matches if self.prio_sufficient(match)]
            if prio:
                for time_entered in range(len(time_list)): # checking if someone has been waiting longer than the threshold adn places tyhem in priority queue
                    if abs(time_list[-1] - time_list[time_entered]) > self.factors["wait_thresh"]:
                        prio_queue.append(waiting_players[time_entered])
            counter = []
            if not matches: # if no matches within allowable difference, go to next player
                continue
            for i in range(len(matches)): # determines how many people in queue are a priority
                count = self.prio(matches[i], prio_queue)
                counter.append(count)
            queue_times = []
            for i in range(len(matches)): # records the total wait time of players in a given match and adds them to a list
                wait = self.wait_count(time_list, waiting_players, matches[i])
                queue_times.append(wait)
            max_prios = max(counter)
            max_wait = 0
            matchup_ind = 0
            for i in range(len(counter)): # goes through wait times of teams and determines which team has been waiting the longest
                if max_prios == counter[i]:
                    if queue_times[i] > max_wait:
                        max_wait = queue_times[i]
                        matchup_ind = i
            matched.append(matches[matchup_ind]) # creates the match with longest wait time of all teammates
            temp_ind = 0
            team_difference = abs(np.mean(matches[matchup_ind][0]) - np.mean(matches[matchup_ind][1]))
            total_team_dif += team_difference
            for i in range(len(matches[matchup_ind])): # records the wait time of each player who is placed on the new team
                for j in range(len(matches[matchup_ind][i])):
                    waiting_players.index(matches[matchup_ind][i][j])
                    waiting_players.remove(matches[matchup_ind][i][j])
                    queue_enter = time_list.pop(temp_ind)
                    if len(time_list) > 0:
                        wait = time_list[-1] - queue_enter
                        time_waited.append(wait)
                    else:
                        time_waited.append(0)
        # Compose responses and gradients.
        print(len(matched))
        print(waiting_players)
        print(time_list)
        avg_team_dif = total_team_dif / len(matched)
        avg_wait = np.mean(time_waited)
        avg_wait_time = avg_wait * 60
        responses = {"average elo difference between teams": avg_team_dif,
                      "average wait time in minutes": avg_wait_time,
                      }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients
"""
Summary
-------
Minimize the expected Elo difference between all pairs of matched
players subject to the expected waiting time being sufficiently small.
"""


class ValorantMatch(Problem):
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
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
            prev_cost : list
                cost of prevention
            upper_thres : float > 0
                upper limit of amount of contamination
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
    def __init__(self, name="VALORANT", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.dim = 1
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.minmax = (-1,)
        self.constraint_type = "stochastic"
        self.variable_type = "continuous"
        self.lower_bounds = (0,)
        self.upper_bounds = (2400,)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"allowable_diff"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (150,)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000
            },
            "upper_time": {
                "description": "upper bound on wait time",
                "datatype": float,
                "default": 5.0
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "upper_time": self.check_upper_time,
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = ValorantMatchmaking(self.model_fixed_factors)

    def check_upper_time(self):
        return self.factors["upper_time"] > 0

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
            "allowable_diff": vector[0]
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
        vector = (factor_dict["allowable_diff"],)
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
        objectives = (response_dict["avg_diff"],)
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
        stoch_constraints = (response_dict["avg_wait_time"],)
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
        det_stoch_constraints = (-1 * self.factors["upper_time"],)
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
        det_objectives = (0,)
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
        return x >= 0

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
        x = (min(max(0, rand_sol_rng.normalvariate(150, 50)), 2400),)
        return x
