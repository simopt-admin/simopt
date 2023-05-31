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
    A model that simulates a matchmaking problem with
    Elo (multi-variate normally distributed) of players that arrival at a non-stationary Poisson rate.
    Returns the average elo difference between matched players and the average wait time of players matched.

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
        self.n_rngs = 3
        self.n_responses = 2
        self.specifications = {
            "poisson_rate": {
                "description": "Rate of Poisson process for player arrivals per hour according to the time of day",
                "datatype": list,
                "default": [20, 15, 18, 20, 20, 23, 30, 25,
                            30, 20, 15, 18, 20, 20, 23, 30, 25,
                            30, 20, 15, 18, 20, 20, 23, 30, 25, 30]
            },
            "player_diff": {
                "description": "Maximum allowable Elo difference between the maximum and minimum Elo ratings of players within a team",
                "datatype": float,
                "default": 200.0
            },
            "team_diff": {
                "description": "The maximum allowable difference in average player ratings between teams matched against each other",
                "datatype": float,
                "default": 400.0
            },
            "team_num": {
                "description": "Number of players on a team",
                "datatype": int,
                "default": 5
            },
            "mean_matrix": {
                "description": "A list of the means of elo, k/d ratio, and win rate",
                "datatype": list,
                "default": [1200.0, 1.0, 50.0]
            },
            "cov_matrix": {
                "description": "The covariance matrix, with the relationships between elo, kill/death ratio, and win rates.",
                "datatype": list,
                "default": [[40000, 28, 950], [28, .04, .8], [950, .8, 100]]
            },
            "wait_thresh": {
                "description": "The amount of time a player waits before they get prioritized for matchmaking, in hours",
                "datatype": float,
                "default": 0.20,
            },
            "player_slack": {
                "description": "If a player is waiting more than the wait threshold, the allowable difference between players on a team increases by this amount",
                "datatype": float,
                "default": 200.0,
            },
            "team_slack": {
                "description": "If a player is waiting more than the wait threshold, the allowable difference between teams increases by this amount",
                "datatype": float,
                "default": 200.0,
            },
            "multiplier_matrix": {
                "description": "A list of the multipliers to calibrate the elo, k/d ratio, and win rate so they can be combined.",
                "datatype": list,
                "default": [1.0, 1.0, 1.0]
            },
        }
        self.check_factor_list = {
            "poisson_rate": self.check_poisson_rate,
            "team_num": self.check_team_num,
            "player_diff": self.check_player_diff,
            "mean_matrix": self.check_mean_matrix,
            "team_diff": self.check_team_diff,
            "wait_thresh": self.check_wait_thresh,
            "player_slack": self.check_player_slack,
            "team_slack": self.check_team_slack,
            "cov_matrix": self.check_cov_matrix,
            "multiplier_matrix": self.check_multiplier_matrix,
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_poisson_rate(self):
        return np.all([rate > 0 for rate in self.factors["poisson_rate"]])

    def check_team_num(self):
        return self.factors["team_num"] > 0

    def check_multiplier_matrix(self):
        return np.all([multiplier > 0 for multiplier in self.factors["multiplier_matrix"]])

    def check_player_diff(self):
        return self.factors["player_diff"] > 0

    def check_mean_matrix(self):
        return np.all([mean > 0 for mean in self.factors["mean_matrix"]])

    def check_wait_thresh(self):
        return self.factors["wait_thresh"] > 0

    def check_player_slack(self):
        return self.factors["player_slack"] > 0

    def check_team_slack(self):
        return self.factors["team_slack"] > 0

    def check_team_diff(self):
        return self.factors["team_diff"] > 0

    def check_cov_matrix(self):  # instead of checking for negative entries, check to see if the covariance matrix is positive definite
        return np.all(np.linalg.eigvals(self.factors["cov_matrix"]) > 0)

    def matchups(self, players):  # returns a list of lists of five players on each team, returning the indeces of the order in which people arrived into the queue
        """

        Parameters
        ----------
        players : a list of the indices of players within the queue

        Returns
        -------
        matches : a list of matchups, which contain two lists of five players

        """
        team_comps = list(combinations(players, self.factors["team_num"]))
        matches = []
        for team in team_comps:  # iterate through all combos
            new_list = [players[j] for j in range(len(players)) if players[j] not in team]
            opp_teams = list(combinations(new_list, self.factors["team_num"]))
            for j in range(len(opp_teams)):
                matches.append([opp_teams[j], team])
        return matches

    def prio(self, matchup, prios):  # counts the number of priority players within a matchup, and returns that count
        """

        Parameters
        ----------
        matchup : a list of two lists of five, with the indices of each player within the list
        prios : a list of players who are within the priority queue

        Returns
        -------
        count : the amount of priority players within the matchup

        """
        count = 0
        for i in range(2):
            for j in range(self.factors["team_num"]):
                for k in range(len(prios)):
                    if prios[k] == matchup[i][j]:
                        count += 1
        return count

    def sufficient(self, matchup, elos, prio_bool):  # obtains the elos of players within a matchup and ensures the teams and players within the team are within the allowable difference
        """

        Arguments
        ----------
        matchup : a list of two lists of five, with the indices of each player within the list
        elos : a list of Elos for all players in the queue
        prio_bool : a boolean stating whether there are players who need to be prioritized

        Returns
        -------
        Returns a boolean stating whether the players in the matchup are within the allowable difference between and within teams

        """
        elo_matchup = []
        for i in range(2):
            team = []
            for j in range(self.factors["team_num"]):
                team.append(elos[matchup[i][j]])
            elo_matchup.append(team)
        if abs(np.mean(elo_matchup[0]) - np.mean(elo_matchup[1])) > self.factors["team_diff"]:
            return False
        for i in range(2):
            difference = max(elo_matchup[i]) - min(elo_matchup[i])
            if prio_bool:
                if difference > (self.factors["player_diff"] + self.factors["player_slack"]):
                    return False
            else:
                if difference > self.factors["player_diff"]:
                    return False
        return True

    def wait_count(self, arrival_times, players_ind, matchup):
        """

        Parameters
        ----------
        arrival_times : a list of all the arrival times of the players within the queue
        players_ind : a list of the indices for all players within the queue
        matchup : the matchup for which the wait time is being calculated

        Returns
        -------
        wait_time : the sum of the wait times for all players within the matchup

        """
        wait_time = 0
        for i in range(2):
            for j in range(self.factors["team_num"]):
                temp_ind = players_ind.index(matchup[i][j])  # 217-218
                wait_time += abs(arrival_times[temp_ind] - arrival_times[-1])
        return wait_time

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
            "avg_team_diff" = the average Elo difference between teams matched against each other
            "avg_wait_time" = the average waiting time
        gradients : dict of dicts
            gradient estimates for each response
        Designate separate random number generators.
        """

        elo_rng = rng_list[0]
        arrival_rng = rng_list[1]
        uni_rng = rng_list[2]

        """
        Initialize statistics.
        Incoming players are initialized with a wait time of 0.
        Simulate arrival and matching and players.

        NSPP: Generate arrival times at the maximum rate at which they arrive
        Multiply by probability for each hour to see if you accept or reject
        """
        max_lambda = max(self.factors["poisson_rate"])
        t_star = 0
        times = []
        while t_star < 24:
            e = arrival_rng.expovariate(max_lambda)
            t_star += e
            u = uni_rng.uniform(0, 1)
            if t_star < 24:
                if u <= self.factors["poisson_rate"][int(np.floor(t_star))] / max_lambda:
                    times.append(t_star)
        mean = self.factors["mean_matrix"]
        cov = self.factors["cov_matrix"]
        arrival_times = [] # 285
        matched = []
        time_waited = []
        waiting_players = []
        total_team_diff = 0
        ind_player = []
        for _ in range(len(times)):
            new_mean = []
            elo, kd, wr = elo_rng.mvnormalvariate(mean, cov, factorized=False)  # create elo, wr, and kd
            while kd < 0 or wr < 0 or elo < 0:
                elo, kd, wr = elo_rng.mvnormalvariate(mean, cov, factorized=False)
            new_mean.extend((elo, kd, wr))
            player_rating = np.dot(self.factors["multiplier_matrix"], new_mean)  # creates player rating based on their elo, kill/death ratio, and win rate percentage
            waiting_players.append(player_rating)  # place player in queue
        for player in range(len(times)):  # iterating through each player as they come in to create teams
            time = times[player]  # update time to the time latest player entered
            ind_player.append(player)
            arrival_times.append(time)  # place time of player in queue in separate list
            if len(ind_player) < self.factors["team_num"] * 2:
                continue
            prio = np.any([abs(time - wait_time) > self.factors["wait_thresh"] for wait_time in arrival_times])  # has anyone been waiting longer than the threshold? True/False
            prio_queue = []  # priority player queue
            matches = self.matchups(ind_player)  # returns a list of 5v5 possible matchups
            # if not prio:
            #     matches = [match for match in matches if self.sufficient(match, waiting_players)]
            # else:
            #     matches = [match for match in matches if self.prio_sufficient(match, waiting_players)]
            matches = [match for match in matches if self.sufficient(match, waiting_players, prio)]
            if prio:
                for time_entered in range(len(arrival_times)):  # checking if someone has been waiting longer than the threshold and places them in priority queue
                    if abs(time - arrival_times[time_entered]) > self.factors["wait_thresh"]:
                        prio_queue.append(ind_player[time_entered])
            counter = []
            print(len(matched), len(ind_player))
            if not matches:  # if no matches within allowable difference, go to next player
                continue
            for i in range(len(matches)):  # determines how many people in queue are a priority
                count = self.prio(matches[i], prio_queue)
                counter.append(count)
            queue_times = []
            for i in range(len(matches)):  # records the total wait time of players in a given match and adds them to a list
                wait = self.wait_count(arrival_times, ind_player, matches[i])
                queue_times.append(wait)  # total wait time for matchups, same length as matches
            max_prios = max(counter)  # finds the team with maximum priority players
            max_wait = 0
            matchup_ind = 0
            for i in range(len(counter)):  # goes through wait times of teams and determines which team has been waiting the longest
                if max_prios == counter[i]:
                    if queue_times[i] > max_wait:
                        max_wait = queue_times[i]
                        matchup_ind = i
            matched.append(matches[matchup_ind])  # creates the match with longest wait time of all teammates
            team_difference = abs(np.mean([waiting_players[player_idx] for player_idx in matches[matchup_ind][0]]) - np.mean([waiting_players[player_idx] for player_idx in matches[matchup_ind][1]]))
            total_team_diff += team_difference
            for i in range(len(matches[matchup_ind])):  # records the wait time of each player who is placed on the new team
                for j in range(len(matches[matchup_ind][i])):
                    temp_ind = ind_player.index(matches[matchup_ind][i][j])
                    ind_player.remove(matches[matchup_ind][i][j])
                    queue_enter = arrival_times.pop(temp_ind)
                    wait = time - queue_enter
                    time_waited.append(wait)
        # Compose responses and gradients.
        print(len(matched))
        print(ind_player)
        print(arrival_times)
        print(matched[0], matched[1])
        avg_team_diff = total_team_diff / len(matched)
        avg_wait = np.mean(time_waited)
        avg_wait_time = avg_wait * 60
        responses = {"avg_team_diff": avg_team_diff,
                     "avg_wait_time": avg_wait_time,
                }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Minimize the expected Elo difference between teams of matched
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
            upper_time : float > 0
                upper limit of time waited
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
    def __init__(self, name="VALORANT-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.dim = 5
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
        self.model_decision_factors = {"player_diff", "team_diff", "wait_thresh", "player_slack", "team_slack"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (150, 200, 0.15, 100, 100)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000
            },
            "upper_time": {
                "description": "upper bound on wait time",
                "datatype": float,
                "default": 10.0
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
            "player_diff": vector[0],
            "team_diff": vector[1],
            "wait_thresh": vector[2],
            "player_slack": vector[3],
            "team_slack": vector[4]
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
        vector = (factor_dict["player_diff"], factor_dict["team_diff"], factor_dict["wait_thresh"], factor_dict["player_slack"], factor_dict["team_slack"])
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
        objectives = (response_dict["avg_team_diff"],)
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
        return np.all(x >= 0)

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
        # use a diagonal matrix for cov matrix
        cov_matrix = [[10, 0, 0, 0, 0], [0, 20, 0, 0, 0], [0, 0, .05, 0, 0], [0, 0, 0, 10, 0], [0, 0, 0, 0, 10]]
        mean_matrix = [200, 400, .20, 200, 200]
        player_diff, team_diff, wait_thresh, player_slack, team_slack = rand_sol_rng.mvnormalvariate(mean_matrix, cov_matrix, factorized=False)
        a = min(max(0, player_diff), 2400)
        b = min(max(0, team_diff), 2400)
        c = min(max(0, wait_thresh), 10)
        y = min(max(0, player_slack), 2400)
        z = min(max(0, team_slack), 2400)
        x = (a, b, c, y, z)
        return x
