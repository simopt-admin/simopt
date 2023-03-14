"""
Summary
-------
Simulate matching of Valorant players on an online platform.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/chessmm.html>`_.
"""
from itertools import combinations
import numpy as np
from trueskill import Rating, quality_1vs1, rate_1vs1

class ValorantMatchmaking(Model):
    """
    A model that simulates a matchmaking problem with a
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
                "default": [20, 15, 18, 20, 20, 23, 30, 25]
            },
            "allowable_diff": {
                "description": "maximum allowable difference between Elo ratings",
                "datatype": float,
                "default": 150.0
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
            "kd_var": {
                "description": "The variance of k/d for multivariate normal distribution of k/d, elo, and win rate of players",
                "datatype": float,
                "default": 0.2
            },
            "wr_mean": {
                "description": "The mean win percentage for multivariate normal distribution of k/d, elo, and win rate of players",
                "datatype": float,
                "default": 50.0
            },
            "wr_var": {
                "description": "The variance of win rate for multivariate normal distribution of k/d, elo, and win rate of players",
                "datatype": float,
                "default": 5.0
            },
            "elo_mean": {
                "description": "The mean of the elo of players in a multivariate normal distribution of k/d, elo, and win rate of players",
                "datatype": float,
                "default": 1300
            },
            "elo_var": {
                "description": "The variance of the elo of players in a multivariate normal distribution of k/d, elo, and win rate of players",
                "datatype": float,
                "default": 300
            },
            "kdwr_cov": {
                "description": "The co-variance of win rate and k/d for multivariate normal distribution of k/d, elo, and win rate of players",
                "datatype": float,
                "default": 5.0
            },
            "kdelo_cov": {
                "description": "The co-variance of elo and k/d for multivariate normal distribution of k/d, elo, and win rate of players",
                "datatype": float,
                "default": 5.0
            },
            "elowr_cov": {
                "description": "The co-variance of win rate and elo for multivariate normal distribution of k/d, elo, and win rate of players",
                "datatype": float,
                "default": 5.0
            },
            "team_dif": {
                "description": "Allowable difference in average player rating between teams matched against each other",
                "datatype": float,
                "default": 300.0
            },
            "wait_thresh": {
                "description": "The amount of time a player waits before they get prioritized for matchmaking, in seconds",
                "datatype": float,
                "default": 300.0,
            },
            "slack": {
                "description": "If someone is waiting more than the wait threshold, the allowable difference increases by this amount",
                "datatype": float,
                "default": 300.0,
            }
        }
        self.check_factor_list = {
            "poisson_rate": self.check_poisson_rate,
            "team_num": self.check_team_num,
            "kd_mean": self.check_kd_mean,
            "kd_var": self.check_kd_var,
            "wr_mean": self.check_wr_mean,
            "wr_var": self.check_wr_var,
            "allowable_diff": self.check_allowable_diff,
            "elo_mean": self.check_elo_mean,
            "elo_var": self.check_elo_var,
            "kdwr_cov": self.check_kdwr_cov,
            "kdelo_cov": self.check_kdelo_cov,
            "elowr_cov": self.check_elowr_cov,
            "team_dif": self.check_team_dif,
            "wait_thresh": self.check_wait_thresh,
            "slack": self.check_slack
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

    def check_kd_mean(self):
        return self.factors["kd_mean"] > 0

    def check_kd_cov(self):
        return self.factors["kd_cov"] > 0

    def check_wr_mean(self):
        return self.factors["wr_mean"] > 0

    def check_wr_cov(self):
        return self.factors["wr_cov"] > 0

    def check_allowable_diff(self):
        return self.factors["allowable_diff"] > 0

    def check_elo_mean(self):
        return self.factors["elo_mean"] > 0

    def check_elo_var(self):
        return self.factors["allowable_diff"] > 0

    def check_kdwr_cov(self):
        return self.factors["kdwr_cov"] > 0

    def check_kdelo_cov(self):
        return self.factors["kdelo_cov"] > 0

    def check_elowr_cov(self):
        return self.factors["elowr_cov"] > 0

    def check_wait_thresh(self):
        return self.factors["wait_thresh"] > 0

    def check_slack(self):
        return self.factors["slack"] > 0

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
            "avg_diff" = the average Elo difference between all pairs
            "avg_wait_time" = the average waiting time
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        elo_rng = rng_list[0]
        arrival_rng = rng_list[1]
        uni_rng = rng_list[2]
        multi_rng = rng_list[3]
        # Initialize statistics.
        # Incoming players are initialized with a wait time of 0.
        wait_times = np.zeros(self.factors["num_players"])
        waiting_players = []
        total_diff = 0
        elo_diffs = []
        arr_times = []
        time = 0
        teams = []
        # Simulate arrival and matching and players.

        """
        NSPP: Generate arrival times at the maximum rate at which they arrive
        Multiply by probability for each hour to see if you accept or reject        
        """
        # thinning the non-stationary poisson process
        max_lambda = max(self.factors["poisson_rate"])
        t_star = []  # gen arrivals
        times = []  # accepted arrivals
        new_t = []
        u = []
        for i in range(len(5000)):
            new_t.append(arrival_rng.expovariate(1 / max_lambda))
        t_star.append(new_t[0])
        for i in range(1, len(new_t)):
            t_star.append(new_t[i]+new_t[i+1])
        t_star = [i for i in t_star if i <= 24]
        for i in range(len(t_star)):
            u.append(uni_rng.uniform(0, 1))

        for i in range(len(t_star)):
            ind = int(np.floor(t_star[i]))
            if self.factors["poisson_rate"][ind] / max_lambda >= u[i]:
                times.append(t_star[i])
        # generated arrival times
        # other d.v.s spread of people within team and time waited: along with max allowable difference between teams
        # need to generate elos, kd stats, and win rate stats... then create a stat that takes these into account when making teams
        # make a team of p players using 10 choose 5. Try all combinations and see if teams can be made
        count = len(times)
        mean = [self.factors["kd_mean"], self.factors["wr_mean"], self.factors["elo_mean"]]
        cov = [[self.factors["kd_var"], self.factors["kdwr_cov"], self.factors["kdelo_cov"]], [self.factors["kdwr_cov"], self.factors["wr_var"], self.factors["elowr_cov"]],
               [self.factors["kdwr_cov"], self.factors["kdwr_cov"], self.factors["elo_var"]]]
        # compound elo, kd, and wr
        # need a loop that makes teams within the elo range
        time_list = []
        matched = []
        time_waited = []
        temp_ind = 0
        for player in range(len(times)): # iterating through each player as they come in to create teams
            time = times[player] # update time to new player
            kd, wr, elo = elo_rng.mvnormalvariate(mean, cov, factorized = False) # create elo, wr, and kd
            while kd or wr or elo < 0:
                kd, wr, elo = elo_rng.mvnormalvariate(mean, cov, False)
            player_rating = kd + wr + elo # compound into one value
            waiting_players.append(player_rating) # place player in queue
            time_list.append(time) # place time of player in queue in separate list
            prio = np.any(abs(time_list[-1] - wait_time) > self.factors["wait_thresh"] for wait_time in time_list) # has anyone been waiting longer than the threshold?
            prio_queue = [] # priority player queue
            for time in range(len(time_list)):
                if abs(time_list[-1] - time) > self.factors["wait_thresh"]:
                    prio_queue.append(time_list[i])
            if not prio_queue:
                team_comp = list(combinations(waiting_players, self.factors["team_num"]))  # makes combinations of teams of x players
            else:
            # have the quality change instead of elo difference
            # class to represent playre
            player = {
                "rating": 0,
                "time": 0,
                "name": "ted"
            }
            if prio is True: # if someone has been waiting longer than the threshold, they get prioritized and the allowable dif increases
                for i in range(len(team_comp)): # iterate through each team combination
                    team_count = 0
                    for j in range(len(self.factors["team_num"])): # iterate through each team member
                        # check team comps to see if teams are within allowable difference + slack
                        if abs(team_comp[i][j] - team_comp[i][self.factors["team_num"]]) > (self.factors["allowable_diff"] + self.factors["slack"]):
                            break
                        else:
                            team_count += 1
                    if team_count == self.factors["team_num"]:
                        for j in range(self.factors["team_num"]):
                            # temp_ind = waiting_players.index(team_comp[i][j])
                            waiting_players.remove(team_comp[i][j]) # need to delete the players in team from waiting queue
                            # del time_list[temp_ind]
                    teams.append(team_comp[i]) # add team to teams rdy to be matched
                    # only remove people once optimal pair is found
                if len(teams) >= 2:
                    team_avg = []
                    teamdif = []
                    records = []
                    lowest_dif = 0
                    # I wave been waiting longer and closest matchups between teams
                    matchups = list(combinations(teams,ant to prioritize players who h 2))  # creating all possible teams of two
                    for i in range(len(teams)):  # obtain averages of each team that has been made
                        avg = sum(teams[i]) / len(teams[i])
                        team_avg.append(avg)
                    for i in range(len(matchups)):
                        if self.factors["team_dif"] >= abs(matchups[i][0] - matchups[i][1]):
                            teamdif.append(abs(matchups[i][0] - matchups[i][1]))
                            records.append(i)
                    lowest_dif = teamdif.index(min(teamdif))  # index of the smallest difference between two teams
                    matched.insert(len(matched), matchups[records[lowest_dif]][0], matchups[records[lowest_dif]][1])

            else:
                for i in range(len(team_comp)): # iterate through each team combination
                    team_count = 0
                    for j in range(len(self.factors["team_num"])): # iterate through each team member
                        # check team comps to see if teams are within allowable difference
                        if abs(team_comp[i][j] - team_comp[i][self.factors["team_num"]]) > self.factors["allowable_diff"]:
                            break
                        else:
                            team_count += 1
                    if team_count == self.factors["team_num"]:
                        for j in range(self.factors["team_num"]):
                            waiting_players.remove(team_comp[i][j]) # need to delete the players in team from waiting queue
                    teams.append(team_comp[i]) # add team to teams rdy to be matched
        # if first player in queue has been waiting > x then prioritize them and increase threshold for teams


                # increase theshold if waiting a while
                # need separate list of when the waiting players entered the queue
        # need to match teams
        for player in range(self.factors["num_players"]):
            # Generate interarrival time of the player.
            time = arrival_rng.poissonvariate(self.factors["poisson_rate"])
            # Generate rating of the player via acceptance/rejection (not truncation).
            player_rating = elo_rng.triangular(self.factors["elo_min"], self.factors["elo_max"], self.factors["elo_mode"])
            while player_rating < 0 or player_rating > 2400:
                player_rating = elo_rng.normalvariate(self.factors["elo_mean"], self.factors["elo_sd"])
            # Attempt to match the incoming player with waiting players in FIFO manner.
            old_total = total_diff
            for p in range(len(waiting_players)):
                matched = False
                if not matched and abs(player_rating - waiting_players[p]) <= self.factors["allowable_diff"]:
                    total_diff += abs(player_rating - waiting_players[p])
                    elo_diffs.append(abs(player_rating - waiting_players[p]))
                    del_player = p
                    matched = True
                    break
                else:
                    wait_times[p] += time
            del waiting_players[del_player]
            del player_tracker[del_player]
            # If incoming player is not matched, add them to the waiting pool.
            if old_total == total_diff:
                waiting_players.append(player_rating)
        # Compose responses and gradients.
        responses = {"avg_diff": np.mean(elo_diffs),
                     "avg_wait_time": np.mean(wait_times)
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Minimize the expected Elo difference between all pairs of matched
players subject to the expected waiting time being sufficiently small.
"""


class ChessAvgDifference(Problem):
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
    def __init__(self, name="CHESS-1", fixed_factors=None, model_fixed_factors=None):
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
        self.model = ChessMatchmaking(self.model_fixed_factors)

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
