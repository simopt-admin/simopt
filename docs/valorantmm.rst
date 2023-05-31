Model: Valorant Matchmaking Model
=============================================

Description:
------------
Valorant players are rated using an Elo system. The problem involves forming and matching teams
based on players' elo rating, win rate, and kill/death ratio. The model will optimize a Valorant
matchmaking model such that a team of p players is matched against another team with a similar
skill level considering various aspects of game play.

The model has multiple decision variables, :math:`x`, is the maximum allowable difference between
the average elo of both teams. There is also the slack variables, :math:`y`, and, :math:`z`, which
represent the increase/decrease in the allowable range if a player has been waiting longer than the
wait threshold. The maximum allowable difference between the average of teams matched, :math:`m`, and the waiting time
threshold at which a player becomes prioritized, :math:`t`. Players arrive
to the matchmaking pool according to a non-stationary Poisson process, following the process in
Table 1. Players are assigned an elo rating, win rate, and kill/death ratio drawn from a
multivariate normal distribution, and players arrive according to a non-stationary Poisson
process. If when a player arrives there are sufficient players with Elo ratings within :math:`x` of the
player's elo, they are matched on a team. If not, the player waits for a team with an appropriate
Elo rating. When a team has been made they will then be matched against another team that has an
average Elo rating within, :math:`m`, range of the average of that team. If a player has been waiting
longer than :math:`t` minutes, they will be prioritized and the variables, :math:`x` and :math:`m`,
increase/decrease by, :math:`y` and :math:`z`, when creating and matching teams with prioritized players.

Sources of Randomness:
----------------------
1. To create the Elo, first generate a multivariate normal distribution with a 1x3 mean matrix
[1300.0, 1.0, 50.0] and 3x3 covariance matrix [[40000, 28, 950], [28, .04, .8], [950, .8, 100]].

2. A non-stationary Poisson process with rate :math:`\lambda` for arrivals according to the time of day they arrive.

3. A uniform distribution for calculating the arrivals of players within the simulation

Model Factors:
--------------
* poisson_rate: Rate of Poisson process for player arrivals per hour according to the time of day

    * Default: [20, 15, 18, 20, 20, 23, 30, 25, 30, 20, 15, 18, 20, 20, 23, 30, 25, 30, 20, 15, 18, 20, 20, 23, 30, 25, 30]

* player_diff: Maximum allowable Elo difference between the maximum and minimum Elo ratings of players within a team.

    * Default: 200.0

* team_diff: The maximum allowable difference in average player ratings between teams matched against each other.

    * Default: 400.0

* team_num: Number of players on a team.

    * Default: 5

* mean_matrix: A list of the means of elo, k/d ratio, and win rate.

    * Default: [1200.0, 1.0, 50.0]

* cov_matrix: The covariance matrix, with the relationships between elo, kill/death ratio, and win rates.

    * Default: [[40000, 28, 950], [28, .04, .8], [950, .8, 100]]

* wait_thresh: The amount of time a player waits before they get prioritized for matchmaking, in hours.

    * Default: 0.20

* player_slack: If a player is waiting more than the wait threshold, the allowable difference between players on a team increases by this amount.

    * Default: 200.0

* team_slack: If a player is waiting more than the wait threshold, the allowable difference between teams increases by this amounts.

    * Default: 200.0

* multiplier_matrix: A list of the multipliers to calibrate the elo, k/d ratio, and win rate so they can be combined.

    * Default: [1.0, 1.0, 1.0]

Responses:
----------
* avg_team_diff: The average Elo difference between teams that have been paired up.

* avg_wait_time: The average waiting time for all players that have been matched up.

References:
===========
Original author of this problem is Nolan Berry (05.28.23).




Optimization Problem: Minimize Average Elo Difference (CHESS-1)
===============================================================

Decision Variables:
-------------------
* player_diff
* team_diff
* wait_thresh
* player_slack
* team_slack

Objectives:
-----------
Minimize the average Elo difference between paired teams and players within the team.

Constraints:
------------
Maximum allowable difference is between 0 and 2400.

Problem Factors:
----------------
* budget: Max # of replications for a solver to take.

  * Default: 1000
  
* upper_time: Upper bound on wait time.

  * Default: 10.0

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: (150, 200, 0.15, 100, 100)

Random Solutions: 
-----------------
First draw :math:`x` from a normal distribution with mean :math:`150` and standard
deviation :math:`50`, then set :math:`x = \min(\max(x, 0), 2400)`.

First draw player_diff, team_diff, wait_thresh, player_slack, team_slack from a multivariate normal distribution
with mean matrix [200, 400, .20, 200, 200] and covariance matrix [[10, 0, 0, 0, 0], [0, 20, 0, 0, 0],
[0, 0, .05, 0, 0], [0, 0, 0, 10, 0], [0, 0, 0, 0, 10]].
Then set :math:`a = \min(\max(0, player_diff), 2400)`, :math:`b = \min(\max(0, team_diff), 2400)`, :math:`c = \min(\max(0, wait_thresh), 10)`,
:math:`y = \min(\max(0, player_slack), 2400)`, and :math:`z = \min(\max(0, team_slack), 2400)`.
Return x = (a, b, c, y, z)

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown