Model: Chess Matchmaking Optimization (CHESS)
==========================================

Description:
------------
Chess players are rated using the Elo rating system, which assigns a (non-unique)
number to each player based on their level of skill. The model simulates an online
chess-matchmaking platform, which tries to match players of similar skill level.

The platform uses a search width :math:`x`, which is the maximum allowable difference
in Elo rating between two matched players. :math:`N` players are drawn from a distribution
of Elo ratings and arrive (independent of their rating) according to a stationary
Poisson process with rate :math:`\lambda`. When a player arrives, and there is an existing,
unmatched player with Elo rating within :math:`x` of the first player's Elo rating, they
are matched. If not, then the player waits for an opponent with an appropriate Elo
rating to arrive.

Sources of Randomness:
----------------------
1. To create the Elo distribution, first generate a normal distribution with mean
:math:`1200` and standard deviation :math:`\frac{1200}{\sqrt(2)*\text{erfcinv}(\frac{1}{50})}`,
where erfcinv is the inverse complementary error function. This results in a distribution
where the 1st percentile is at :math:`0`, and the 99th percentile is at :math:`2400`.
Next, truncate the distribution at :math:`0` and :math:`2400`.
2. A stationary Poisson process with rate :math:`\lambda` for arrivals.

Model Factors:
--------------
* elo_mean: Mean of normal distribution for Elo rating.

    * Default: 1200.0

* elo_sd: Standard deviation of normal distribution for Elo rating.

    * Default: 1200 / (np.sqrt(2) * special.erfcinv(1 / 50))

* poisson_rate: Rate of Poisson process for player arrivals.

    * Default: 1.0

* num_players: Number of players.

    * Default: 1000

* allowable_diff: Maximum allowable difference between Elo ratings.

    * Default: 150.0

Responses:
---------
* avg_diff: The average Elo difference between all pairs.

* avg_wait_time: The average waiting time.

References:
===========
Original author of this problem is Bryan Chong (March 15, 2015).




Optimization Problem: Minimize Average Elo Difference (CHESS-1)
========================================================

Decision Variables:
-------------------
* allowable_diff

Objectives:
-----------
Minimize the average Elo difference between all pairs of matched players.

Constraints:
------------
Maximum allowable difference is between 0 and 2400.

The average waiting time is :math:`\leq \delta`.

Problem Factors:
----------------
* budget: Max # of replications for a solver to take.

  * Default: 1000
  
* upper_time: Upper bound on wait time.

  * Default: 5.0

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: (150,)

Random Solutions: 
------------------
First draw :math:`x` from a normal distribution with mean :math:`150` and standard
deviation :math:`50`, then set :math:`x = \min(\max(x, 0), 2400)`.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown