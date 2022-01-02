
Chess Matchmaking Optimization
==============================

Original author of this problem is Bryan Chong (March 15, 2015).

Chess players are rated using the Elo rating system, which assigns a (non-unique)
number to each player based on their level of skill. The problem considered here
has to do with optimizing online chess matchmaking, such that players are matched
with players of similar skill level.

The decision variable :math:`x` is the search width --- the maximum allowable difference
in Elo rating between two matched players. :math:`N` players are drawn from a distribution
of Elo ratings and arrive (independent of their rating) according to a stationary
Poisson process with rate :math:`\lambda`. When a player arrives, and there is an existing,
unmatched player with Elo rating within :math:`x` of the first player's Elo rating, they
are matched. If not, then the player waits for an opponent with an appropriate Elo
rating to arrive.

The optimization problem is thus to minimize the average Elo difference between all
pairs of matched players, such that the average waiting time is :math:`\leq \delta`.

*Recommended Parameter Settings:* To create the Elo distribution, first generate
a normal distribution with mean :math:`1200` and standard deviation :math:`\frac{1200}{\sqrt(2)*\text{erfcinv}(\frac{1}{50})}`,
where erfcinv is the inverse complementary error function. This results in a distribution
where the 1st percentile is at :math:`0`, and the 99th percentile is at :math:`2400`.
Next, truncate the distribution at :math:`0` and :math:`2400`.

Let :math:`N = 10000`, :math:`\delta = 5` minutes, and :math:`\lambda = 1` minute.

*Starting Solution(s):* Let :math:`x = 150`. If multiple starting solutions are required,
first draw :math:`x` from a normal distribution with mean :math:`150` and standard
deviation :math:`50`, then set :math:`x = \max(x, 30)`.

*Measurement of Time:* Number of replications of :math:`N` generated players.

*Optimal Solution:* Unknown.

*Known Structure:* Unknown.