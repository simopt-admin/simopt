Model: Voting Machines in Ohio (Voting)
==========================================

Description:
------------
In rough terms the problem is how many touch-screen voting machines should be allocated to each precinct in Franklin County, Ohio, in order to keep waiting times small?

    1. Voters arrive according to a Poisson process with constant arrival rate :math:`λ_i`` to Precinct :math:`i, i = 1, 2, . . . , p` over :math:`h = 13` hours. The arrival rates depend on the number of registered voters :math:`r_i` in a district and a turnout parameter :math:`τ_i` given as a percentage of registered voters. The arrival rate to Precinct :math:`i` is then :math:`λ_i = r_iτ_i/h`.
    
    2. The polls are open for :math:`h = 13` hours, do not accept arriving voters after :math:`h` hours, but stay open to allow all voters that were queued at time :math:`h` to vote.
    
    3. The turnout parameter :math:`τ_i` is given by :math:`a_i + b_iT` where :math:`a_i` is the midpoint turnout percentage, and :math:`b_i` is the turnout range specific to Precinct :math:`i`. The random variable :math:`T` is triangularly distributed on [−1, 1] with mode 0 and is the same for all precincts.
    
    4. The time taken for a voter to vote at any precinct is gamma distributed with mean 7.5 minutes and standard deviation 2 minutes. The voter is using a machine for this entire time.
    
    5. Voting machines break down at the start of voting with probability 0.05, and can be repaired in a time that is gamma distributed with mean 1 hour and standard deviation 20 minutes. They do not subsequently break down.
    
    6. Voters do not get fed up in line and abandon.
    
    7. Voters do not switch to provisional ballots when waiting times are long.
    
    8. There are 4600 machines. Machines are to be allocated once at the beginning of the day, and not reallocated.

Let :math:`\bar{W_i}` be the observed average time in system (queueing plus voting) of the voters that go to Precinct :math:`i`. We want to identify the machine allocation that minimizes :math:`E\max^p_{i=1} \bar{W_i}`. (This is the quantity that would “get us on the news” if the machine allocation were poor.)

Sources of Randomness:
----------------------
<The number and nature of sources of randomness.>

Model Factors:
--------------
* <factor1name>: <short description>

    * Default: <default value>

* <factor2name>: <short description>

    * Default: <default value>

* <factor3name>: <short description>

    * Default: <default value>

Responses:
---------
* <response1name>: <short description>

* <response2name>: <short description>

* <response3name>: <short description>


References:
===========
This model is adapted from the article <article name with full citation + hyperlink to journal/arxiv page> 




Optimization Problem: <problem_name> (<problem_abbrev>)
========================================================

Decision Variables:
-------------------
* <dv1name that matches model factor name>
* <dv2name that matches model factor name>

Objectives:
-----------
<Description using response names. Use math if it is helpful.>

Constraints:
------------
<Description using response names. Use math if it is helpful.>

Problem Factors:
----------------
* <factor1name>: <short description>

  * Default: <default value>
  
* <factor2name>: <short description>

  * Default: <default value>

Fixed Model Factors:
--------------------
* <factor1name>: <fixed value>

* <factor2name>: <fixed value>

Starting Solution: 
------------------
* <dv1name>: <dv1initialvalue>

* <dv2name>: <dv2initialvalue>

Random Solutions: 
------------------
<description of how to generate random solutions>

Optimal Solution:
-----------------
<if known, otherwise unknown>

Optimal Objective Function Value:
---------------------------------
<if known, otherwise unknown>


Optimization Problem: <problem_name> (<problem_abbrev>)
========================================================

...
