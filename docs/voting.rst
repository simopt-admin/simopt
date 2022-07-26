Model: Voting Machines in Ohio (Voting)
=======================================

Description:
------------
After allocating touch-screen voting machines to the precincts in Franklin County, Ohio, the model simulates voter's experience on voting day.

    1. Voters arrive according to a Poisson process with constant arrival rate :math:`λ_i`` to Precinct :math:`i, i = 1, 2, . . . , p` over :math:`h = 13` hours. The arrival rates depend on the number of registered voters :math:`r_i` in a district and a turnout parameter :math:`τ_i` given as a percentage of registered voters. The arrival rate to Precinct :math:`i` is then :math:`λ_i = r_iτ_i/h`.
    
    2. The polls are open for :math:`h = 13` hours, do not accept arriving voters after :math:`h` hours, but stay open to allow all voters that were queued at time :math:`h` to vote.
    
    3. The turnout parameter :math:`τ_i` is given by :math:`a_i + b_iT` where :math:`a_i` is the midpoint turnout percentage, and :math:`b_i` is the turnout range specific to Precinct :math:`i`. The random variable :math:`T` is triangularly distributed on [−1, 1] with mode 0 and is the same for all precincts.
    
    4. The time taken for a voter to vote at any precinct is gamma distributed with mean 7.5 minutes and standard deviation 2 minutes. The voter is using a machine for this entire time.
    
    5. Voting machines break down at the start of voting with probability 0.05, and can be repaired in a time that is gamma distributed with mean 1 hour and standard deviation 20 minutes. Repaired machines do not break down again.
    
    6. Voters do not get fed up in line and abandon.
    
    7. Voters do not switch to provisional ballots when waiting times are long.
    
    8. There are 4600 machines. Machines are to be allocated once at the beginning of the day and are not reallocated.


Sources of Randomness:
----------------------

    1. Whether a machine is broken down at the start of the day. 
   
    2. How long a broken down machine will take to fix.
   
    3. Voting turnout percentage.
   
    4. Interarrival times of voters.
   
    5. Time to finish voting for each voter.

Model Factors:
--------------
* mach_allocation: number of machines allocation for precinct i

    * Default: [10, 10, 10, 10, 10]

* n_mach: max number of machines available

    * Default: 50

* mid_turn_per: midpoint turnout percentage for precinct i

    * Default: [10, 15, 10, 50, 30]

* turn_ran: turnout range specific to precinct i

    * Default: [.4, .2, .6, .3, .1] 

* reg_vote: number of registered voters in precinct i

    * Default: [100, 200, 100, 400, 200]

* mean_time2vote: the mean time for the gamma distributed time taken to vote

    * Default: 7.5

* stdev_time2vote: the standard deviation for the gamma distributed time to vote

    * Default: 2

* mean_repair: voting machines are repaired according to a gamma distribution, this is the mean time, minutes

    * Default: 60

* stdev_repair: standard deviation for gamma distribution for time to repair a machine, minutes

    * Default: 20

* bd_prob: probability at which the voting machines break down (bd)

    * Default: 0.05

* hours: number of hours open to vote

    * Default: 13.0

* n_prec: number of precincts

    * Default: 5

Responses:
----------
* prec_avg_waittime: all wait times for all precincts

* perc_no_waittime: the precentage of voters that did not have to wait at each precinct 

References:
===========
This model is adapted from Shane G. Henderson's problem "Voting Machines in Ohio" created on November 7th, 2008.


Optimization Problem: Minimize Max Wait Time (VOTING-1)
========================================================

Decision Variables:
-------------------
* Mach_allocation

Objectives:
-----------
Let :math:`\bar{W_i}` be the observed average time in system (queueing plus voting) of the voters that go to Precinct :math:`i`. We want to identify the machine allocation that minimizes :math:`E\max^p_{i=1} \bar{W_i}`. (This is the quantity that would “get us on the news” if the machine allocation were poor.)

Constraints:
------------

* :math:`\sum`(mach_allocation) = n_mach

* mach_allocation(k) > 0

Problem Factors:
----------------
* initial_solution: Initial solution from which solvers start.

  * Default: (10, 10, 10, 10, 10)
  
* budget: Max # of replications for a solver to take.

  * Default: 10000

Fixed Model Factors:
--------------------
N/A

Starting Solution: 
------------------
* mach_allocation: (10, 10, 10, 10, 10)

Random Solutions: 
-----------------
Generate allocations uniformly at random from the set of vectors (of length equal to the number of precincts) whose values sum to the number of machines.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown