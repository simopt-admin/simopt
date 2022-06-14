Model: M/M/1 Queue
==========================================

Description:
------------
This is a model simulates an M/M/1 queue with an Exponential
interarrival time distribution and an Exponential service time
distribution.

Sources of Randomness:
----------------------
1. Exponential interarrival times.
2. Exponential service times.

Model Factors:
--------------
* lambda: Rate parameter of interarrival time distribution.

    * Default: 1.5

* mu: Rate parameter of service time distribution.

    * Default 3.0

* warmup: Represents the number of people as warmup before collecting statistics

    * Default: 50

* people: Represents the number of people from which to calculate the average sojourn time.

    * Default: 200
  
Respones:
---------
* avg_sojourn_time: The average of sojourn time of customers (time customers spend in the system).

* avg_waiting_time: The average of waiting time of customers.

* frac_cust_wait: The fraction of customers who wait.


References:
===========
This example is adapted from Cheng, R and Kleijnen,J.(1999). Improved Design of Queueing Simulation Experience with Highly Heteroscedastic Responses. Operations Research, v. 47, n. 5, pp. 762-777 (https://pubsonline.informs.org/doi/abs/10.1287/opre.47.5.762)



Optimization Problem: Minimize average sojourn time plus penalty (MM1-1)
========================================================


Decision Variables:
-------------------
* mu (service rate parameter)


Objectives:
-----------
Minimize the expected average sojourn time plus a penalty for increasing the rate :math:`c\mu^2`.

Constraints:
------------
No deterministic or stochastic constraints.
Box constraints for non-negativity of mu.

Problem Factors:
----------------
* budget: Max # of replications for a solver to take.

    * Default: 1000

* cost: Cost for increasing service rate.

    * Default: 0.1

Fixed Model Factors:
--------------------
None

Starting Solution: 
------------------
* mu: 3.0

Random Solutions: 
------------------
Generate mu from an exponential distribution with mean 3.

Optimal Solution:
-----------------
None.

Optimal Objective Function Value:
---------------------------------
None.
