Model: M/M/1 Queue
==========================================

Description:
------------
This is a model simulates an M/M/1 queue with an Exponential
interarrival time distribution and an Exponential service time
distribution. The optimal objective is to minimize the average sojourn time 
for each entities enter the stations. (M/M/1: A stochastic process Represents a customer flow system with a certain state space)

.. math::

   \frac{ \sum_{t=0}^{N}f(t,k) }{N}

Sources of Randomness:
----------------------
<The number and nature of sources of randomness.>

Model Factors:
--------------
* warmup: Represents the number of people as warmup before collecting statistics

    * Default: 50

* people: Represents the number of people from which to calculate the average sojourn time.

    * Default: 200
  
Respones:
---------
* avg_sojourn_time: The average of sojourn time calculated using data stored in customers' matrix

* avg_waiting_time: The average of waiting time calculated using data stored in customers' matrix

* frac_cust_wait: The fraction of customers who are waiting


References:
===========
This example is adapted from Cheng, R and Kleijnen,J.(1999). Improved Design of Queueing Simulation Experience with Highly Heteroscedastic Responses. Operations Research, v. 47, n. 5, pp. 762-777





Optimization Problem: Minimization of average sojourn time of M/M/1 Queue (M/M/1: A stochastic process Represents a flow system with a certain state space)
========================================================


Decision Variables:
-------------------
* mu


Objectives:
-----------
The goal is to minimize the average sojourn time of this M/M/1 Queue under certain cost and numbers of replications.

Constraints:
------------
No deterministic and stochastic constraints described.

Problem Factors:
----------------
* lambda: Rate parameter of interarrival time distribution between entities

  * Default: 1.5
  
* mu: Rate parameter of service time distribution

  * Default: 3.0

* warmup: Number of people as warmup before collecting statistics
  
  * Default: 20

* people: Number of people from which to calculate the average sojourn time
  
  * Default: 50

Fixed Model Factors:
--------------------
None

Starting Solution: 
------------------
* mu: 3.0

Random Solutions: 
------------------
Using random-number generator rng.MRG32k3a object to generate random solutions from starting or restarting solvers

Optimal Solution:
-----------------
By running the demo_model test function, 
the optimal response for average sojourn time is 0.58169
the optimal response for average waiting time is 0.25554
the optimal response for customers who are waiting is 0.48

Optimal Objective Function Value:
---------------------------------
The objective function value is 0.58169, which reflects the optimal minimized average sojourn time.


Optimization Problem: Minimization of average sojourn time of M/M/1 Queue (M/M/1: A stochastic process Represents a flow system with a certain state space)
========================================================

...