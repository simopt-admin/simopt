Model: throughput
==========================================

Description:
------------
A model that simulates a working station with an n-stage flow line and a finite buffer storage, 
an infinite number of jobs in front of Station. A single server at each station with 
an Exponential(x) service time. The optimal objective solution is to find a buffer allocation and service rates
such that the throughput (average output of the flow line per unit time) is maximized.
.. math::

   No specific math equation

Sources of Randomness:
----------------------
This model has assigned random generater to arrival rate lists and service rate lists.

Model Factors:
--------------
* buffer: Parameter of the buffer allocation distribution

  * Default: 10.0
  
* prate: rate parameter lambda for the exponential distribution used to generate random processing times for each stations

  * Default: 10.0

* n: The number of the total working station.
  
  * Default: 3

* warmup: Represents the number of people as warmup before collecting statistics

  * Default: 2000

* jobs: Represents the number of people required for calculating throughput.

  * Default: 50




Respones:
---------
* Thoughtput: The average output of the flow line per unit time, represent as 50/T.

* rate_list: The service rate list that the throughput of the system is maximized.

* buffer_list: The buffer list that the throughput of the system is maximized.


References:
===========
This example is taken, almost verbatim, from the article Pichitlamken, J., B. L. Nelson and L. J. Hong, 2006. A sequential procedure for neighborhood selection-of-the-best in optimization via
simulation. European Journal of Operational Research 173, 283–298.





Optimization Problem: Maximization of throughput of the system


Decision Variables:
-------------------
* buffer: b

* prate: r


Objectives:
-----------
The goal is to find a buffer allocation and service rates such that the throughput (average output of the flow line per unit time) is maximized.

Constraints:
------------
b2 + ... bn <= B

r2 + ... rn <= R

Problem Factors:
----------------

* warmup: Number of people as warmup before collecting statistics
  
  * Default: 2000

* jobs: Represents the number of people required for calculating throughput.

  * Default: 50

* B: Represents the maximum number of the sum of the buffer of n stations

  * Default: 20

* R: Represents the maximum number of the sum of the service rate of n stations.

  * Default: 20

* Budget: Maximum number of replications for a solver to take
  * Default: 1000

Fixed Model Factors:
--------------------
None

Starting Solution: 
------------------
None

Random Solutions: 
------------------
Using random-number generator rng.MRG32k3a object to generate random solutions from starting or restarting solvers

Optimal Solution:
-----------------
According to Pichitlamken et al. (2006), there are 2 solutions to the discrete-service-rate moderate-sized problem, 
namely r = (6, 7, 7), b = (12, 8) and r = (7, 7, 6), b = (8, 12) with an expected throughput

Optimal Objective Function Value:
---------------------------------
The rate list of (6, 7, 7) and (7, 7, 6) with buffer list of (12, 8) and (8, 12)


Optimization Problem: The optimal buffer allocation and service rates generated throughput (average output of the flow line per unit time) is 5.776. 
========================================================

...