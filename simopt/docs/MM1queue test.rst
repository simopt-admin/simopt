Model: <model_name> (<model_abbreviation>)
==========================================

Description:
------------
<A paragraph describing the stochastic model. Use math if it is helpful.>

Sample math... :math:`S = 1500`

Sample math... 

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

* <factor3name>: <short description>

    * Default: <default value>

Respones:
---------
* average sojourn time: <short description>

* the average waiting time: <short description>

* fraction of customers wait: <short description>


References:
===========
This example is adapted from Cheng, R and Kleijnen,J.(1999). Improved Design of Queueing Simulation Experience with Highly Heteroscedastic Responses. Operations Research, v. 47, n. 5, pp. 762-777




Optimization Problem: Since the model simulates an M/M/1 queue with an Exponential
    interarrival time distribution and an Exponential service time
    distribution, The optimal objective is to minimize the average sojourn time 
    for each entities enter the stations. (<problem_abbrev>)
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