Model: COVID Testing (COVID)
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
There are 6 sources of randomness. Random number streams are allocated to Poisson random variables. 

Model Factors:
--------------
* num_groups: Number of groups.

    * Default: 3

* transmission_rate: Rate of transmission.

    * Default: (0.22, 0.072, 0.0018, 0.0018, 0.061, 0.15, 0.0018, 0.0034, 0.0039, 0.072)

* group_size: Size of each group.

    * Default: (8123, 3645, 4921)

* lamb_exp_inf: Mean time from exposure to infectious.

    * Default: 2.

* lamb_inf_sym: Mean time from infectious to symptom onset.

    * Default: 3.

* lamb_sym: Mean time in symptomatic state.

    * Default: 12.

* lamb_iso: Mean number of isolations.

    * Default: 0.85

* n: Number of days to simulate.

    * Default: 100

* init_infect_percent: Initial proportion of infected.

    * Default: (0.00156, 0.00161, 0.00166)

* freq: Testing frequency of each group.

    * Default: (3/7, 2/7, 1/7)
    
* asymp_rate: Rate of asymptomatic.

    * Default: 0.35

Responses:
---------
* num_infected: number of infected individuals


References:
===========
This model is adapted from the Supplementary Information for the article
Modeling for COVID-19 College Reopening Decisions: Cornell, A Case Study.




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