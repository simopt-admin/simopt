Model: Parameter Estimation (paramesti)
==========================================

Description:
------------
A model that simulates MLE estimators for a two-dimentinal :math:`{\\beta}`  variable.
    Returns the 2-D vector x_star that maximizes the probability of seeing
    parameters x in 2-D beta probability density function.


.. math::

   \frac{ \sum_{t=0}^{N}f(t,k) }{N}

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

Respones:
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