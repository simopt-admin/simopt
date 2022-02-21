Model: FacilitySizing (facilitysizing)
==========================================

Description:
------------
This example is adapted from the article Rengarajan, T., & Morton, D.P. (2009). Estimating the
Efficient Frontier of a Probabilistic Bicriteria Model. Proceedings of the 2009 Winter Simulation
Conference. [1]
Problem Statement:
The facility-sizing problem is formulated as follows: m facilities are to be installed, each with capacity
xi ≥ 0, i = 1, . . . , m. Then the random demand ξi arrives at facility i, with a known joint distribution
of the random vector ξ = (ξ1, . . . , ξm).
Our goal is to keep the cost of installation as well as the probability of violating demand low. A
realization of the demand, ξ = (ξ1, . . . , ξm), is said to be satisfied by the capacity x if xi ≥ ξi
, ∀i =
1, . . . , m. Thus the risk of failing to satisfy demand ξ = (ξ1, . . . , ξm) is p(x) = P(ξ  x). Let 2 ∈ [0, 1]
be a risk-level parameter, then we obtain the probabilistic constraint:

P(ξ  x) ≤ 2
Meanwhile, the unit cost of installing facility i is ci

, and hence the total cost is h(x) = Pm
i=1 cixi
. The

facility-sizing problem then can be formulated as
min Pm
i=1 cixi
s.t. P(ξ  x) ≤ 2
x ≥ 0

Recommended Parameter Settings: Suppose there are m = 40 facilities and the per unit cost
for each facility i is ci = 1. The demand vector ξ follows a multivariate normal distribution with mean
10 and variance 1 for each component, and correlation coefficient ρi,j = 0.8, i 6= j. Furthermore, ξ is
truncated so that ξ ≥ 0. 2 = 5%.

Sample math... :math:`S = 1500`

Sample math... 

.. math::

   \frac{ \sum_{t=0}^{N}f(t,k) }{N}

Sources of Randomness:
----------------------
1 random demand vector ξ follows a multivariate normal distribution with mean
10 and variance 1 for each component, and correlation coefficient ρi,j = 0.8, i 6= j.

Model Factors:
--------------
* mean_vec: Location parameters if the multivariate normal distribution
      Dataype: list
    * Default: [100,100,100]

* cov: Covariance of multivariate normal distribution
      Datatype: list
    * Default: [[2000, 1500, 500,], [1500, 2000, 750], [500, 750, 2000]]

* capacity: Capactiy
      Datatype: list
    * Default: [150, 300, 400]

* n_fac: The number of facilities
      Datatype: integer
    * Default: 3

    
Respones:
---------
* stockout_flag: binary variable;
                  0: all facilities satisfy the demand
                  1: at least one of the facilities did not satisfy the demand

* n_fac_stockout: integer value;
                  the number of facilities which cannot satisfy the demand

* n_cut: integer value; 
          the number of total demand which cannot satisfy the demand 


References:
===========
This model is adapted from the article <article name with full citation + hyperlink to journal/arxiv page> 




Optimization Problem: FacilitySizingTotalCost (Problem) (facilitysizing)
========================================================

Decision Variables:
-------------------
* <dv1name that matches model factor name> inventory at each facility 
*  capacity 

Objectives:
-----------
Minimize the (deterministic) total cost of installing capacity at
facilities subject to a chance constraint on stockout probability.

Constraints:
------------
<Description using response names. Use math if it is helpful.> 1 stocahstic consraint

Problem Factors:
----------------
* epsilon: maximum allowed probability of stocking out.
* Datatype: float 
  * Default: 0.05
  
* installation_costs: Cost to install a unit of capacity at each facility 
* Datatype: tuple
  * Default: (1, 1, 1)

Fixed Model Factors:
--------------------
* meanA_vec: [100, 100, 100]

* cov: [[2000, 1500, 500], [1500, 2000, 750], [500, 750,2000]]

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
