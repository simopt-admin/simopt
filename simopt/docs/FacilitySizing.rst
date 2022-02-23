Model: FacilitySizing (facilitysizing)
==========================================

Description:
------------

The facility-sizing problem is formulated as follows: :math:`m` facilities are to be installed, each with capacity
:math:`xi ≥ 0, i = 1, . . . , m`. Then the random demand :math:`ξi` arrives at facility :math:`i`, with a known joint distribution
of the random vector :math:`ξ = (ξ1, . . . , ξm)`.
Our goal is to keep the cost of installation as well as the probability of violating demand low. 
A realization of the demand, :math:`ξ = (ξ1, . . . , ξm)`, is said to be satisfied by the capacity :math:`x` if :math:`xi ≥ ξi, ∀i = 1, . . . , m`. 
Thus the risk of failing to satisfy demand :math:`ξ = (ξ1, . . . , ξm)` is :math:`p(x) = P(ξ  x)`. Let :math:`epsilon ∈ [0, 1]` be a risk-level parameter, then we obtain the probabilistic constraint:

:math:`P(ξ  x) ≤ epsilon`
Meanwhile, the unit cost of installing facility i is :math:`ci`, and hence the total cost is :math:`h(x) = Pmi=1 cixi`. 

The facility-sizing problem then can be formulated as
min :math:`SUM cixi`

s.t. :math:`P(ξ !<= x) ≤ epsilon`
:math:`x ≥ 0`


Recommended Parameter Settings: 

Suppose there are :math:`m = 40` facilities and the per unit cost
for each facility i is :math:`ci = 1`. The demand vector :math:`ξ` follows a multivariate normal distribution with mean10 and variance 1 for each component, and correlation coefficient :math:`ρi,j = 0.8` , :math:`i != j`. Furthermore, :math:`ξ` is truncated so that :math:`ξ ≥ 0`. :math:`epsilon = 5%`.


Starting Solutions: :math:`Xi = 1500` 



Sources of Randomness:
----------------------
1 random demand vector :math: `ξ` follows a multivariate normal distribution with mean
10 and variance 1 for each component, and correlation coefficient :math:`ρi,j = 0.8` , :math:`i != j` .

Model Factors:
--------------
* mean_vec: Location parameters if the multivariate normal distribution
    * Dataype: list
    * Default: [100,100,100]

* cov: Covariance of multivariate normal distribution
    * Datatype: list
    * Default: [[2000, 1500, 500,], [1500, 2000, 750], [500, 750, 2000]]

* capacity: Capactiy
    * Datatype: list
    * Default: [150, 300, 400]

* n_fac: The number of facilities
    * Datatype: integer
    * Default: 3

* inventory: Inventory at a facility 
    * Datatype: integer
    * Default: 0
    
Respones:
---------
* :math:`stockout_flag`: binary variable;
                  0: all facilities satisfy the demand 
                           1: at least one of the facilities did not satisfy the demand

* :math:`n_fac_stockout`: integer value;
                  the number of facilities which cannot satisfy the demand

* :math:`n_cut`: integer value; 
          the number of total demand which cannot satisfy the demand 


References:
===========
This model is adapted from the article Rengarajan, T., & Morton, D.P. (2009). Estimating the Efficient Frontier of a Probabilistic Bicriteria Model. Proceedings of the 2009 Winter Simulation Conference. 




Optimization Problem: FacilitySizingTotalCost (Problem) (facilitysizing)
========================================================

Decision Variables:
-------------------
* :math:`x`: inventory at each facility 
* :math:`capacity` 

Objectives:
-----------
Minimize the (deterministic) total cost of installing capacity at
facilities subject to a chance constraint on stockout probability.

Constraints:
------------
1 stocahstic consraint: :math:`P(Stockout)` <= :math:`epsilon`
1 deterministic constraints: :math:`x`
1 box constraint: 0 < :math:`x` < infintiy

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
