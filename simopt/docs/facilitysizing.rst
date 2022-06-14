Model: Facility Sizing
==========================================

Description:
------------

The facility-sizing problem is formulated as follows: :math:`m` facilities are to be installed, each with capacity
:math:`xi ≥ 0, i = 1, . . . , m`. Then the random demand :math:`ξi` arrives at facility :math:`i`, with a known joint distribution
of the random vector :math:`ξ = (ξ1, . . . , ξm)`.

A realization of the demand, :math:`ξ = (ξ1, . . . , ξm)`, is said to be satisfied by the capacity :math:`x` if :math:`xi ≥ ξi, ∀i = 1, . . . , m`. 

Sources of Randomness:
----------------------
1. random demand vector :math:`ξ` follows a multivariate normal distribution and correlation coefficients :math:`ρi,j` , :math:`i != j` .

Model Factors:
--------------
* :math:`\mu`: Mean vector of the multivariate normal distribution.
    * Default: [100, 100, 100]

* :math:`\Sigma`: Variance-covariance matrix of multivariate normal distribution.
    * Default: [[2000, 1500, 500], [1500, 2000, 750], [500, 750, 2000]]

* :math:`capacity`: Inventory capacities of the facilities.
    * Default: [150, 300, 400]

* :math:`n_facility`: The number of facilities.
    * Default: 3


Respones:
---------
* :math:`stockout_flag`:
                  0: all facilities satisfy the demand 
                           1: at least one of the facilities did not satisfy the demand

* :math:`n_stockout`:
                  the number of facilities which cannot satisfy the demand

* :math:`n_cut`:
          the amount of total demand which cannot be satisfied 


References:
===========
This model is adapted from the article Rengarajan, T., & Morton, D.P. (2009). Estimating the Efficient Frontier of a Probabilistic Bicriteria Model. Proceedings of the 2009 Winter Simulation Conference. `(https://www.informs-sim.org/wsc09papers/048.pdf)`


Optimization Problem: Minimize Total Cost (FACSIZE-1)
========================================================

Our goal is to minimize the total costs of installing capacity while keeping the probability of stocking out low. 

The probability of failing to satisfy demand :math:`ξ = (ξ_1, . . . , ξ_m)` is :math:`p(x) = P(ξ !<= x)`. Let :math:`epsilon ∈ [0, 1]` be a risk-level parameter, then we obtain the probabilistic constraint:

:math:`P(ξ !<= x) ≤ epsilon`

Meanwhile, the unit cost of installing facility i is :math:`ci`, and hence the total cost is :math:`\sum_{i=1}^n c_i x_i`. 

Decision Variables:
-------------------
* :math:`capacity` 

Objectives:
-----------
Minimize the (deterministic) total cost of installing capacity.

Constraints:
------------
1 stochastic constraint: :math:`P(Stockout) <= epsilon`.
Box constraints: 0 < :math:`x_i` < infinity for all :math:`i`.

Problem Factors:
----------------
* budget: Max # of replications for a solver to take.
      * Default: 10000

* epsilon: Maximum allowed probability of stocking out.
      * Default: 0.05
  
* installation_costs: Cost to install a unit of capacity at each facility 
      * Default: (1, 1, 1)

Fixed Model Factors:
--------------------
None

Starting Solution: 
------------------
* capacity: (300, 300, 300)

Random Solutions: 
------------------
* Each facility's capacity is Uniform(0, 300).

Optimal Solution:
-----------------
None

Optimal Objective Function Value:
---------------------------------
None


Optimization Problem: Maximize Service Level (FACSIZE-2)
========================================================

Our goal is to maximize the probability of not stocking out subject to a budget
constraint on the total cost of installing capacity.

Decision Variables:
-------------------
* :math:`capacity` 

Objectives:
-----------
Maximize the probability of not stocking out.

Constraints:
------------
1 deterministic constraint: sum of facility capacity installation costs less than an installation budget.
Box constraints: 0 < :math:`x_i` < infinity for all :math:`i`.

Problem Factors:
----------------
* budget: Max # of replications for a solver to take.
      * Default: 10000

* installation_costs: Cost to install a unit of capacity at each facility.
      * Default: (1, 1, 1)

* installation_budget: Total budget for installation costs.
      * Default: 500.0

Fixed Model Factors:
--------------------
None

Starting Solution: 
------------------
* capacity: (100, 100, 100)

Random Solutions: 
------------------
* Use acceptance rejection to generate capacity vectors uniformly from space of vectors summing to less than installation budget.

Optimal Solution:
-----------------
None

Optimal Objective Function Value:
---------------------------------
None
