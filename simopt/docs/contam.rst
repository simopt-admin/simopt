Model: Contamination Control Problem (CONTAM)
==========================================

Description:
------------
Consider a food supply chain consisting of :math:`n` stages. Suppose there exists
a possibility that pathogenic microorganisms and other poisonous elements contaminate
some fraction of the food supply at each stage. Specifically, let the growth rate
of contamination at the stage :math:`i` of the chain be denoted by the random variable
:math:`\Lambda_i`, :math:`0 \leq \Lambda_i \leq 1` for :math:`i = 1, 2, ..., n`. If
a prevention effort is made at the stage :math:`i`, the contamination decreases by
the random rate :math:`\Gamma_i`, :math:`0 \leq \Gamma_i \leq 1` with associated
prevention cost :math:`c_i`. Let the prevention decision variable :math:`u_i = 1`
if a prevention measure is executed at the stage :math:`i`, and :math:`u_i = 0` otherwise.


Sources of Randomness:
----------------------
1. Contamination rate :math:`\Lambda_i ~ Beta(1, \frac{17}{3})` for :math:`i = 1, 2, ..., n`;
2. Restoration rate :math:`\Gamma_i ~ Beta(1, \frac{3}{7})` for :math:`i = 1, 2, ..., n`;

Model Factors:
--------------
* contam_rate_alpha: Alpha parameter of beta distribution for growth rate of contamination at each stage.

    * Default: 1.0

* contam_rate_beta: Beta parameter of beta distribution for growth rate of contamination at each stage.

    * Default: 17/3

* restore_rate_alpha: Alpha parameter of beta distribution for rate that contamination decreases by after prevention effort.

    * Default: 1.0

* restore_rate_beta: Beta parameter of beta distribution for rate that contamination decreases by after prevention effort.

    * Default: 3/7

* initial_rate_alpha: Alpha parameter of beta distribution for initial contamination fraction.

    * Default: 1.0

* initial_rate_beta: Beta parameter of beta distribution for initial contamination fraction.

    * Default: 30.0

* stages: Stage of food supply chain.

    * Default: 5

* prev_decision: Prevention decision.

    * Default: (0, 0, 0, 0, 0)

Responses:
---------
* level: A list of contamination levels over time.


References:
===========
This model is adapted from the article "Contamination control in food supply chain" [1].
Prepared by Kaeyoung Shin and Raghu Pasupathy of Virginia Tech, 12/18/2010.

[1] Y. Hu, J. Hu, Y. Xu, and F. Wang. Contamination control in food supply
chain. In *Proceedings of the 2010 Winter Simulation Conference*, 2010.
https://dl.acm.org/doi/abs/10.5555/2433508.2433840



Optimization Problem: ContaminationTotalCostDisc (CONTAM-1)
========================================================

Decision Variables:
-------------------
* prev_decision

Objectives:
-----------
Minimize the (deterministic) total cost of prevention efforts (prev_cost * prev_decision).

.. image:: contam.PNG
  :alt: The CONTAM formulation has failed to display
  :width: 400

Constraints:
------------
The contaminated fraction :math:`X_i` at the stage :math:`i`
should not exceed a pre-specified upper limit :math:`p_i` with probability at least :math:`1 - \epsilon_i`. prev_decision is discrete. (See above.)

Problem Factors:
----------------
* initial_solution: Initial solution.

  * Default: (1, 1, 1, 1, 1)
  
* budget: Max # of replications for a solver to take.

  * Default: 10000

* prev_cost: Cost of prevention.

  * Default: [1, 1, 1, 1, 1]

* error_prob: Error probability.

  * Default: [0.2, 0.2, 0.2, 0.2, 0.2]

* upper_thres: Upper limit of amount of contamination.

  * Default: [0.1, 0.1, 0.1, 0.1, 0.1]

Fixed Model Factors:
--------------------
* n/a

Starting Solution: 
------------------
* initial_solution: (1, 1, 1, 1, 1)

Random Solutions: 
------------------
Generate a tuple of 0s and 1s with equal probability.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown


Optimization Problem: ContaminationTotalCostCont (CONTAM-2)
========================================================

Decision Variables:
-------------------
* prev_decision

Objectives:
-----------
Minimize the (deterministic) total cost of prevention efforts (prev_cost * prev_decision).

Constraints:
------------
The contaminated fraction :math:`X_i` at the stage :math:`i`
should not exceed a pre-specified upper limit :math:`p_i` with probability at least :math:`1 - \epsilon_i`. prev_decision is continuous.

Problem Factors:
----------------
* initial_solution: Initial solution.

  * Default: (1, 1, 1, 1, 1)
  
* budget: Max # of replications for a solver to take.

  * Default: 10000

* prev_cost: Cost of prevention.

  * Default: [1, 1, 1, 1, 1]

* error_prob: Error probability.

  * Default: [0.2, 0.2, 0.2, 0.2, 0.2]

* upper_thres: Upper limit of amount of contamination.

  * Default: [0.1, 0.1, 0.1, 0.1, 0.1]

Fixed Model Factors:
--------------------
* n/a

Starting Solution: 
------------------
* initial_solution: (1, 1, 1, 1, 1)

Random Solutions: 
------------------
Generate a tuple of standard uniform solutions.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown