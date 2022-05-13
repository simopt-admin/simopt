Model: Stochastic Activity Network (SAN)
==========================================

Description:
------------
Consider the following stochastic activity network (SAN) where the arcs are labeled
from 1 through 13. (SANs are also known as PERT networks and are used in planning
large-scale projects.) Each arc :math:`i` is associated with a task with random duration
:math:`X_i`. Task durations are independent.

.. image:: san.PNG
  :alt: The SAN diagram has failed to display
  :width: 500

Sources of Randomness:
----------------------
1. Task durations are exponentially distributed with mean :math:`\theta_i`.

Model Factors:
--------------
* num_arcs: Number of arcs.

    * Default: 13

* num_nodes: Number of nodes.

    * Default: 9

* arc_means: Mean task durations for each arc.

    * Default: (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

Responses:
---------
* longest_path_length: Length/duration of the longest path.


References:
===========
This model is adapted from Avramidis, A.N., Wilson, J.R. (1996).
Integrated variance reduction strategies for simulation. *Operations Research* 44, 327-346.
(https://pubsonline.informs.org/doi/abs/10.1287/opre.44.2.327)

Optimization Problem: Minimize Longest Path Plus Penalty (SAN-1)
========================================================

Decision Variables:
-------------------
* arc_means

Objectives:
-----------
Suppose that we can select :math:`\theta_i > 0` for each :math:`i`,
but there is an associated cost. In particular, we want to minimize :math:`ET(\theta) + f(\theta)`,
where :math:`T(\theta)` is the (random) duration of the longest path from :math:`a`
to :math:`i` and :math:`f(\theta) = \sum_{i=1}^{9}\theta_i^{-1}`.

The objective function is convex in :math:`\theta`. An IPA estimator of the gradient
is also given in the code.

Constraints:
------------
We require that :math:`theta_i > 0` for each :math:`i`.

Problem Factors:
----------------
* budget: Max # of replications for a solver to take.

  * Default: 10000

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: (1,) * 13

Random Solutions: 
------------------
Sample each arc mean uniformly from a lognormal distribution with 
2.5- and 97.5-percentiles at 0.1 and 10 respectively.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown