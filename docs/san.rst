Model: Stochastic Activity Network (SAN)
========================================

Description:
------------
Consider a stochastic activity network (SAN) with :math:`n` arcs, where each arc :math:`i`
is associated with a task with random duration :math:`X_i`. Task durations
are independent and exponentially distributed with mean :math:`\theta_i`. 
Use :math:`sum\_lb` to denote the lower bound to the sum of arc_means.
SANs are also known as PERT networks and are used in planning
large-scale projects. 

An example SAN with 13 arcs is given in the following figure:

.. image:: san.PNG
  :alt: The SAN diagram has failed to display
  :width: 500

Sources of Randomness:
----------------------
1. Task durations are exponentially distributed with mean :math:`\theta_i`.

Model Factors:
--------------
* num_nodes: Number of nodes.

    * Default: 9

* arcs: List of arcs.

    * Default: [(1, 2), (1, 3), (2, 3), (2, 4), (2, 6), (3, 6), (4, 5),
                (4, 7), (5, 6), (5, 8), (6, 9), (7, 8), (8, 9)]

* arc_means: Mean task durations for each arc.

    * Default: (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

Responses:
----------
* longest_path_length: Length/duration of the longest path.


References:
===========
This model is adapted from Avramidis, A.N., Wilson, J.R. (1996).
Integrated variance reduction strategies for simulation. *Operations Research* 44, 327-346.
(https://pubsonline.informs.org/doi/abs/10.1287/opre.44.2.327)

Optimization Problem: Minimize Longest Path Plus Penalty (SAN-1)
================================================================

Decision Variables:
-------------------
* arc_means

Objectives:
-----------
Suppose that we can select :math:`\theta_i > 0` for each :math:`i`,
but there is an associated cost. In particular, we want to minimize :math:`ET(\theta) + f(\theta)`,
where :math:`T(\theta)` is the (random) duration of the longest path from source node
to sink node (i.e. from :math:`a` to :math:`i` in the above example with 13 arcs) 
and :math:`f(\theta) = \sum_i \theta_i^{-1}` where :math:`n`
is the number of arcs.

The objective function is convex in :math:`\theta`. An IPA estimator of the gradient
is also given in the code.

Constraints:
------------
We require that :math:`\theta_i > 0` for each :math:`i`.

Problem Factors:
----------------
* budget: Max # of replications for a solver to take.

  * Default: 10000

* arc_costs: Cost associated to each arc.

  * Default: (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: (8,) * 13

Random Solutions: 
-----------------
Sample each arc mean uniformly from a lognormal distribution with 
2.5th- and 97.5th-percentiles at 0.1 and 10 respectively.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown


Optimization Problem: Minimize Longest Path with Constraint on Sum of Arc Means (SAN-2)
=======================================================================================

Decision Variables:
-------------------
* arc_means

Objectives:
-----------
Suppose that we can select :math:`\theta_i > 0` for each :math:`i`. 
Unlike the original san problem, we now want to minimize :math:`ET(\theta)`,
where :math:`T(\theta)` is the (random) duration of the longest path from source node
to sink node (i.e. from :math:`a` to :math:`i` in the above example with 13 arcs).

The objective function is convex in :math:`\theta`. An IPA estimator of the gradient
is also given in the code.

Constraints:
------------
We require that :math:`\theta_i > 0` for each :math:`i`.
Additionaly, we include another constraint to impose a lower bound to the sum of arc_means.
which is :math:`\sum_i \theta_i \geq sum\_lb`

Problem Factors:
----------------
* budget: Max # of replications for a solver to take.

  * Default: 10000

* arc_costs: Cost associated to each arc.

  * Default: (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

* sum_lb: The lower bound for sum of arc_means.

  * Default: 100.0

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: (8,) * 13

Random Solutions: 
-----------------
Use acceptance-rejection to sample each arc_mean uniformly from a lognormal distribution with 
2.5th- and 97.5th-percentiles at 0.1 and 10 respectively such that the arc_mean remains on the feasible region.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown