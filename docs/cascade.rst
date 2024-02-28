Model: Progressive Cascade Process (CASCADE)
====================================================================

Description:
------------
Consider a network represented by a directed graph :math:`G = (V, E)`. The graph :math:`G` contains the edge weight (activation probabilities) 
:math:`p_e` for :math:`e \in E`, the node activation costs :math:`c_v` for :math:`v \in V`, 
and the initial node activation probabilities :math:`u_v \in [0, 1]` for :math:`v \in V`.
An edge :math:`e = (i,j)` is activated suggests that if node :math:`i` is activated, so will node :math:`j`.

In each replication, we simulate progressive cascade(s) of the network following the independent cascade model as described in Kempe et al. 2003. 
We first generate the activated edges, which form a subgraph of the original network; then, we generate the
nodes that are activated initially. Lastly, we count the total number of nodes in the connected components of the initially activated nodes.

Note: the input graph :math:`G` is pre-generated through a random Gnp graph with :math:`n = 30` and :math:`p = 0.4`. We further transform 
:math:`G` to a directed acyclic graph by keeping only edges directing from lower indices to higher indices.
The resulting graph is stored in **DAG.graphml**.

Sources of Randomness:
----------------------
1. A node :math:`v` would be activated initially following a Bernoulli distribution with :math:`p = u_v`.

2. An edge :math:`e` would be activated following a Bernoulli distribution with :math:`p = p_e`.

Model Factors:
--------------
* num_subgraph: Number of subgraphs to generate.

    * Default: 10

* init_prob: Probability of initiating the nodes.

    * Default: [0.1, ..., 0.1]


Respones:
---------
* mean_num_activated: The average total number of activated nodes among all subgraphs.


References:
===========
Kempe, D., Kleinberg, J., & Tardos, É. (2003, August). Maximizing the spread of influence through a social network. 
In Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 137-146).

Optimization Problem: Maximize Number of Activated Nodes (CASCADE-1)
====================================================================

Decision Variables:
-------------------
* init_prob (:math:`u_v`)

Objectives:
-----------
Maximize the expected number of activated nodes.

Constraints:
------------
All decision variables should be between 0 and 1.
The expected total activation cost should be within a cost budget :math:`B`.

:math:`\sum c_v \cdot u_v \leq B`

Problem Factors:
----------------
* budget: Max # of replications for a solver to take

  * Default: 10000

* B: Budget for the activation costs

  * Default: 200

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: [0.1, ..., 0.1]

Random Solutions: 
-----------------
* :math:`u_v` is sampled uniformly from convex shapes defined by linear constraints. We adopt the hit-and-run algorithm, a Markov Chain Monte Carlo method.

* The **get_multiple_random_solution** function allows for more efficiently generating multiple random solutions using hit-and-run.


Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown
