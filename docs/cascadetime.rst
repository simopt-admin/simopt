Model: Non-progressive Cascade Process (CASCADETIME)
====================================================================

Description:
------------
In each replication, we simulate non-progressive cascade(s) process within a finite time :math:`\tau`. The non-progressive cascade process is similar
to the progressive one (see the .rst file for CASCADE) but within a finite time horizon and including the probability of becoming inactive 
at each time step :math:`\beta_v` for node :math:`v \in V`. Besides, the initial node activation probabilities become :math:`u_v^t \in [0,1] \forall v \in V,
t = \{1, 2, \ldots, \tau\}`, which is time-dependent.

According to Kempe et al. 2003, we can reduce the non-progressive cascade progress to the progressive one, The reduction can be achieved by transforming the original
graph :math:`G` to a layered graph :math:`H = (V', E')` on :math:`\tau \cdot |V|` nodes, where :math:`V'` consists of :math:`v_t^i` for each node :math:`i` in :math:`G`
and each time-step :math:`t \leq \tau`. :math:`E'` is obtained by connecting each node in :math:`H` with itself and its neighbors in :math:`G` indexed by the previous time step.
The edge weight is specified as follows: :math:`p(v_t^i, v_{t+1}^j) = p_{ij}, p(v_t^i, v_{t+1}^i) = 1 - \beta_i`

Nonetheless, we realize that simply using :math:`u_v^t` as decision variables can make these two problems become extremely high-dimensional and hard to solve. 
To reduce the dimension, we adopt an approach for "grouping" the variables together: 

given :math:`K`, a constant number of groups, we divide the node set :math:`V` into :math:`K` node subsets 
:math:`\{V_k'\}_{k = \{1, 2, \ldots, K\}}`. The decision variables become :math:`u_k^t`, which represents the probability of activating any node in :math:`V_k'` at time step :math:`t`. 
Notice that :math:`u_v^t = u_k^t` for :math:`v \in V_k`.


Note: the input graph :math:`G` is pre-generated through a random Gnp graph with :math:`n = 30` and :math:`p = 0.4`. We further transform 
:math:`G` to a directed acyclic graph by keeping only edges directing from lower indices to higher indices.
The resulting graph is stored in **DAG.graphml**.

Sources of Randomness:
----------------------
1. A node :math:`v_t` would be activated initially following a Bernoulli distribution with :math:`p = u_v^t`.

2. An edge :math:`e` would be activated following a Bernoulli distribution with :math:`p = p_e`, where :math:`p_e` corresponds to the edge weight.

Model Factors:
--------------
* num_subgraph: Number of subgraphs to generate.

    * Default: 10

* num_group (:math:`K`): Number of node groups.

    * Default: 3

* init_prob: Probability of initiating the nodes for each group at each time step.

    * Default: np.array([0.05, 0, 0] * 10)

* T: Number of time steps for the cascade process.

    * Default: 10

* beta: Probability of de-activating a node at each time step.

    * Default: 0.4

Respones:
---------
* mean_num_activated: The average total number of activated nodes among all subgraphs.

* mean_num_activated_end: Mean number of activated nodes at the end.


References:
===========
Kempe, D., Kleinberg, J., & Tardos, É. (2003, August). Maximizing the spread of influence through a social network. 
In Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 137-146).

Sheldon, D., Dilkina, B., Elmachtoub, A. N., Finseth, R., Sabharwal, A., Conrad, J., ... & Vaughan, W. (2012). 
Maximizing the spread of cascades using network design. arXiv preprint arXiv:1203.3514.


Optimization Problem: Maximize Total Activated Nodes (CASCADETIME-1)
=====================================================================

Decision Variables:
-------------------
* init_prob (:math:`u_k^t`)

Objectives:
-----------
Maximize the expected total number of activated nodes throughout the entire time period.

Constraints:
------------
All decision variables should be between 0 and 1.
The expected total activation cost should be within a cost budget :math:`B`.
:math:`\sum_{t \leq \tau} \sum_{k \in K} \sum_{v \in V_k} c_v \cdot u_k^t \leq B`

Problem Factors:
----------------
* budget: Max # of replications for a solver to take

  * Default: 10000

* B: Budget for the activation costs

  * Default: 500

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: tuple(np.array([0.01, 0.01, 0.01] * 10))

Random Solutions: 
-----------------
* :math:`u_k^t` is sampled uniformly from convex shapes defined by linear constraints. We adopt the hit-and-run algorithm, a Markov Chain Monte Carlo method.

* The **get_multiple_random_solution** function allows for more efficiently generating multiple random solutions using hit-and-run.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown


Optimization Problem: Maximize End Activated Nodes (CASCADETIME-2)
=====================================================================

Decision Variables:
-------------------
* init_prob (:math:`u_k^t`)

Objectives:
-----------
Maximize the expected number of activated nodes at the end of the time period.

Constraints:
------------
All decision variables should be between 0 and 1.
The expected total activation cost should be within a cost budget :math:`B`.
:math:`\sum_{t \leq \tau} \sum_{k \in K} \sum_{v \in V_k} c_v \cdot u_k^t \leq B`

Problem Factors:
----------------
* budget: Max # of replications for a solver to take

  * Default: 10000

* B: Budget for the activation costs

  * Default: 500

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: tuple(np.array([0.01, 0.01, 0.01] * 10))

Random Solutions: 
-----------------
* :math:`u_k^t` is sampled uniformly from convex shapes defined by linear constraints. We adopt the hit-and-run algorithm, a Markov Chain Monte Carlo method.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown
