Network Queueing System Design
==============================

See the :mod:`simopt.models.network` module for API details.

Model: Network Queueing System Design (Network)
-----------------------------------------------

Description
^^^^^^^^^^^

This model represents a communication system where arriving messages are routed through a network based on chosen routing percentages. There are :math:`N` random messages that arrive following a Poisson process with a rate of :math:`λ` that need to go to a particular destination, and there are :math:`n` networks available to process these messages. When a message arrives there is a :math:`p_i%` chance that it will be processed by network :math:`i`. The per message processing cost is :math:`c_1, c_2,..., c_i` depending on which network the message is routed through. It also takes time for a message to go through a network. This transit time is denoted by :math:`S_i` for each network :math:`i` and :math:`S_i` follows a triangular distribution with lower limit :math:`a_i`, upper limit :math:`b_i`, and mode :math:`c_i`. Each network behaves like a single-server queue with first-in-first-out service discipline.There is a cost for the length of time a message spends in network :math:`i` measured by :math:`c_i` per unit of time.

Sources of Randomness
^^^^^^^^^^^^^^^^^^^^^

1. Interarrival time of a message.
2. The network a message is routed to. 
3. The transit time of a message; depends on the network.

Model Factors
^^^^^^^^^^^^^

* process_prob: Probability that a message will go through a particular network i.
    * Default: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
* cost_process: Message processing cost of network i.
    * Default: [1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10]
* cost_time: Cost for the length of time a message spends in a network i per unit of time.
    * Default: [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
* mode_transit_time: Mode time of transit for network i following a triangular distribution.
    * Default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
* lower_limits_transit_time: Lower limits for the triangular distribution for the transit time.
    * Default: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
* upper_limits_transit_time: Upper limits for the triangular distribution for the transit time.
    * Default: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
* arrival_rate: Arrival rate of messages following a Poisson process.
    * Default: 1
* n_messages: Number of messages that arrive and need to be routed.
    * Default: 1000
* n_networks: Number of networks.
    * Default: 10

Responses
^^^^^^^^^

* total_cost: Total cost spent to route and process all messages.

References
^^^^^^^^^^

Barton, R. R., & Meckesheimer, M. (2006). Metamodel-Based Simulation Optimization.
S.G. Henderson and B.L. Nelson (Eds.), Handbook in OR & MS, Vol. 13.

Optimization Problem: Minimize Total Cost (NETWORK-1)
-----------------------------------------------------

Decision Variables
^^^^^^^^^^^^^^^^^^

* process_prob

Objectives
^^^^^^^^^^

The objective is to minimize total costs, the sum of time costs and network costs for all messages.

Constraints
^^^^^^^^^^^

* :math:`0 \le p_i \le 1` for all :math:`i = 1, 2, ..., n`
* :math:`\sum_{i=1}^n p_i = 1`

:math:`p_1, p_2,..., p_n \in [0, 1]` are the routing probabilities.

Problem Factors
^^^^^^^^^^^^^^^

* initial_solution: Initial solution from which solvers start.
    * Default: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
* budget: Max # of replications for a solver to take.
    * Default: 1000

Fixed Model Factors
^^^^^^^^^^^^^^^^^^^

N/A

Starting Solution
^^^^^^^^^^^^^^^^^

* process_prob: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

Random Solutions
^^^^^^^^^^^^^^^^

Generate allocations uniformly at random from the set of vectors of length equal to the number of networks whose values are greater than 0 and less than 1.

Optimal Solution
^^^^^^^^^^^^^^^^

Unknown

Optimal Objective Function Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unknown
