Model: Open Jackson Network
===============================================

Description:
------------
This model represents an Open Jackson Network with Poisson arrival time, exponential service time, and probabilistic routing.

Sources of Randomness:
----------------------
There are 3 sources of randomness in this model:
1. Exponential inter-arrival time of customers at each station.
2. Exponential service time of customers at each station.
3. Routing of customers at each station after service.

Model Factors:
--------------
* number_queues: The number of queues in the network.
    * Default: 3

* arrival_alphas: The rate parameter of the exponential distribution for the inter-arrival time of customers at each station.
    * Default: [1,1,1,1,1]

* service_mus: The rate parameter of exponential distribution for the service time of customers at each station.
    * Default: [2,2,2,2,2]

* routing_matrix: The routing probabilities for a customer at station i to go to service j after service. 
    The departure probability from station i is :math: `1 - \sum_{j=1}^{n} (P_{ij})`
    where n is the number of stations, and P is the routing matrix.
    * Default: [[0.1, 0.1, 0.2, 0.2, 0],
                [0.1, 0.1, 0.2, 0.2, 0],
                [0.1, 0.1, 0, 0.1, 0.3],
                [0.1, 0.1, 0.1, 0, 0.3],
                [0.1, 0.1, 0.1, 0.1, 0.2]]

* t_end: The time at which the simulation ends.
    * Default: 200

* warm_up: The time at which the warm-up period ends. Relevant only when steady_state_initialization is False.
    * Default: 100

* steady_state_initialization: Whether to initialize with queues sampled from steady state. 
    If so, we sample geometric distribution with parameter lambdas/service_mus for each queue and initialize the queues with the sample.
    * Default: True

Below Factors are only relevant when creating random instances of the Model

* density_p: The probability of an edge existing in the graph in the random instance. Higher the value, denser the graph.
    * Default: 0.5

* random_arrival_parameter: The parameter for the random arrival rate exponential distribution when creating a random instance.
    * Default: 1


Responses:
----------
* average_queue_length: The time-average queue length at each station.

References:
===========
This model is adapted from Jackson, James R. (1957).
"Networks of waiting lines". Operations Research. 4 (4): 518–521.
(doi:10.1287/opre.5.4.518)

Optimization Problem: OpenJacksonMinQueue (OPENJACKSON-1)
================================================================

Decision Variables:
-------------------
* service_mus

Objectives:
-----------
Minimize the sum of average queue length at each station.

Constraints:
------------
We require that the sum of service_mus at each station to be less than service_rates_budget.

Problem Factors:
----------------
* budget: Max # of replications for a solver to take.

  * Default: 1000

* service_rates_budget: Total budget to be allocated to service_mus_budget.

  * Default: 150

Below factors are only relevant when creating random instances of the Problem

* gamma_mean: Scale of the mean of gamma distribution when generating service rates upper bound in random instances.

  * Default: 0.5

* gamma_scale: Shape of gamma distribution when generating service rates upper bound in random instances.

  * Default: 5

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: lambdas * (service_rates_budget/sum(lambdas))

Random Solutions: 
-----------------
Sample a Dirichlet distribution that sum to service_rates_budget - sum(lambdas). Then add lambdas to the sample.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown