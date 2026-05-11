Toll Networks
=======================
See the :mod:'simopt.models.tollnw' module for API details.

Model: Toll Road Improvements in a Newtork
----------------------------------------

Description
^^^^^^^^^^^

This model represents a connected directed graph :math:`G= (V,E)` describing a network 
of toll roads. Customers wishing to travel from point :math:`i` to point :math:`j` 
arrive according to a Poisson process with rate :math:`lambda_{ij}`. 

Each directed edge represents a toll road with price :math:`p_{ij}`. Travel times on each
road are triangularly distributed with route-specific parameters. The network operates as
an open Jackson network with interconnected queues.

Each road has the capacity for one vehicle at a time. If a vehicle is currently traveling
on a road, arriving vehicles form a queue and wait until the road becomes available.

The operator can choose to invest in road improvements. If :math:`x_{ij}` hundred dollars 
is spent improving road :math:`(i,j)`, the resulting mode of the triangular travel-time 
distribution becomes:

:math:`a_{ij} + (b_{ij} - a_{ij}) \exp(-x_{ij})`

where:
- :math:`a_{ij}` is the minimum travel time
- :math:`b_{ij}` is the maximum travel time


If no investment is made (i.e. :math:`x_{ij} = 0`), the mode equals :math:`b_{ij}`. 

The operator seeks to maximize expected profit over a finite horizon :math:`T`:

:math:`\mathbb{E}[\text{Profit}] = \sum_{(i,j) \in E} p_{ij} \mathbb{E}[N_{ij}(T)] - \sum_{(i,j) \in E} x_{ij}`

where:
:math:`N_{ij}(T)` is the number of trips completed on road :math:`(i,j)` before time :math:`T`.



Sources of Randomness
^^^^^^^^^^^^^^^^^^^^^

There are three source of randomness:

1. External arrivals modeled as Poisson processes with rates :math:`lambda_{ij}`
2. Internal routing decisions governed by routing matrix :math:`R`
3. Travel times following triangular distribution


Model Factors
^^^^^^^^^^^^^

- adjacency_matrix: graph structure describing road connectivity
- lambda_ij: external Poisson arrival rates
    - default: :math:`\lambda_{ij} = i + j` for :math:`i \neg j`, :math:`i, j \in \{1,2,3\}`
- routing_matrix: internal routing matrix with exit probabilities
- pij: toll/amount charged per completed trips
    - default: 1 for all roads
- aij: minimum travel time of triangular distribution
    - default: 0
- bij: maximum travel time of triangular distribution
    - default: 1
- investment_levels: investment decisions :math:`x_{ij}`
- T: length of operating horizon
    - default: 96 (12 hours)
- time_unit: one unit equals 10 minutes


Recommended Parameter Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`G` be a complete graph with three vertices:

..math::

    A = 
    \begin{bmatrix}
    0 & 1 & 1 \\
    1 & 0 & 1 \\
    1 & 1 & 0
    \end{bmatrix}

Routing Matrix:

..math::

    R =
    \begin{bmatrix}
    0 & 1/3 & 1/3 & 1/3
    1/2 & 0 & 1/4 & 1/4
    1/5 & 1/5 & 0 & 3/5
    \end{bmatrix}

The last column represents exiting the network.

Responses
^^^^^^^^^

- total_profit: expected profit over horizon :math:`T`
- total_trips_completed: total completed trips across all roads
- road_trip_counts: vector of :math:`N_{ij}(T)` values
- average_queue_lengths: time-averaged queue lengths per road

References
^^^^^^^^^^

Eckman, D. (2017). Toll Road Improvements in a Network.

Optimization Problem: Maximize Expected Profit (TOLLNETWORK-1)
--------------------------------------------------------------

Decision Variables
^^^^^^^^^^^^^^^^^^

- investment_levels: :math:`x_{ij}` (continuous, :math:`\ge 0`)

Objective
^^^^^^^^^^

Maximize expected profit over horizon :math:`T`.

Constraints
^^^^^^^^^^^

- :math:`x_{ij} \ge 0` for all roads
- Continuous decision Variables xij>=0

Problem Factors
^^^^^^^^^^^^^^^

- Budget: maximum number of replications for solver
    - default: 1000

Fixed Model Factors
^^^^^^^^^^^^^^^^^^^

- aij = 0
- bij = 1
- pij = 1
- T = 96

Starting Solution
^^^^^^^^^^^^^^^^^

- :math:`x_{ij} = 1` for all roads

Random Solutions
^^^^^^^^^^^^^^^^

Generate continuous nonnegative investment vectors over all roads

Optimal Solution
^^^^^^^^^^^^^^^^

Unknown

Optimal Objective Function Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unknown