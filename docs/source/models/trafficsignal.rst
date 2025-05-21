Traffic Signal Configuration
============================

See the :mod:`simopt.models.trafficsignal` module for API details.

Model: Traffic Signal
---------------------

Description
^^^^^^^^^^^

This model simulates a traffic system composed of arrival and departure nodes, direction and bidirectional roads, and a series of intersections.
A visual representation of this simplified traffic light roadmap, illustrating the flow of traffic within this structured network, is provided in this document.

The layout of the traffic network is shown below:

.. image:: _static/trafficlight_roadmap.png
   :alt: TrafficLight Roadmap
   :align: center

The graph depicts a system with:

* **6 Allowable Starting Nodes:** These nodes represent entry points into the road network: **N1, N2, E2, S2, S1, and W1**.
* **6 Allowable Ending Nodes:** These nodes represent exit points from the road network: **N1, N2, E1, S2, S1, and W2**.
* **4 Intersections:** These central nodes serve as critical junctions where traffic flow can change direction: **A, B, C, and D**.

The road network consists of two distinct types of roads:

* **Artery Roads (Vertical):** Represented by **red lines**, these **bidirectional** roads handle the majority of traffic flow and are primarily oriented vertically.
* **Vein Roads (Horizontal):** Represented by **blue lines**, these **unidirectional** roads are smaller in capacity and primarily oriented horizontally.

The arrows on the roads indicate the permitted direction of traffic flow.
This roadmap is designed to help understand the fundamental structure and
traffic patterns within a simplified urban environment.

.. note:: 
    Cars are not allowed to turn left at intersections due to technical constraints of the model.

Each car enters the system through one of six designated arrival nodes.
The probability of selecting a given arrival node is proportional to its lambda value:

.. math::
    :label: eq_lambda_selection

    \frac{\lambda_i}{\sum_{j=0}^{7} \lambda_j}
    \text{ where } \lambda_i \text{ is the arrival rate for node i.}

The ``lambdas`` parameter defines these arrival rates.
It is a list of 8 values specifying the Poisson rate parameters (`\lambda`) for car arrivals at each peripheral node, listed in the following order:

.. table:: Lambda Index Mappings
    :align: center

    +-------+------+----------+-------------------------+
    | Index | Node | Entrance | Default :math:`\lambda` |
    +=======+======+==========+=========================+
    | 0     | N1   | ✓        | 2                       |
    +-------+------+----------+-------------------------+
    | 1     | N2   | ✓        | 2                       |
    +-------+------+----------+-------------------------+
    | 2     | E1   |          | 0                       |
    +-------+------+----------+-------------------------+
    | 3     | E2   | ✓        | 1                       |
    +-------+------+----------+-------------------------+
    | 4     | S2   | ✓        | 2                       |
    +-------+------+----------+-------------------------+
    | 5     | S1   | ✓        | 2                       |
    +-------+------+----------+-------------------------+
    | 6     | W2   |          | 0                       |
    +-------+------+----------+-------------------------+
    | 7     | W1   | ✓        | 1                       |
    +-------+------+----------+-------------------------+

Cars are not allowed to spawn at exit-only nodes (E1 and W2), so ``lambdas[2]`` and ``lambdas[6]`` must be set to ``0``.
Additionally, the car generation rates at vein nodes (E2 and W1) must not exceed the arrival rates at any of the artery entry points (N1, N2, S1, S2).

For each arriving car, the lambda value associated with the selected entry node determines the distribution of interarrival times.
Once a car enters the system, it is randomly assigned a destination node based on a weighted transition matrix derived from the ``transition_probs`` factor.

Each entry in this matrix represents a *relative weight* indicating how likely a car is to travel from one node to another.
Larger weights increase the chances of selecting that path, but the values do not need to sum to 1.
These weights are normalized internally during destination selection.

The symbolic node weight matrix is shown below:

.. table:: Node Transition Weight Matrix (Unnormalized)
   :align: center

   +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
   | From \\ To  | N1          | N2          | E1          | E2          | S2          | S1          | W2          | W1          |
   +=============+=============+=============+=============+=============+=============+=============+=============+=============+
   | N1          | X           | X           | X           | X           | X           | :math:`P_0` | :math:`P_1` | X           |
   +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
   | N2          | :math:`P_2` | X           | :math:`P_3` | X           | :math:`P_2` | X           | :math:`P_3` | X           |
   +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
   | E1          | X           | X           | X           | X           | X           | X           | X           | X           |
   +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
   | E2          | :math:`P_4` | :math:`P_4` | :math:`P_5` | X           | :math:`P_4` | X           | :math:`P_6` | X           |
   +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
   | S2          | X           | :math:`P_0` | :math:`P_1` | X           | X           | X           | X           | X           |
   +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
   | S1          | :math:`P_2` | X           | :math:`P_3` | X           | :math:`P_2` | X           | :math:`P_3` | X           |
   +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
   | W2          | X           | X           | X           | X           | X           | X           | X           | X           |
   +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
   | W1          | :math:`P_4` | X           | :math:`P_6` | X           | :math:`P_4` | :math:`P_4` | :math:`P_5` | X           |
   +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+

.. note:: 
    Cells marked with ``X`` represent disallowed transitions that cannot occur in the simulation.
    Each row is internally normalized to sum to 1 during routing.

The symbolic parameters :math:`P_0` through :math:`P_6` correspond to entries in the ``transition_probs`` list.
These values act as **weights** rather than strict probabilities, and are normalized during destination selection to ensure proper routing behavior.
Specifically, :math:`P_i` refers to ``transition_probs[i]``.

For example, given the default values of ``transition_probs = [7, 3, 3, 2, 5, 2, 3]``, the resulting transition matrix is:

.. table:: Default Transition Matrix (Populated from ``transition_probs``)
   :align: center

   +------------+------+-----+-----+-----+-----+-----+-----+-----+
   | From \\ To | N1   | N2  | E1  | E2  | S2  | S1  | W2  | W1  |
   +============+======+=====+=====+=====+=====+=====+=====+=====+
   | N1         | X    | X   | X   | X   | X   | 70% | 30% | X   |
   +------------+------+-----+-----+-----+-----+-----+-----+-----+
   | N2         | 30%  | X   | 20% | X   | 30% | X   | 20% | X   |
   +------------+------+-----+-----+-----+-----+-----+-----+-----+
   | E1         | X    | X   | X   | X   | X   | X   | X   | X   |
   +------------+------+-----+-----+-----+-----+-----+-----+-----+
   | E2         | 25%  | 25% | 10% | X   | 25% | X   | 15% | X   |
   +------------+------+-----+-----+-----+-----+-----+-----+-----+
   | S2         | X    | 70% | 30% | X   | X   | X   | X   | X   |
   +------------+------+-----+-----+-----+-----+-----+-----+-----+
   | S1         | 30%  | X   | 20% | X   | 30% | X   | 20% | X   |
   +------------+------+-----+-----+-----+-----+-----+-----+-----+
   | W2         | X    | X   | X   | X   | X   | X   | X   | X   |
   +------------+------+-----+-----+-----+-----+-----+-----+-----+
   | W1         | 25%  | X   | 15% | X   | 25% | 25% | 10% | X   |
   +------------+------+-----+-----+-----+-----+-----+-----+-----+

Each vehicle finds the shortest available path to its destination using the current road network. The traffic system opens at 8:00 AM and closes at 10:00 AM. Time is measured in seconds. When the system closes, any remaining cars in the queue exit immediately.

Sources of Randomness
^^^^^^^^^^^^^^^^^^^^^

There are 3 sources of randomness in this model:

1. Randomized selection of the **arrival node** for each car (see Equation :eq:`eq_lambda_selection`).
2. The arrival time of the **first car** is fixed at 1 second. The arrival times of subsequent cars follow an exponential distribution with rate parameter :math:`\lambda_i`, where *i* is the index of the arrival node selected for the **previous** car.
3. The probability for each node to selected as the destination for cars from different arrival nodes (see above).

Model Factors
^^^^^^^^^^^^^

* lambdas: Rate parameter of the time interval distribution, in seconds, for generating each car.  
    * Default: [2, 2, 0, 1, 2, 2, 0, 1]
* runtime: The number of seconds that the traffic model runs.
    * Default: 7200
* numintersections: The number of intersections in the traffic model.
    * Default: 4
* decision_vector: Delay, in seconds, in light schedule based on distance from first intersection.
    * Default: [1, 2, 3]
* speed: Constant speed in meter/second for the cars.
    * Default: 5
* carlength: Length in meters of each car.
    * Default: 4.5
* reaction: Reaction time in seconds of cars in queue.
    * Default: 0.1
* transition_probs: The transition probability of a car end at each point from their current starting point.
    * Default: [7, 3, 3, 2, 5, 2, 3]
* pause: The pause in seconds before move on a green light.
    * Default: 0.1
* car_distance: The distance between cars.
    * Default: 0.5
* length_arteries: The length in meters of artery roads between each intersection or node.
    * Default: 100
* length_veins: The length in meters of vein roads between each intersection or node.
    * Default: 100
* redlight_arteries : The length of redlight duration of artery roads in each intersection.
    * Default: [10, 10, 10, 10]
* redlight_veins : The length of redlight duration of vein roads in each intersection.
    * Default : [20, 20, 20, 20]

Responses
^^^^^^^^^

* WaitingTime: The average queuing time of the number of cars in the model
* SystemTime: The average time of the number of cars arriving the destination in the model
* AvgQueueLen: The average queue length of the number of cars in the model 
* OverflowPercentage: The ratio of overflow time to total system time.
* OverflowPercentageOver51: Whether the overflow time is larger than 51% of the total system time.

References
^^^^^^^^^^

This model are adapted from the following articles: 

Ito, H., K. Tsutsumida., T. Matsubayashi., T, Kurashima., and H, Toda. (2019). Coordinated traffic signal control via bayesian optimization for hierarchical conditional spaces. Proceedings of the 2019 Winter Simulation Conference, 3645–3656.

Osorio, C., and L, Chong. (2012). An efficient simulation-based optimization algorithm for large-scale transportation problems. Proceedings of the 2012 Winter Simulation Conference, 1–11.

Optimization Problem: Minimize Waiting Time (MinWaitingTime-1)
--------------------------------------------------------------

Decision Variables
^^^^^^^^^^^^^^^^^^

* decision_vector

Objectives
^^^^^^^^^^

Minimize average queueing time for each car in the system.

Constraints
^^^^^^^^^^^

* :math:`lambda_i`> 0 for each i.

Problem Factors
^^^^^^^^^^^^^^^

* Budget: Max # of replications for a solver to take.
    * Default: 1000

Fixed Model Factors
^^^^^^^^^^^^^^^^^^^

N/A

Starting Solution
^^^^^^^^^^^^^^^^^

* decision_vector = [1, 2, 3]

Random Solutions
^^^^^^^^^^^^^^^^

Generate a solution from a space of vectors of length 3.

Optimal Solution
^^^^^^^^^^^^^^^^

Unknown.

Optimal Objective Function Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unknown.