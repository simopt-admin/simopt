Model: Traffic Control (trafficlight)
==========================================

Description:
------------
A model that simulates a series of intersections and their light schedules. 
As cars travel through the system, their waiting time is tracked.

This model simulates a number of cars traveling through a series of intersections. These intersections are on
their own schedules, and their interval is defined at the beginning of a simulation run. As the model generates
cars, they travel through the system according to the intersections' light schedules and to their own 
speed and capability.

Ultimately, many different factors of the model can be changed in order to see the circumstances in which average
waiting time for any given car is the lowest it can be. This will create safer roads and happier drivers.

Sources of Randomness:
----------------------
There are three sources of randomness in this model. All are created using the mrg32k3a random number generator.

The first two are modeled with uniform distributions and represent the start and end points of a car's path.

* Start and End Points: Random integer from 0 to the number of intersections

The third is modeled with an exponential distribution and represents the interarrival times of the cars.

* Interarrival Times: Exponentially distributed random variable with parameter lambda.

Model Factors:
--------------
* lambda: Rate parameter of interarrival time distribution. (float)

    * Default: 0.5

* runtime: Total time that the simulation runs. (float)

    * Default: 50

* numintersections: Number of intersections. (int)

    * Default: 4

* interval: Interval of time between light changes. (float)

    * Default: 5    

* offset: Delay in light schedule based on distance from first intersection. (list)

    * Default: [0, 0, 0, 0]

* speed: Constant that represents speed of cars when moving. (float)

    * Default: 2.5  

* distance: Distance of travel between roads. (float)

    * Default: 5

* carlength: Length of each car. (float)

    * Default: 1  

* reaction: Reaction time of cars in queue. (float)

    * Default: 0.1

Responses:
---------
* WaitingTime: The average time a car sits in a queue.


Optimization Problem: Minimum Waiting Time in System (MinWaitingTime)
========================================================

Decision Variables:
-------------------
* offset

Objectives:
-----------
Minimize the average waiting time (WaitingTime) in system for the cars.

Constraints:
------------
All values in offset should be greater than 0.

Problem Factors:
----------------
* initial_solution: Initial solution from which the solver starts.

  * Default: (0, 0, 0, 0)

* budget: Max # of replications for a solver to take.

  * Default: 100


Starting Solution: 
------------------
* offset: [0, 0, 0, 0]

Optimal Solution:
-----------------
 Unknown

Optimal Objective Function Value:
---------------------------------
Unknown


...