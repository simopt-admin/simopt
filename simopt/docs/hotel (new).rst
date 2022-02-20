Model: Hotel Revenue Management (HOTEL)
==========================================

Description:
------------
<A paragraph describing the stochastic model. Use math if it is helpful.>

Sources of Randomness:
----------------------
1. Stationary Poisson process with rate :math:`\lambda_i` for arrivals of each product.

Model Factors:
--------------
* num_products: Number of products: (rate, length of stay).

    * Default: 56

* lambda: Arrival rates for each product.

    * Default: 

* num_rooms: Hotel capacity.

    * Default: 100

* discount_rate: Discount rate.

    * Default: 100

* rack_rate: Rack rate (full price).

    * Default: 200

* product_incidence: Incidence matrix.

    * Default: 

* time_limit: Time after which orders of each product no longer arrive (e.g. Mon night stops at 3am Tues or t=27).

    * Default: 

* time_before: Hours before t=0 to start running (e.g. 168 means start at time -168).

    * Default: 168

* runlength: Runlength of simulation (in hours) after t=0.

    * Default: 168

* booking_limits: Booking limits.

    * Default: tuple([100 for _ in range(56)])

Respones:
---------
* revenue: Expected revenue.


References:
===========
n/a




Optimization Problem: HotelRevenue (HOTEL-1)
========================================================

Decision Variables:
-------------------
* booking_limits

Objectives:
-----------
Maximize the expected revenue.

Constraints:
------------
<Description using response names. Use math if it is helpful.>

Problem Factors:
----------------
* initial_solution: Initial solution.

  * Default: tuple([0 for _ in range(56)])
  
* budget: Max # of replications for a solver to take.

  * Default: 10000

Fixed Model Factors:
--------------------
* n/a

Starting Solution: 
------------------
* initial_solution: tuple([0 for _ in range(56)])

Random Solutions: 
------------------
Let each :math:`b_i` (element in tuple) be distributed Uniformly :math:`(0,C)`.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown