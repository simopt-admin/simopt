Model: Restaurant Table Management (TABLEALLOCATION)
====================================================

Description:
------------
Floor space in a restaurant is allocated to N different sizes of tables, with capacities
:math:`c = [c_1, c_2,..., c_N ], c_i \in Z_+^{n}`. A table of capacity :math:`c_i` can seat 
:math:`c_i` customers or fewer. 
The restaurant can seat a maximum of :math:`K` customers at a time, 
regardless of table allocation, and is open for :math:`T` consecutive hours. 
The decision variable is the vector :math:`x` representing
the number of tables allocated to each size.

Customers arrive in groups of size :math:`j \in \{1, ..., max_i(c_i)\}` according to a homogeneous 
Poisson process with rate :math:`\lambda_j`. A group of customers is seated at the smallest possible 
available table. If there are no available tables large enough, the group of customers 
leaves immediately. Service time per group is exponential and revenue per group is fixed.

Sources of Randomness:
----------------------
Groups of customers arrive according to a homogenrous Poisson process. Group size is randomly generated 
with probability proportional to each group's average arrival rate. Service time per group is exponential.

Model Factors:
--------------
* n_hour: Number of hours to simulate.

    * Default: 3

* capacity: Maximum total capacity.

    * Default: 80

* table_cap: Capacity of each type of table.
  
    * Default: [2, 4, 6, 8]

* lambda: Average number of arrivals per hour.

    * Default: [3, 6, 3, 3, 2, 4/3, 6/5, 1]

* service_time_means: Mean service time in minutes.
  
    * Default: [20, 25, 30, 35, 40, 45, 50, 60]

* table_revenue: Revenue earned for each group size.

    * Default: [15, 30, 45, 60, 75, 90, 105, 120]

* num_tables: Number of tables of each capacity.

    * Default: [10,5,4,2]

Responses:
----------
* total_revenue: Total revenue earned over the simulation period.

* service_rate: Fraction of customer arrivals that are seated.

References:
===========
Original author of this problem is Bryan Chong (March 10, 2015).


Optimization Problem: Maximize Revenue (TABLEALLOCATION-1)
==========================================================

Decision Variables:
-------------------
* num_tables

Objectives:
-----------
Maximize the total expected revenue for a restaurant operation.

Constraints:
------------
Number of seats in the restaurant :math:`<= capacity`.

Problem Factors:
----------------
* budget: Max # of replications for a solver to take.

  * Default: 1000

Fixed Model Factors:
--------------------
N/A

Starting Solution: 
------------------
* num_tables: [10, 5, 4, 2]. Corresponds to 10 tables of size 2, 5 tables of size 4, 4 tables of size 6, and 2 tables of size 8.

Random Solutions: 
-----------------
Distribute total capacity uniformly across table sizes. If the remaining capacity is smaller than the smallest table size, keep the last table allocation as a starting solution.

Optimal Solution:
-----------------
Unknown.

Optimal Objective Function Value:
---------------------------------
Unknown.