
Restaurant Table Management
==================================================

**Problem Description**

Floor space in a restaurant is allocated to N different sizes of tables, with capacities
:math:`c = [c_1, c_2,..., c_N ], c_i \in Z_+^{n}`. A table of capacity :math:`c_i` can seat 
:math:`c_i` or less customers. 
The restaurant can seat a maximum of :math:`K` customers at a time, 
regardless of table allocation, and is open for :math:`T` consecutive hours. 
The decision variable is the vector :math:`x \in (Z_+ ∪ \{0\})^N` , 
the number of tables allocated to each size.


Customers arrive in groups of size :math:`j \in \{1, ..., max_i(c_i)\}` according to a homogeneous 
Poisson process with rate :math:`\lambda_j`. A group of customers is seated at the smallest possible 
available table. If there are no available tables large enough, the group of customers 
leaves immediately. Service time per group is exponential with means 
:math:`s = [s_1, s_2, ..., s_{max_i(c_i)}]`, and revenue per group is fixed at 
:math:`r = [r_1, r_2, ..., r_{max_i(c_i)}]`.


Let :math:`N_j(t)`` be the number of groups of size :math:`j` that were seated by time :math:`t` with 
:math:`N(t) = [N_1(t),...,N_{max_{i}(c_i)} (t)]`. 
The optimization problem is thus to maximize the total expected revenue:


  maximize :math:`N(T)r'` 

  subject to :math:`xc' ≤ K` 


**Recommended Parameter Settings:** 
:math:`T = 180minutes, K = 80`,

:math:`c = (2,4,6,8)`

:math:`λ = (3, 6, 3, 3, 2, 4/3 , 6/5 , 1) ∗ 60`

:math:`s = (20, 25, 30, 35, 40, 45, 50, 60)`

:math:`r = (15, 30, 45, 60, 75, 90, 105, 120)`

**Starting Solution(s):** For one starting solution, distribute total capacity equally across table sizes, then distribute remaining capacity by descending table size. Eg. for the above settings, :math:`80/4 = 20` capacity is
assigned to each size, and the remaining 6 capacity is portioned to a table of size 6: :math:`x = (10, 5, 4, 2)`.
For multiple starting solutions, distribute total capacity uniformly across table sizes, then distribute
remaining capacity by descending table size.

**Measurement of Time:** Number of replications of length :math:`T`.

**Optimal Solutions:** Unknown.

**Known Structure:** Unknown.