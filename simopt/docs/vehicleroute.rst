
Vehicle Routing Problems with Stochastic Demands and Travel Times
==================================================================


{Categorical variables, constrained.}
This example is taken from the article Gendreau, M., G. Laporte and R. Séguin. 1996.
Invited Review: Stochastic vehicle routing. European Journal of Operational Research
88, 3-12. [#f1]_

**Problem Statement:**

Consider the *Vehicle Routing Problem* (VRP) defined on a graph :math:`G = (V, A)`, 
where :math:`V = {v_0,v_1,v_2,\ldots,v_n}`` is a set of vertices and :math:`A = {(v_i,v_j) : i \neq j, v_i,v_j \in V}` 
is a set of arcs. Vertex :math:`v_0` denotes a depot where :math:`m` identical vehicles
are based, while the remaining vertices correspond to :math:`n` customers. A matrix :math:`C = (c_{ij})`
is defined on :math:`A` with :math:`c_{ij}` representing the distance between :math:`v_i` and :math:`v_j` . A travel time
matrix :math:`T = (T_{ij})` is also defined on :math:`A` with :math:`T_{ij}` denoting the random travel time from
:math:`v_i` to :math:`v_j`. A vector :math:`D = (D_1,D_2,\ldots,D_n)` represents the random demands from the
customers :math:`1,\ldots,n`. Each vehicle, having the same capacity :math:`Q`, will start from the depot,
visit a subset of the customers, and return to the depot. Service levels are measured in two ways. First, 
we evaluate the probability of the total demand along each route not exceeding the capacity :math:`Q`. Second, we 
evaluate the probability of the travel time along each route not exceeding some time limit :math:`B`. Our goal is 
to find the set of routes for the vehicles minimizing the total distance traveled by the vehicles while maintaining desired
service levels. :math:`\theta = (\theta_1,\ldots,\theta_m)` denotes the routes for the vehicles :math:`1,\ldots,m`, where
:math:`\theta_k \subset A` and is the set of arcs for vehicle :math:`k` to travel along :math:`(1 \leq k \leq m)`. The problem can then
be formulated as follows:

.. math::
    \min ~~~ \sum\limits_{i,j=1}^{n}(D_i - Q)^{+} \cdot x_{ij} \cdot c_{ij} \\
    \text{s.t.} ~~~ \sum\limits_{i = 1}^{n}x_{ij} = 1, j = 1,2,\ldots,n \\
    \sum\limits_{j = 1}^{n}x_{ij} = 1, i = 1,2,\ldots,n \\
    \sum\limits_{i = 0}^{n}x_{i0} = m \\
    \sum\limits_{i = 0}^{n}x_{0j} = m \\
    P(\sum\limits_{i,j = 1}^{n}T_{ij} \leq B) \geq \beta_k, 1 \leq k \leq m \\
    x_{ij} \in \{0, 1\}, i,j = 1, 2 \ldots n

**Recommended Parameter Settings:** 
:math:`n = m = 5, \alpha_k = \beta_k = 0.9` for :math:`1\leq k \leq m`, :math:`Q = 350`,
:math:`B = 240`, and

.. math::
    C = \begin{pmatrix}
        0 &35 &78 &76 &98 &55\\
        35 &0 &60 &59 &91 &81\\
        78 &60 &0 &3 &37 &87\\
        76 &59 &3 &0 &36 &83\\
        98 &91 &37 &36 &0 &84\\
        55 &81 &87 &83 &84 &0
        \end{pmatrix}

:math:`T_{ij}` is uniformly distributed between :math:`0.5c_{ij}` and :math:`1.5c_{ij}`for :math:`0\leqi,j\leqn`. :math:`Di`` is uniformly
distributed between 110 and 190. The :math:`T_{ij}` 's and :math:`D_i`'s are independent.

**Starting Solutions:** Each vehicle serves exactly one customer, i.e., each vehicle starts at
the depot, visits exactly one customer, and returns to the depot.

**Measurement of Time:**  The use of one set of travel times and demands.

**Optimal Solutions:** Unknown.

**Known Structure:** None.

**References**

.. [#f1] Gendreau, M., G. Laporte, and R. Séguin. 1996. Invited Review: Stochastic vehicle routing. *European Journal of Operational Research* 88: 3-12.