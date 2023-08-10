Solver: Coordinate Search (COORDSRCH)
=====================================

Description:
------------
Sequentially searches each coordinate axis for a local optimum. It takes :math:`N_k` replications at each solution visited in :math:`k^th` iteration. :math:`N_k` is non-decreasing through iterations.

Modifications & Implementation:
-------------------------------
Each new solution represents a local optimum on its corresponding coordinate axis.

Scope:
------
* objective_type: single

* constraint_type: deterministic

* variable_type: discrete

* gradient_observations: not available

Solver Factors:
---------------
* crn_across_solns: Use CRN across solutions?

    * Default: True

* ini_sample_size: Sample size for :math:`1^st` iteration; ini_sample_size\ :math:`> 0` .

    * Default: 100

* sample_size_slope: Sample size increment per iteration; sample_size_slope\ :math:`\geq 0` .

    * Default: 10

* :math:`m_0`: :math:`2^{m_0}` is the maximum step size the line search may take; :math:`m_0 \geq 0` .

    * Default: 4

* :math:`z^{max}`: Controls the maximum distance from :math:`{\hat{\mathbf{x}}}_{k-1}^\ast`; :math:`z \leq z^{max}+2^{m_0}`; :math:`z^{max} > 0` .

    * Default: 50

References:
===========
This solver is adapted from the article Hong, L.J., (2005).
Discrete Optimization via Simulation Using Coordinate Search.
*Proceedings of the Winter Simulation Conference*, Orlando, FL, USA, 2005, pp.803-810.
(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1574325)