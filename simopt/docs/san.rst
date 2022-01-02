
Stochastic Activity Network (SAN) Duration
==========================================

Consider the following stochastic activity network (SAN) where the arcs are labeled
from 1 through 13. (SANs are also known as PERT networks, and are used in planning
large-scale projects. This SAN is adapted from Avramidis, A.N., Wilson, J.R. (1996).
Integrated variance reduction strategies for simulation. Operations Research 44, 327-346.)
Each arc :math:`i` is associated with a task with random duration :math:`X_i`. Task
durations are independent.

.. image:: san.PNG
  :alt: The SAN diagram has failed to display
  :width: 500

Suppose that :math:`X_i` is exponentially distributed with mean :math:`\theta_i`
for each :math:`i`. Suppose that we can select :math:`\theta_i > 0` for each :math:`i`,
but there is an associated cost. In particular, we want to minimize :math:`ET(\theta) + f(\theta)`,
where :math:`T(\theta)` is the (random) duration of the longest path from :math:`a`
to :math:`i` and :math:`f(\theta) = \sum_{i=1}^{9}\theta_i^{-1}`. We require that
:math:`theta_i \in [0.01, 100]` for each :math:`i`.

(Continuous variables, unconstrained -- deterministic upper and lower bounds on vars only).

*Starting Solution(s):* Start from :math:`\theta_0 = [1, 1, ..., 1]`. If multiple
initial solutions are required, sample uniformly from :math:`[0.5, 5]^{13}`.

*Measurement of Time:* One evaluation of the longest path (and its gradient).

*Recommended Budgets:* :math:`10,000` and :math:`100,000`

*Optimal Solution:* Unknown.

*Known Structure:* The objective function :math:`ET(\theta) + f(\theta)` is convex
in :math:`\theta`. An IPA estimator of the gradient is also given in the code.