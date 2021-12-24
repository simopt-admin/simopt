
Newsvendor Problem under Dynamic Consumer Substitution
======================================================

.. math::
  \DeclareMathOperator*{\argmax}{argmax}

(Integer-ordered variables, constrained.)

This example is adapted from the article Mahajan, S., & van Ryzin, G. (2001). *Stocking Retail
Assortments under Dynamic Consumer Substitution*. Operations Research, 49(3), 334-351. [#f1]_

**Problem Statement:**

In the newsvendor problem under dynamic consumer substitution, the retailer sells :math:`n` substitutable
products :math:`j = 1, \ldots, n`, each with price :math:`p^j` and cost :math:`c^j` . The only decision variable is the vector of initial
inventory levels :math:`x = (x_1, \ldots, x_n)`.

The newsvendor problem considered here differs from the classical newsvendor problem in that the 
demand is not given by a predetermined distribution, but depends on the initial inventory levels :math:`x` as
well. The customers :math:`t = 1, \ldots, T` arrive in order and each can choose one product that is in-stock when
he/she arrives, namely any element in :math:`S(x_t) = \{j : x^j_t > 0\} \cup \{0\}` where :math:`0` denotes the no-purchase
option.

Each customer :math:`t` assigns a utility :math:`U^j_t` to option :math:`j = 0, \ldots, n`, and thus :math:`U_t = (U^0_t, U^1_t, \ldots, U^n_t)` is his/her
vector of utilities. Note that :math:`U^j_t` is the utility of product :math:`j` net of the price :math:`p^j`, and therefore could be 
negative. Since the *no-purchase* option :math:`0` incurs neither utility nor cost, one can assume :math:`U^0_t = 0`.
Customer :math:`t` maximizes his/her utility by making the choice

.. math::
  d(x_t,U_t) = \argmax_{j\in S(x_t)} U^j_t


| Let :math:`\omega = \{U_t : t = 1, \ldots, T\}`denote the sample path, and assume that `\omega` follows the probability distribution :math:`P`. We consider a one-period inventory model and assume :math:`P(T < +\infty) = 1`.
| The retailer knows the probability measure :math:`P`, and his/her objective is to choose :math:`x` that maximizes total expected profit.

**Recommended Parameter Settings:** 

Use the Multinomial Logit (MNL) model. :math:`U^0_t, U^1_t, \ldots, U^n_t` are mutually independent random variables
of the form

.. math::
  U^j_t = u^j + \epsilon^j_t

where :math:`uj` is a constant and :math:`\epsilon^j_t`, :math:`j = 0, 1, \ldots, n` are mutually independent Gumbel random variables with
:math:`P(\epsilon^j_t \leq z) = \exp(-e^{-(z/\mu+\gamma)})` (:math:`\gamma` is Euler's constant,  :math:`\gamma \approx 0.5772`.)

Setting 1: :math:`n = 2, T = 5, u_j = 1, \mu = 1`.

Setting 2: :math:`n = 10, T = 30, u_j = 5 + j, \mu = 1`.

**Starting Solutions:** :math:`x = (2, 3)` in Setting 1, :math:`x = (3, 3, \ldots, 3)` in Setting 2.

**Measurement of Time:**  Number of sample paths :math:`\omega` simulated.

**Optimal Solutions:** Unknown.

**Known Structure:** Under the recommended parameter settings of Multinomial Logit model, the choice probabilities are given by

.. math::
  q^j(S) &= P(U^j_t = \text{max}\{U^i_t: i \in S\}) \\
        &= \frac{v_j}{\sum_{i \in S}v^i + v^0},

where

.. math::
  vj =
    \begin{cases}
      e^{u_j / \mu} & j \in S,\\
      e^{u_0 / \mu} & j = 0.
    \end{cases}

**References**

.. [#f1] Mahajan, S., & van Ryzin, G. (2001). *Stocking Retail Assortments under Dynamic Consumer Substitution*. Operations Research, 49(3), 334-351.