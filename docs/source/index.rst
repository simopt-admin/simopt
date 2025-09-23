Welcome to SimOpt's documentation!
==================================

The purpose of the SimOpt testbed is to encourage development and constructive comparison of simulation-optimization (SO) solvers (algorithms).
We are particularly interested in the finite-time performance of solvers, rather than the asymptotic results that one often finds in related literature.

For the purposes of this site, we define simulation as a very general technique for estimating statistical measures of complex systems.
A system is modeled as if the probability distributions of the underlying random variables were known.
Realizations of these random variables are then drawn randomly from these distributions.
Each replication gives one observation of the system response, i.e., an evaluation of the objective function.
By simulating a system in this fashion for multiple replications and aggregating the responses, one can compute statistics and use them for evaluation and design.

The paper  `Pasupathy and Henderson (2006) <https://www.informs-sim.org/wsc06papers/028.pdf>`_ explains the original motivation for the testbed, and the follow-up paper `Pasupathy and Henderson (2011) <https://www.informs-sim.org/wsc11papers/363.pdf>`_ describes an earlier interface for MATLAB implementations of problems and solvers.
The paper `Dong et al. (2017) <https://www.informs-sim.org/wsc17papers/includes/files/179.pdf>`_  conducts an experimental comparison of several solvers in SimOpt and analyzes their relative performance.
The recent Winter Simulation Conference paper `Eckman et al. (2019) <https://www.informs-sim.org/wsc19papers/374.pdf>`_ describes in detail the recent changes to the architecture of SimOpt and the control of random number streams.

The :mod:`simopt.models` module contains the simulation logic to simulate a variety of systems and SO test problems built around these models.
The :mod:`simopt.solvers` module provides users with the latest SO solvers to solve different types of SO problems.
The two modules are intended to help researchers evaluate and compare the finite-time performance of existing solvers.

The source code consists of the following modules:

* The :mod:`simopt.base` module contains class definitions for models, problems, and solvers.
* The :mod:`simopt.experiment_base` module contains class definitions and functions for running experiments with simulation-optimization solvers. 
* The :mod:`simopt.data_farming_base` module contains class definitions and functions for running data-farming experiments.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Component Descriptions

   models
   solvers

.. toctree::
   :maxdepth: 1
   :caption: Developer's Guide

   Environment Setup <development/environment_setup>
   Testing Guide <development/testing>
   API Reference <autoapi/simopt/index>

.. toctree::
   :maxdepth: 1
   :caption: External Resources

   SimOpt Repo <https://github.com/simopt-admin/simopt>
   MRG32k3a RNG Repo <https://github.com/simopt-admin/mrg32k3a>

Acknowledgments
---------------

An earlier website for `SimOpt <http://www.simopt.org>`_ was developed through work supported by the National Science Foundation under grant nos. DMI-0400287, CMMI-0800688 and CMMI-1200315.
Recent work on the development of SimOpt has been supported by the National Science Foundation under grant nos. DGE-1650441, CMMI-1537394, CMMI-1254298, CMMI-1536895, IIS-1247696, and TRIPODS+X DMS-1839346, by the Air Force Office of Scientific Research under grant nos. FA9550-12-1-0200, FA9550-15-1-0038, and FA9550-16-1-0046, and by the Army Research Office under grant no. W911NF-17-1-0094.
Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation (NSF).
