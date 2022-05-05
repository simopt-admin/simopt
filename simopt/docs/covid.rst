Model: COVID-19 Disease Progression and Testing Frequency (COVID)
=================================================================

Description:
------------
COVID-19 is a contagious respiratory disease with a high trasmission rate. A college campus implements
regular survelliance testing to identify, isolate, and reduce disease spread. However, model can also
be generalized and simulate the interaction among any given groups of people.
The initial prevalance level of the disease is :math:`init_infect_percent`. The population is divided 
into different groups with intra-group interaction matrix :math:`inter_rate`. There is a probability of :math:`p_trans` 
transmissions per interaction. The transmission rate per individual can be calculated by multiplying 
:math:`inter_rate` by :math:`p_trans`. The disease progression for each individual follows the following semi-Markov process:

.. image:: covid_compartments.png
  :width: 400

After recovery, we assume that an individual cannot be reinfected. Once tested positive, the patient will be moved to isolated states 
and follow the same disease progression.

The simulation is generated as follows:

1. For each day in :math:`n` and for each group :math:`g`, generate newly exposed individuals.

2. For each newly exposed individual in group :math:`g`:

    (a) Generate :math:`exp_days` number of days remaining in exposed.

    (b) For each day in exposed, move to isolation exposed if tested positive.

    (c) At the end of :math:`exp_days`, move to infectious/isolation infectious

    (d) Generate :math:`inf_days` number of days remaining in infectious/isolation infectious.

    (e) For each day in infectious, move to isolation infectious if tested positive.

    (f) At the end of :math:`inf_days`, move to symptomatic/asymptomatic/isolation (a)symptomatic.

    (g) Generate :math:`symp_asymp_days` number of days remaining in symptomatic/asymptomatic/isolation (a)symptomatic.

    (h) At the end of :math:`symp_asymp_days`, move to recovered.


Sources of Randomness:
----------------------
There are six sources of randomness.

1. The number of newly exposed individuals on each day follows a Poisson distribution with mean equal to transmission rate times
number of free infected (infectious + symptomatic + asymptomatic) individuals times number of susceptible individuals.

2. The number of days from exposed to infectious for each individual is Poisson distributed with mean :math:`lamb_exp_inf`.

3. The number of days from infectious to symptomatic/asymptomatic for each individual is Poisson distributed with mean :math:`lamb_inf_sym`.

4. The number of days from symptomatic/asymptomatic to recovered for each individual is Poisson distributed with mean :math:`lamb_sym`.

5. An individual in infectious state has a :math:`asymp_rate` chance of being asymptomatic.

6. An exposed/infectious/symptomatic/asymptomatic individual in group :math:`g` has a probability 
:math:`freq_g` of being tested and moved to the isolated states.

Model Factors:
--------------
* num_groups: Number of groups.

    * Default: 3

* n: Number of days to simulate.

    * Default: 200

* p_trans: Probability of transmission per interaction.

    * Default: 0.018

* inter_rate: Interaction rates between two groups per day

    * Default: (10.58, 5, 2, 4, 6.37, 3, 6.9, 4, 2)

* group_size: Size of each group.

    * Default: (8123, 4921, 3598)

* lamb_exp_inf: Mean number of days from exposed to infectious.

    * Default: 2.0

* lamb_inf_sym: Mean number of days from infectious to symptomatic.

    * Default: 3.0

* lamb_sym: Mean number of days from symptomatic/asymptomatic to recovered.

    * Default: 12.0

* init_infect_percent: Initial prevalance level.

    * Default: (0.00200, 0.00121, 0.0008)

* freq: Testing frequency of each group.

    * Default: (0/7, 0/7, 0/7)

* asymp_rate: Probability of being asymptomatic.

    * Default: 0.35

* false_neg: False negative rate.

    * Default: 0.12

Respones:
---------
* num_infected: Number of infected individuals per day

* num_susceptible: Number of susceptible individuals per day

* num_exposed: Number of exposed individuals per day

* num_recovered: Number of recovered individuals per day

* total_cases: Total number of infected individuals

References:
===========
This model is adapted from the article Frazier, Peter I et al. “Modeling for COVID-19 college reopening decisions: Cornell, a case study.” Proceedings of the National Academy of Sciences of the United States of America vol. 119,2 (2022): e2112532119. doi:10.1073/pnas.2112532119



Optimization Problem: CovidMinInfect (COVID-1)
========================================================

Decision Variables:
-------------------
* freq

Objectives:
-----------
Find the optimal testing frequency for each group which minimizes the expected total number of infected individuals over time :math:`n`.


Constraints:
------------
The total number of tests per day should be smaller than testing_cap.

Problem Factors:
----------------
* initial_solution: Initial solution from which solvers start.

  * Default: (0/7, 0/7, 0/7)
  
* budget: Max # of replications for a solver to take.

  * Default: 300

* testing_cap: Maxi testing capacity per day.

  * Default: 7000

* budget: Max # of replications for a solver to take

  * Default: 300


Fixed Model Factors:
--------------------
* n/a


Starting Solution: 
------------------
* initial_solution: (0/7, 0/7, 0/7)
  

Random Solutions: 
------------------
Sample each :math:`x_i` in a simplex.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown
