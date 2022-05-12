Optimization of COVID-19 Testing Frequency
==================================================

**Problem Description**

COVID-19 is a contagious respiratory disease with a high trasmission rate. A college campus implements
regular survelliance testing to identify, isolate, and reduce disease spread. The population is divided 
into three groups: undergraduate, graduate, and faculty/staff with interaction rates :math:`inter_rate`. 
There is a probability of :math:`p_trans` transmissions per interaction. The initial proportion of infected
is :math:`init\_infect\_percent`. The disease progression for each individual is generated according to the following semi-Markov process:

.. image:: disease_progression.png

The number of days one takes to go from exposed to infectious is Poisson distributed with mean :math:`lamb\_exp\_inf`. 
An infected individual has a :math:`asymp\_rate` chance of being asymptomatic. The number of 
days one takes to go from infectious to symptom onset is Poisson distributed with mean :math:`lamb\_inf\_sym`.
After recovery, we assume that one cannot be reinfected.

We assume that a test has a false negative rate :math:`false\_neg` and a false positive rate :math:`false\_pos`.
Once tested positive, the patient will be isolated for :math:`iso\_day` number of days and follow the same disease
progression until recovery.

The decision variables is :math:`freq`, the testing frequency for each group.

The simulation is generated as follows:

  1. Generate number of people exposed on day n:
        :math:`new\_exp = t_rate * free\_inf_n * sus_n/(free_n)` 
        where :math:`t_rate = inter_rate * p_trans`

  2. For each individual in this group, generate their future disease progression and testing:

    (a) 
    
    (b) 

    (c)

    (d)

    (e)

  3. 


We will find the optimal testing frequency for each group which minimizes the total number of infections. 


**Recommended Parameter Settings:** 

"inter_rate": (10.58, 5, 2, 4, 10.58, 3, 1, 2, 10.57)

"p_trans": 0.018

"lamb_exp_inf": 2

"lamb_inf_sym": 3

"asymp_rate": 0.35

"false_pos": 0

"false_neg": 0.12

"iso_day": 12

**Starting Solutions:** (1/7, 1/7, 1/7)

**Measurement of Time:**  Number of simulation replications of length :math:`n`.

**Optimal Solutions:** Unknown.

**Known Structure:** Unknown.