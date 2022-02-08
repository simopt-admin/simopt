
Continuous Newsvendor Problem
===================================================

**Problem Description**

This problem was received from Dr. Eckman's SimOpt Library and the following is a description of a Continuous Version
of the Newsvendor Problem. This example can be used for different situations, assuming that the same variables are being 
used as factors and the same responses are necessary. The object ordered in this example can be tentative and can be 
applied to different Newsvendor problems as long as the factors remain stagnant. 

In this particular problem, a vendor orders a fixed quantity of liquid, assigned random variable :math:`x`, at the beginning of 
each working day to be sold to customers throughout the day. The quantity of liquid ordered has a cost to the vendor,
:math:`c`, and a price at which the vendor sells it to the customer, :math:`s`. At the end of the day, if there is a quantity of liquid
that has not been sold, it canpython be salvaged at a price, :math:`w`. The demand each day can be determined by a Burr Type XII 
distribution, it is denoted by :math:`D` and contains the parameters, alpha and beta. The Burr Type XII Distribution can have 
values that range anywhere from :math:`[0,∞)` and it also has a cumulative distribution function that is 
represented by the equation, :math:`F(x) = 1 - (1-x^α)^{-β}` where :math:`x, α,` and :math:`β` are all positive. There is a simulation that can 
calculate random variates from the Burr Type XII distribution. Ultimately, the goal of this problem is to determine the 
quantity of :math:`x` liquid that needs to be ordered to maximize the expected profit for the vendor. The assumptions made for
the solution procedure for this problem are that the parameter values and the distribution of the demand (Burr Type XII) 
are unknown. 

==================================================

**Factors:**
 *Cost* (:math:`c`) ---> The amount it costs to the newsvendor to purchase one unit volume of liquid. The default value for cost will be 5 dollars.

 *Price* (:math:`s`) ---> The amount the newsvendor sells one unit volume of liquid for. The default value for price will be 9 dollars. 

 *Salvage Price* (:math:`w`) ---> At the end of each day, if there is liquid left over, each unit volume of liquid can be salvaged for a specific price. The default value for salvage price is 1 dollar. 

 *Alpha and Beta for Demand Distribution* (:math:`α` and :math:`β`) ---> The Burr Type XII Distribution that is being used for demand has certain parameters denoted by alpha and beta. The default values for alpha and beta are 2 and 20, respectively.

**Response:**

 *Quantity of Liquid* (:math:`x`) ---> This will be the volume of liquid that needs to be ordered at the beginning of each day in order to maximize the expected profit. 

 *Default Profit* (:math:`e`) ---> This will be the maximum expected profit that correlates to the quantity of liquid ordered, x.

Optimization Problem 1:

Objective: 

Constraints: 

Decision Variables: 

Fixed Factor Values: 

Starting Solution:

Optimal Solution:

Optimal Objective Function Value:

Optimization Problem 2: 