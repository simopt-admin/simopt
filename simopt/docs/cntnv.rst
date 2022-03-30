
Model: Continuous Newsvendor Problem (CNTNV)
============================================

Description:
In this particular problem, a vendor orders a fixed quantity of liquid, assigned random variable :math:`x`, at the beginning of 
each working day to be sold to customers throughout the day. The quantity of liquid ordered has a cost to the vendor,
:math:`c`, and a price at which the vendor sells it to the customer, :math:`s`. At the end of the day, if there is a quantity of liquid
that has not been sold, it can be salvaged at a price, :math:`w`. The demand each day can be determined by a Burr Type XII 
distribution, it is denoted by :math:`D` and contains the parameters, :math:`α` and :math:`β`. The Burr Type XII Distribution can have 
values that range anywhere from :math:`[0,∞)` and it also has a cumulative distribution function that is 
represented by the equation, :math:`F(x) = 1 - (1-x^α)^{-β}` where :math: `x, α,` and :math:`β` are all positive. There is a simulation that can 
calculate random variates from the Burr Type XII distribution. Ultimately, the goal of this problem is to determine the 
quantity of :math:`x` liquid that needs to be ordered to maximize the expected profit for the vendor. 

Sources of Randomness:
There is one source of randomness in this problem and it is used to calculate the daily demand. The daily demand has a Burr Type XII
distribution from :math:`[0,∞)` and it also has a cumulative distribution function that is represented by the equation, :math:`F(x) = 1 - (1-x^α)^{-β}` 
where :math:`x, α,` and :math:`β` are all positive.

Model Factors: 

 *Cost* (:math:`c`) - The amount it costs to the newsvendor to purchase one unit volume of liquid. The default value for cost will be 5 dollars.

 *Price* (:math:`s`) - The amount the newsvendor sells one unit volume of liquid for. The default value for price will be 9 dollars. 

 *Salvage Price* (:math:`w`) - At the end of each day, if there is liquid left over, each unit volume of liquid can be salvaged for a specific price. The default value for salvage price is 1 dollar. 

 *Alpha and Beta for Demand Distribution* (:math:`α` and :math:`β`) ---> The Burr Type XII Distribution that is being used for demand has certain parameters denoted by alpha and beta. The default values for alpha and beta are 2 and 20, respectively.

 *Quantity of Liquid* (:math:`x`) - This will be the volume of liquid that needs to be ordered at the beginning of each day in order to maximize the expected profit. 

Responses:

 *Default Profit* - This will be the maximum expected profit that correlates to the quantity of liquid ordered, x.

References: 

This model is adapted from Dr. Eckman's SimOpt Problems Library, CtsNews Folder.

Optimization Problem: Continuous Newsvendor
===========================================

Decision Variables:

*Fixed Quantity Ordered* (:math:`x`) - This model will determine the amount of liquid, :math:`x` to order at the beginning of each
day.

Objectives: 

The purpose of this model is to maximize the profit for the newsvendor. 

Constraints: 
Nonnegativity for all factors

Problem Factors:
*Budget* - Refers to the amount of replications the solver will run when doing the newsvendor problem.

Fixed Model Factors:
Empty

Starting Solution:

Starting Solution = 0

Model Default Factors:
*Purchase Price* - The price that the liquid is purchased at is 5 dollars.
*Sales Price* - The price that the liquid is sold for is 9 dollars.
*Salvage Price* - The price that the liquid can be salvaged at is 1 dollar.
*Burr_c* - The alpha constant for the Burr random distribution is set to 2
*Burr_k* - The beta constant for the Burr random distribution is set to 20

Optimal Solution:
Global minimum at :math:`x* = (1/((1-r^)^{1/β})-1)^{1/α}`
The optimal solution is :math:`x*` = 0.1878

Optimal Objective Function Value:
The maximum expected profit is 0.4635

