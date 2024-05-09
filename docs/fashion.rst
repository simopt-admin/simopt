Model: Fashion (Fashion)
==========================================

Description:
------------

Consider an inventory model for an online catalog retailer where stock is ordered twice throughout the sales season. Once, at the beginning, and once more at a specified time t. During the entirety of the sales season, customers can place orders, and if inventory is sold out, those orders will be considered backorders (if they choose to order). The likelehood that a customer would backorder is represented by the function Be−(t+11−d). The variables B and d can take on only positive values. After the second quantity of stock is ordered, possible backorders become lost sales due to the fact that inventory cannot be sold if the customer can never receive it. There is also a 0.35 chance that a customer will return their order and they will then recieve a refund based on the original selling price of the product. The model could be seen as a two stage stochastic program where the reorder quantity is the recourse decision variable. 


Sources of Randomness:
----------------------

There are 3 sources of randomness in this model:

1. The weekly demand for clothing over a sales season.
2. The probability a customer will return an item.
3. The probability that a customer is willing to place a backorder if their preferred item is currently out of stock.

The binomial distribution is the probability distribution of the number of successes in a fixed number of independent Bernoulli trials with a common sucess probability. In this case, a success would be a customer choosing to return an item or place a backorder. 

Model Factors:
---------------

- `leadtime`: The time between the ordering and receiving of products.
  - Default: 11

- `product_price`: Fixed price of products, does NOT change.
  - Default: 20.0

- `n_weeks`: Number of weeks to simulate during the sales season.
  - Default: 22

- `initial_order_quantity`: Initial retailer order quantity for inventory. 
  - Default: 1000

- `second_order_time`: The time t when the second retailer order is placed. Will be received with a certain leadtime. 
  - Default: 7

- `return_chance`: The probability that a customer returns an item. 
  - Default: 0.35

- `return_refund`: The percentage of the original product price the customer gets refunded when making a return.
  - Default: 0.5
 
- `backorder_d`: The constant d from the backorder function. 
  - Default: 3
  
- `backorder_B`: The constant B from the backorder function. 
  - Default: 1.5

Responses:
----------

- `final_end_inventory`: Total inventory left at the end of the sales season. 

- `backorders`: Total number of unfulfilled backorders. 

- `lost_sales`: Lost sales (in products) incurred due to lack of inventory.

- `profit`: Total profit (revenue - cost), where the cost of the reorder quantity is included in the cost. 

References:
------------

This model is adapted from the article: Marshall Fisher, Kumar Rajaram, Ananth Raman, (2001) Optimizing Inventory Replenishment of Retail Fashion Products.
Manufacturing & Service Operations Management 3(3):230-241


Optimization Problem: Maximize expected profit by choosing an initial order quantity and reorder time. 
=======================================================================

Decision Variables:
--------------------

- `initial_order_quantity`
- `second_order_time`

Objectives:
------------

Maximize expected profit for the online catalog retailer, where profit includes revenue from sales and cost includes lost sales, backorders, and obsolete inventory at the end of the sales season. 

Constraints:
-------------


- `initial_order_quantity > 0`
- ` 1 ≤ second_order_time ≤ n_weeks - leadtime`



Problem Factors:
------------------

- `initial_solution`: initial solution from which solvers start.
  - Default: (1000, 7)
- `budget`: Max # of replications for a solver to take.
  - Default: 1000

Starting Solution:
-------------------

- `initial_order_quantity = 1000`
- `second_order_time = 7`

Random Solutions:
------------------

Choose an initial order quantity that is normally distributed with mean 1000, variance of 200 and round to a positive integer. For second order time, pick an integer uniformly from 1 to (n_weeks - leadtime). 

Optimal Solution:
------------------

- Unknown

Optimal Objective Function Value:
----------------------------------

- Unknown
