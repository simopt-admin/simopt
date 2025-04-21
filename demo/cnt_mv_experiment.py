"""
This script is intended to help with debugging a model.
It imports a model, initializes a model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
import numpy as np

# Import random number generator.
from mrg32k3a.mrg32k3a import MRG32k3a

# Import model.

from simopt.models.cntnv import CntNV 


# -----------------------------------------------


# set fixed factors for all experiments
n_products = 2
mat_to_prod = [[1, 2, 1, 3],
            [1, 1, 3, 1]]
process_cost = [.1,.1]
sales_price = [10,15]
total_budget = 10000
order_quantity = [20,20,20,20]
n = 100 # number simulation replications

# Experiment 1
corr_matrix = [[1,0], [0,1]]
means = [[0,0],[.25,.5],[0,0]]
t_intervals = [8,21]
profit = []
stockout_qty_1 = []
stockout_qty_2 = []
stockout_1 = []
stockout_2 = []
fixed_factors = {"rank_corr": corr_matrix,
                 "poi_mean": means, 
                 "t_intervals": t_intervals, 
                 "total_budget": total_budget, 
                 "order_quantity": order_quantity,
                 "num_product": n_products,
                 "mat_to_prod": mat_to_prod,
                 "process_cost": process_cost,
                 "sales_price": sales_price}
mymodel = CntNV(fixed_factors)
# running replications.
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]
for i in range(n):
    # Run a single replication of the model.
    responses, gradients = mymodel.replicate(rng_list)
    for key, value in responses.items():
        if key == 'profit':
            profit.append(value)
        elif key == 'stockout_qty':
            stockout_qty_1.append(value[0])
            stockout_qty_2.append(value[1])
        elif key == 'stockout':
            stockout_1.append(value[0])
            stockout_2.append(value[1])

# construct 95% CI
mean_profit = np.mean(profit)
mean_so_1 = np.mean(stockout_qty_1)
mean_so_2 = np.mean(stockout_qty_2)
prob_so_1 = np.mean(stockout_1)
prob_so_2 = np.mean(stockout_2)

hw_profit = 1.96*np.std(profit)/np.sqrt(n)
hw_so_1 = 1.96*np.std(stockout_qty_1)/np.sqrt(n)
hw_so_2 = 1.96*np.std(stockout_qty_2)/np.sqrt(n)
hw_p_so_1 = 1.96*np.sqrt(abs(prob_so_1*(1-prob_so_1)/n)) 
hw_p_so_2 = 1.96*np.sqrt(abs(prob_so_2*(1-prob_so_2)/n))

lb_profit,ub_profit = mean_profit-hw_profit, mean_profit+hw_profit
lb_so_1, ub_so_1 = mean_so_1-hw_so_1, mean_so_1+hw_so_1
lb_so_2, ub_so_2 = mean_so_2-hw_so_2, mean_so_2+hw_so_2
lb_p_so_1, ub_p_so_1 = prob_so_1-hw_p_so_1, prob_so_1+hw_p_so_1
lb_p_so_2, ub_p_so_2 = prob_so_2-hw_p_so_2, prob_so_2+hw_p_so_2

print("Experiment 1 results:")
print(f"proft: ({lb_profit},{ub_profit}) ")
print(f"Num stockout 1: ({lb_so_1},{ub_so_1}) ")
print(f"Num stockout 2: ({lb_so_2},{ub_so_2}) ")
print(f"Prob stockout 1: ({lb_p_so_1},{ub_p_so_1}) ")
print(f"Prob stockout 2: ({lb_p_so_2},{ub_p_so_2}) ")
print()
print()

# Experiment 2
corr_matrix = [[1,0], [0,1]]
means = [[0,0],['t/48','t/24'],[.25,.5],[0,0]]
t_intervals = [8,12,21]
profit = []
stockout_qty_1 = []
stockout_qty_2 = []
stockout_1 = []
stockout_2 = []
fixed_factors = {"rank_corr": corr_matrix,
                 "poi_mean": means, 
                 "t_intervals": t_intervals, 
                 "total_budget": total_budget, 
                 "order_quantity": order_quantity,
                 "num_product": n_products,
                 "mat_to_prod": mat_to_prod,
                 "process_cost": process_cost,
                 "sales_price": sales_price}
mymodel = CntNV(fixed_factors)
# running replications.
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]
for i in range(n):
    # Run a single replication of the model.
    responses, gradients = mymodel.replicate(rng_list)
    for key, value in responses.items():
        if key == 'profit':
            profit.append(value)
        elif key == 'stockout_qty':
            stockout_qty_1.append(value[0])
            stockout_qty_2.append(value[1])
        elif key == 'stockout':
            stockout_1.append(value[0])
            stockout_2.append(value[1])

# construct 95% CI
mean_profit = np.mean(profit)
mean_so_1 = np.mean(stockout_qty_1)
mean_so_2 = np.mean(stockout_qty_2)
prob_so_1 = np.mean(stockout_1)
prob_so_2 = np.mean(stockout_2)

hw_profit = 1.96*np.std(profit)/np.sqrt(n)
hw_so_1 = 1.96*np.std(stockout_qty_1)/np.sqrt(n)
hw_so_2 = 1.96*np.std(stockout_qty_2)/np.sqrt(n)
hw_p_so_1 = 1.96*np.sqrt(abs(prob_so_1*(1-prob_so_1)/n)) 
hw_p_so_2 = 1.96*np.sqrt(abs(prob_so_2*(1-prob_so_2)/n))

lb_profit,ub_profit = mean_profit-hw_profit, mean_profit+hw_profit
lb_so_1, ub_so_1 = mean_so_1-hw_so_1, mean_so_1+hw_so_1
lb_so_2, ub_so_2 = mean_so_2-hw_so_2, mean_so_2+hw_so_2
lb_p_so_1, ub_p_so_1 = prob_so_1-hw_p_so_1, prob_so_1+hw_p_so_1
lb_p_so_2, ub_p_so_2 = prob_so_2-hw_p_so_2, prob_so_2+hw_p_so_2

print("Experiment 2 results:")
print(f"proft: ({lb_profit},{ub_profit}) ")
print(f"Num stockout 1: ({lb_so_1},{ub_so_1}) ")
print(f"Num stockout 2: ({lb_so_2},{ub_so_2}) ")
print(f"Prob stockout 1: ({lb_p_so_1},{ub_p_so_1}) ")
print(f"Prob stockout 2: ({lb_p_so_2},{ub_p_so_2}) ")  
print()
print()  
        
# Experiment 3
corr_matrix = [[1,.7], [.7,1]]
means = [[0,0],[.25,.5],[0,0]]
t_intervals = [8,21]
profit = []
stockout_qty_1 = []
stockout_qty_2 = []
stockout_1 = []
stockout_2 = []
fixed_factors = {"rank_corr": corr_matrix,
                 "poi_mean": means, 
                 "t_intervals": t_intervals, 
                 "total_budget": total_budget, 
                 "order_quantity": order_quantity,
                 "num_product": n_products,
                 "mat_to_prod": mat_to_prod,
                 "process_cost": process_cost,
                 "sales_price": sales_price}
mymodel = CntNV(fixed_factors)
# running replications.
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]
for i in range(n):
    # Run a single replication of the model.
    responses, gradients = mymodel.replicate(rng_list)
    for key, value in responses.items():
        if key == 'profit':
            profit.append(value)
        elif key == 'stockout_qty':
            stockout_qty_1.append(value[0])
            stockout_qty_2.append(value[1])
        elif key == 'stockout':
            stockout_1.append(value[0])
            stockout_2.append(value[1])

# construct 95% CI
mean_profit = np.mean(profit)
mean_so_1 = np.mean(stockout_qty_1)
mean_so_2 = np.mean(stockout_qty_2)
prob_so_1 = np.mean(stockout_1)
prob_so_2 = np.mean(stockout_2)

hw_profit = 1.96*np.std(profit)/np.sqrt(n)
hw_so_1 = 1.96*np.std(stockout_qty_1)/np.sqrt(n)
hw_so_2 = 1.96*np.std(stockout_qty_2)/np.sqrt(n)
hw_p_so_1 = 1.96*np.sqrt(abs(prob_so_1*(1-prob_so_1)/n)) 
hw_p_so_2 = 1.96*np.sqrt(abs(prob_so_2*(1-prob_so_2)/n))

lb_profit,ub_profit = mean_profit-hw_profit, mean_profit+hw_profit
lb_so_1, ub_so_1 = mean_so_1-hw_so_1, mean_so_1+hw_so_1
lb_so_2, ub_so_2 = mean_so_2-hw_so_2, mean_so_2+hw_so_2
lb_p_so_1, ub_p_so_1 = prob_so_1-hw_p_so_1, prob_so_1+hw_p_so_1
lb_p_so_2, ub_p_so_2 = prob_so_2-hw_p_so_2, prob_so_2+hw_p_so_2      

print("Experiment 3 results:")
print(f"proft: ({lb_profit},{ub_profit}) ")
print(f"Num stockout 1: ({lb_so_1},{ub_so_1}) ")
print(f"Num stockout 2: ({lb_so_2},{ub_so_2}) ")
print(f"Prob stockout 1: ({lb_p_so_1},{ub_p_so_1}) ")
print(f"Prob stockout 2: ({lb_p_so_2},{ub_p_so_2}) ")        
 