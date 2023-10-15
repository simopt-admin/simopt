"""
This script is intended to help with debugging a model.
It imports a model, initializes a model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import random number generator.
from mrg32k3a.mrg32k3a import MRG32k3a

# Import model.
from simopt.models.dynamnews import DynamNews

n = 100 # sample_size_per_system
num_products = 3
max_inventory = 10
k = max_inventory**num_products

profit_matrix = np.zeros((n, k))
n_prod_stockout_matrix = np.zeros((n, k))
n_missed_orders_matrix = np.zeros((n, k))
fill_rate_matrix = np.zeros((n, k))

# Other factors
num_customer = 10 # 30
c_utility = [0, 1, 2]
price = [9, 8, 7]
cost = [5, 3, 4]

# IF RUNNING ONE SYSTEM
# profit_vector = np.zeros(n)
# n_prod_stockout_vector = np.zeros(n)
# n_missed_orders_vector = np.zeros(n)
# fill_rate_vector = np.zeros(n)
# fixed_factors = {"num_prod": num_products,
#                              "num_customer": num_customer,
#                              "c_utility": c_utility,
#                              "price": price,
#                              "cost": cost,
#                              "init_level": (3, 3, 3)}  # This one is the decision variable. 
# mymodel = DynamNews(fixed_factors)
# rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]
# # If doing CRN across systems:
# # rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]
# for rep_idx in range(n):
#     # Can manually increment subsubstreams too,
#     # but not necessary for this experiment.
#     responses, _ = mymodel.replicate(rng_list)
#     profit_vector[rep_idx] = responses["profit"]
#     n_prod_stockout_vector[rep_idx] = responses["n_prod_stockout"]
#     n_missed_orders_vector[rep_idx] = responses["n_missed_orders"]
#     fill_rate_vector[rep_idx] = responses["fill_rate"]
# print("profit", profit_vector)
# #print("n_stockout", n_prod_stockout_vector[rep_idx] = responses["n_prod_stockout"]
# print("n_unmet", n_missed_orders_vector)
# print("fill_rate", fill_rate_vector)


# IF RUNNING ALL SYSTEMS
ind_sampling_counter = 0
for prod1init in range(max_inventory):
    for prod2init in range(max_inventory):
        for prod3init in range(max_inventory):
            fixed_factors = {"num_prod": num_products,
                             "num_customer": num_customer,
                             "c_utility": c_utility,
                             "price": price,
                             "cost": cost,
                             "init_level": (prod1init, prod2init, prod3init)}  # This one is the decision variable. 
            # Initialize a model.
            mymodel = DynamNews(fixed_factors)
            # Initialize random number generators.
            # If doing independent sampling across systems.
            rng_list = [MRG32k3a(s_ss_sss_index=[ind_sampling_counter, ss, 0]) for ss in range(mymodel.n_rngs)]
            # If doing CRN across systems:
            # rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]
            for rep_idx in range(n):
                # Can manually increment subsubstreams too,
                # but not necessary for this experiment.
                responses, _ = mymodel.replicate(rng_list)
                profit_matrix[rep_idx][ind_sampling_counter] = responses["profit"]
                n_prod_stockout_matrix[rep_idx][ind_sampling_counter] = responses["n_prod_stockout"]
                n_missed_orders_matrix[rep_idx][ind_sampling_counter] = responses["n_missed_orders"]
                fill_rate_matrix[rep_idx][ind_sampling_counter] = responses["fill_rate"]
            # Increment counter.
            ind_sampling_counter += 1

# -----------------------------------------------

profit_matrix_df = pd.DataFrame(profit_matrix)
profit_matrix_df.to_excel('profits.xlsx', index=False)
fill_rate_df = pd.DataFrame(fill_rate_matrix)
fill_rate_df.to_excel('fill_rates.xlsx', index=False)

plt.scatter(np.mean(profit_matrix, axis=0), np.mean(fill_rate_matrix, axis=0), s=1)
plt.xlabel("average profit")
plt.ylabel("average fill rate")
plt.title("sample means based on 100 reps")
plt.show()