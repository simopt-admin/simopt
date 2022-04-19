"""
This script is intended to help with debugging an model.
It imports an model, initializes an model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""
import numpy as np
import sys
import os.path as o
import matplotlib.pyplot as plt

sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import random number generator.
from rng.mrg32k3a import MRG32k3a

# Import model.
# Replace <filename> with name of .py file containing model class.
# Replace <model_class_name> with name of model class.
# Ex: from models.mm1queue import MM1Queue
from models.ironore import IronOre
from models.dynamnews import DynamNews
from models.covid_individual import COVID

# Fix factors of model. Specify a dictionary of factors.
# Look at Model class definition to get names of factors.
# Ex: for the MM1Queue class,
#     fixed_factors = {"lambda": 3.0,
#                      "mu": 8.0}
# c_utility = []
# for j in range(1, 11):
#     c_utility.append(5 + j)

# fixed_factors = {
#     "num_prod": 10,
#     "num_customer": 30,
#     "c_utility": c_utility,
#     "init_level": 3 * np.ones(10),
#     "price": 9 * np.ones(10),
#     "cost": 5 * np.ones(10)}  
fixed_factors = {} # Resort to all default values.

# Initialize an instance of the specified model class.
# Replace <model_class_name> with name of model class.
# Ex: mymodel = MM1Queue(fixed_factors)
mymodel = COVID(fixed_factors)

# Working example for MM1 model. (Commented out)
# -----------------------------------------------
# from models.mm1queue import MM1Queue
# fixed_factors = {"lambda": 3.0, "mu": 8.0}
# mymodel = MM1Queue(fixed_factors)
# -----------------------------------------------

# The rest of this script requires no changes.

# Check that all factors describe a simulatable model.
# Check fixed factors individually.
for key, value in mymodel.factors.items():
    print(f"The factor {key} is set as {value}. Is this simulatable? {bool(mymodel.check_simulatable_factor(key))}.")
# Check all factors collectively.
print(f"Is the specified model simulatable? {bool(mymodel.check_simulatable_factors())}.")

# Create a list of RNG objects for the simulation model to use when
# running replications.
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]

# Run a single replication of the model.
responses, gradients = mymodel.replicate(rng_list)
print("\nFor a single replication:")
print("\nResponses:")
for key, value in responses.items():
    print(f"\t {key} is {value}.")
# print("\n Gradients:")
# for outerkey in gradients:
#     print(f"\tFor the response {outerkey}:")
#     for innerkey, value in gradients[outerkey].items():
#         print(f"\t\tThe gradient w.r.t. {innerkey} is {value}.")

plt.plot(np.arange(0, mymodel.factors["n"]), np.sum(responses['num_infected'], axis = 1), color = 'green', label = 'Infected')
plt.plot(np.arange(0, mymodel.factors["n"]), np.sum(responses['num_exposed'], axis = 1), color = 'orange', label = 'Exposed')
plt.plot(np.arange(0, mymodel.factors["n"]), np.sum(responses['num_susceptible'], axis = 1), color = 'blue', label = 'Susceptible')
plt.plot(np.arange(0, mymodel.factors["n"]), np.sum(responses['num_recovered'], axis = 1), color = 'red', label = 'Recovered')
plt.plot(np.arange(0, mymodel.factors["n"]), np.sum(responses['num_isolation'], axis = 1), color = 'purple', label = 'Isolation')
plt.legend()
plt.xlabel('Number of simulation days')
plt.ylabel('Population')
plt.yticks(np.arange(0, int(max(np.sum(responses['num_susceptible'], axis = 1))), 2000))
# plt.ylim(0, 1000)

plt.savefig('seir_no_test.png', bbox_inches = 'tight')


# --------------------------------------------------
# plt.figure()
# f, axes = plt.subplots(2,1)
# axes[1].plot(np.arange(0, mymodel.factors["n"]), np.sum(responses['num_infected'], axis = 1), color = 'green', label = 'Infected')
# axes[1].plot(np.arange(0, mymodel.factors["n"]), np.sum(responses['num_exposed'], axis = 1), color = 'orange', label = 'Exposed')
# axes[0].plot(np.arange(0, mymodel.factors["n"]), np.sum(responses['num_susceptible'], axis = 1), color = 'blue', label = 'Susceptible')
# axes[1].plot(np.arange(0, mymodel.factors["n"]), np.sum(responses['num_recovered'], axis = 1), color = 'red', label = 'Recovered')
# axes[1].plot(np.arange(0, mymodel.factors["n"]), np.sum(responses['num_isolation'], axis = 1), color = 'purple', label = 'Isolation')
# axes[0].legend()
# axes[1].legend()
# axes[0].set_ylabel('Population')
# axes[1].set_xlabel('Number of simulation days')
# axes[1].set_ylabel('Population')
# plt.savefig('seir.png', bbox_inches = 'tight')

# for i in range(mymodel.factors["num_groups"]):
#     fig1, ax1 = plt.subplots()
#     ax1.plot(np.arange(0, mymodel.factors["n"]), responses['num_infected'][:,i], color = 'green', label = 'num_infected in group '+str(i))
#     ax1.plot(np.arange(0, mymodel.factors["n"]), responses['num_exposed'][:,i], color = 'orange', label = 'num_exposed in group '+str(i))
#     ax1.plot(np.arange(0, mymodel.factors["n"]), responses['num_susceptible'][:,i], color = 'blue', label = 'num_susceptible in group '+str(i))
#     ax1.plot(np.arange(0, mymodel.factors["n"]), responses['num_recovered'][:,i], color = 'red', label = 'num_recovered in group '+str(i))  
#     ax1.plot(np.arange(0, mymodel.factors["n"]), responses['num_isolation'][:,i], color = 'purple', label = 'num_isolation in group '+str(i))
#     ax1.legend()
#     fig1.savefig('seir_group'+str(i)+'.png', bbox_inches = 'tight')
