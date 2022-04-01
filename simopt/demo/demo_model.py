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
print("\n Gradients:")
for outerkey in gradients:
    print(f"\tFor the response {outerkey}:")
    for innerkey, value in gradients[outerkey].items():
        print(f"\t\tThe gradient w.r.t. {innerkey} is {value}.")

plt.plot(np.arange(0, mymodel.factors["n"]), responses['num_infected'], color = 'green', label = 'num_infected')
plt.plot(np.arange(0, mymodel.factors["n"]), responses['num_exposed'], color = 'orange', label = 'num_exposed')
plt.plot(np.arange(0, mymodel.factors["n"]), responses['num_susceptible'], color = 'blue', label = 'num_susceptible')
plt.plot(np.arange(0, mymodel.factors["n"]), responses['num_recovered'], color = 'red', label = 'num_recovered')
plt.legend()
plt.savefig('seir.png')

# import numpy as np
# total_dist = []
# for _ in range(1000):
#     responses, gradients = mymodel.replicate(rng_list)
#     total_dist.append(responses["total_dist"])

# print(np.mean(total_dist))
