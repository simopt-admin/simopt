import numpy as np
from rng.mrg32k3a import MRG32k3a
from oracles.mm1queue import MM1Queue
from data_farming_base import DesignPoint, DataFarmingExperiment
from csv import DictReader

oracle_fixed_factors = {'mu': 5} # default overrides from GUI, others set as defaults
#myoracle = MM1Queue(fixed_factors=oracle_fixed_factors)
#design_pt_factors = {'lambda': 1} # extracted from row of design matrix
#myoracle.factors.update(design_pt_factors)

# # Create a design point
# mydesignpt = DesignPoint(oracle=myoracle)

# rng_list = [MRG32k3a(s_ss_sss_index = [0, ss, 0]) for ss in range(myoracle.n_rngs)]
# for rng in rng_list:
#     print(rng.s_ss_sss_index)
# print("setup complete")
# mydesignpt.oracle.attach_rngs(rng_list)

# for rng in mydesignpt.oracle.rng_list:
#     print(rng.s_ss_sss_index)
# print("Received rngs")

# # Run 10 replications of the oracle at the design_point
# mydesignpt.simulate(m=10)

oracle_name = "MM1Queue"

myexperiment = DataFarmingExperiment(oracle_name=oracle_name, oracle_fixed_factors=oracle_fixed_factors, design_filename=None)
myexperiment.run(n_reps=10, crn_across_design_pts=False)

print('I ran this.')

# SCRATCH
# --------------------------------
# from csv import DictReader
# # open file in read mode
# with open('example_design_matrix.csv', 'r') as read_obj:
#     # pass the file object to DictReader() to get the DictReader object
#     csv_dict_reader = DictReader(read_obj)
#     # iterate over each line as a ordered dictionary
#     for row in csv_dict_reader:
#         # row variable is a dictionary that represents a row in csv
#         print(row)
