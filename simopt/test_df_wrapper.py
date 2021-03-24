import numpy as np
from rng.mrg32k3a import MRG32k3a
# from oracles.mm1queue import MM1Queue
from data_farming_base import DesignPoint, DataFarmingExperiment
from csv import DictReader

factor_headers = ["purchase_price", "sales_price", "salvage_price", "order_quantity"]
myexperiment = DataFarmingExperiment(oracle_name="CNTNEWS", factor_settings_filename="oracle_factor_settings", factor_headers=factor_headers, design_filename=None, oracle_fixed_factors={})
myexperiment.run(n_reps=10, crn_across_design_pts=False)
myexperiment.print_to_csv(csv_filename="cntnews_data_farming_output")

print("I ran this.")


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
