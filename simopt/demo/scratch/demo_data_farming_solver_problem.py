"""
This script is intended to help with running a data-farming experiment on
a simulation model. It creates a design of model factors and runs multiple
replications at each configuration of the model. Outputs are printed to a file.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# from models.mm1queue import MM1Queue
from data_farming_base import DesignPoint, DataFarmingExperiment #, DataFarmingMetaExperiment
from csv import DictReader


factor_headers = ["purchase_price", "sales_price", "salvage_price", "order_quantity"]
myexperiment = DataFarmingExperiment(model_name="CNTNEWS", factor_settings_filename="model_factor_settings", factor_headers=factor_headers, design_filename=None, model_fixed_factors={})
myexperiment.run(n_reps=10, crn_across_design_pts=False)
myexperiment.print_to_csv(csv_filename="cntnews_data_farming_output")

# solver_factor_headers = ["sample_size"]
# myMetaExperiment = DataFarmingMetaExperiment(solver_name="RNDSRCH",
#                                              problem_name="FACSIZE-2",
#                                              solver_factor_headers=solver_factor_headers,
#                                              solver_factor_settings_filename="", # solver_factor_settings",
#                                              design_filename="random_search_design",
#                                              solver_fixed_factors={},
#                                              problem_fixed_factors={},
#                                              model_fixed_factors={})
# myMetaExperiment.run(n_macroreps=20)
# myMetaExperiment.post_replicate(n_postreps=100, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)
# # myMetaExperiment.calculate_statistics() # solve_tols=[0.10], beta=0.50)
# # myMetaExperiment.print_to_csv(csv_filename="meta_raw_results")

# print("I ran this.")


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