import numpy as np
from rng.mrg32k3a import MRG32k3a
# from oracles.mm1queue import MM1Queue
from data_farming_base import DesignPoint, DataFarmingExperiment, DataFarmingMetaExperiment
from csv import DictReader


factor_headers = ["purchase_price", "sales_price", "salvage_price", "order_quantity"]
myexperiment = DataFarmingExperiment(oracle_name="CNTNEWS", factor_settings_filename="oracle_factor_settings", factor_headers=factor_headers, design_filename=None, oracle_fixed_factors={})
myexperiment.run(n_reps=10, crn_across_design_pts=False)
myexperiment.print_to_csv(csv_filename="cntnews_data_farming_output")

# solver_factor_headers = ["sample_size"]
# myMetaExperiment = DataFarmingMetaExperiment(solver_name="RNDSRCH",
#                                              problem_name="CNTNEWS-1",
#                                              solver_factor_settings_filename="solver_factor_settings",
#                                              solver_factor_headers=solver_factor_headers, 
#                                              design_filename=None,
#                                              solver_fixed_factors={},
#                                              problem_fixed_factors={}, 
#                                              oracle_fixed_factors={})
# myMetaExperiment.run(n_macroreps=2, crn_across_solns=True)
# myMetaExperiment.post_replicate(n_postreps=50, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)
# myMetaExperiment.calculate_statistics(solve_tol=0.10, beta=0.50)
# myMetaExperiment.print_to_csv(csv_filename="meta_raw_results.csv")

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
