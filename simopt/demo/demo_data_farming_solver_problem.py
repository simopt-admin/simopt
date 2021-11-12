"""
This script is intended to help with running a data-farming experiment on
a simulation model. It creates a design of model factors and runs multiple
replications at each configuration of the model. Outputs are printed to a file.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))


from data_farming_base import DataFarmingMetaExperiment
from csv import DictReader


# Specify the name of the solver as it appears in directory.py
solver_name = "RNDSRCH"

# Specify the names of the model factors (in order) that will be varied.
solver_factor_headers = ["sample_size"]

# Specify the name of the problem as it appears in directory.py
problem_name = "FACSIZE-2"

# If creating the design, provide the name of a .txt file containing
# the following:
#    - one row corresponding to each solver factor being varied
#    - three columns:
#         - first column: lower bound for factor value
#         - second column: upper bound for factor value
#         - third column: (integer) number of digits for discretizing values
#                         (e.g., 0 corresponds to integral values for the factor)
#solver_factor_settings_filename = "solver_factor_settings"
solver_factor_settings_filename = None

# OR, if the design has been created, provide the name of a .text file
# containing the following:
#    - one row corresponding to each design point
#    - the number of columns equal to the number of factors being varied
#    - each value in the table gives the value of the factor (col index)
#      for the design point (row index)
# E.g., design_filename = "solver_factor_settings_design"
#design_filename = None
design_filename = "random_search_design"

# OPTIONAL: Provide additional overrides for default solver/problem/model factors.
# If empty, default factor settings are used.
solver_fixed_factors = {}
problem_fixed_factors = {}
model_fixed_factors = {}

# Specify a common number of macroreplications of each version of the solver
# to run on the problem, i.e., the number of runs at each design point.
n_macroreps = 5

### NOT YET IMPLEMENTED.
# Specify whether to use common random numbers across different design points.
# crn_across_design_pts = True

# Specify the number of postreplications to take at each recommended solution
# from each macroreplication at each design point.
n_postreps = 100

# Specify the number of postreplications to take at x0 and x*.
n_postreps_init_opt = 200

# Specify the CRN control for postreplications.
crn_across_budget=True  # Default
crn_across_macroreps=False  # Default
crn_across_init_opt=True  # Default

# No code beyond this point needs to be edited.

# Create DataFarmingExperiment object.
myDFMetaExperiment = DataFarmingMetaExperiment(solver_name=solver_name,
                                               problem_name=problem_name,
                                               solver_factor_headers=solver_factor_headers,
                                               solver_factor_settings_filename=solver_factor_settings_filename,
                                               design_filename=design_filename,
                                               solver_fixed_factors=solver_fixed_factors,
                                               problem_fixed_factors=problem_fixed_factors,
                                               model_fixed_factors=model_fixed_factors
                                               )

# Run macroreplications at each design point.
myDFMetaExperiment.run(n_macroreps=n_macroreps)

# Postprocess the experimental results from each design point.
myDFMetaExperiment.post_replicate(n_postreps=n_postreps,
                                  crn_across_budget=crn_across_budget,
                                  crn_across_macroreps=crn_across_macroreps
                                  )
myDFMetaExperiment.post_normalize(n_postreps_init_opt=n_postreps_init_opt,
                                  crn_across_init_opt=crn_across_init_opt
                                  )

# myMetaExperiment.calculate_statistics() # solve_tols=[0.10], beta=0.50)
# myMetaExperiment.print_to_csv(csv_filename="meta_raw_results")


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