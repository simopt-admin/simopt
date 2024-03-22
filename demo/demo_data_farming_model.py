"""
This script is intended to help with running a data-farming experiment on
a simulation model. It creates a design of model factors and runs multiple
replications at each configuration of the model. Outputs are printed to a file.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore


from simopt.data_farming_base import DataFarmingExperiment

# Specify the name of the model as it appears in directory.py
model_name = "CNTNEWS"

# Specify the names of the model factors (in order) that will be varied.
factor_headers = ["purchase_price", "sales_price", "salvage_price", "order_quantity"]

# If creating the design, provide the name of a .txt file containing
# the following:
#    - one row corresponding to each model factor being varied
#    - three columns:
#         - first column: lower bound for factor value
#         - second column: upper bound for factor value
#         - third column: (integer) number of digits for discretizing values
#                         (e.g., 0 corresponds to integral values for the factor)
# factor_settings_filename = "model_factor_settings"
factor_settings_filename = None

# OR, if the design has been created, provide the name of a .text file
# containing the following:
#    - one row corresponding to each design point
#    - the number of columns equal to the number of factors being varied
#    - each value in the table gives the value of the factor (col index)
#      for the design point (row index)
# E.g., design_filename = "model_factor_settings_design"
# design_filename = None
design_filename = "model_factor_settings_design"

# Specify a common number of replications to run of the model at each
# design point.
n_reps = 10

# Specify whether to use common random numbers across different versions
# of the model.
crn_across_design_pts = True

# Specify filename for outputs.
output_filename = "cntnews_data_farming_output"

# No code beyond this point needs to be edited.

# Create DataFarmingExperiment object.
myexperiment = DataFarmingExperiment(model_name=model_name,
                                     factor_settings_filename=factor_settings_filename,
                                     factor_headers=factor_headers,
                                     design_filename=design_filename,
                                     model_fixed_factors={}
                                     )

# Run replications and print results to file.
myexperiment.run(n_reps=n_reps, crn_across_design_pts=crn_across_design_pts)
myexperiment.print_to_csv(csv_filename=output_filename)
