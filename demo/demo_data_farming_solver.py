"""
This script is intended to help with running a data-farming experiment on
a solver. It creates a design of solver factors and runs multiple
macroreplications at each version of the solver. Outputs are printed to a file.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))


from data_farming_base import DataFarmingMetaExperiment


# Specify the name of the solver as it appears in directory.py
# solver_name = "RNDSRCH"
solver_name = "ASTRODF"

# Specify the names of the model factors (in order) that will be varied.
# solver_factor_headers = ["sample_size"]
solver_factor_headers = ["eta_1", "eta_2"]

# Specify the name of the problem as it appears in directory.py
# problem_name = "FACSIZE-2"
problem_name = "SSCONT-1"

# If creating the design, provide the name of a .txt file containing
# the following:
#    - one row corresponding to each solver factor being varied
#    - three columns:
#         - first column: lower bound for factor value
#         - second column: upper bound for factor value
#         - third column: (integer) number of digits for discretizing values
#                         (e.g., 0 corresponds to integral values for the factor)
# solver_factor_settings_filename = "solver_factor_settings"
solver_factor_settings_filename = None

# OR, if the design has been created, provide the name of a .text file
# containing the following:
#    - one row corresponding to each design point
#    - the number of columns equal to the number of factors being varied
#    - each value in the table gives the value of the factor (col index)
#      for the design point (row index)
# E.g., design_filename = "solver_factor_settings_design"
# design_filename = None
# design_filename = "random_search_design"
design_filename = "astrodf_design"

# OPTIONAL: Provide additional overrides for default solver/problem/model factors.
# If empty, default factor settings are used.
solver_fixed_factors = {}
problem_fixed_factors = {}
model_fixed_factors = {}

# Specify a common number of macroreplications of each version of the solver
# to run on the problem, i.e., the number of runs at each design point.
n_macroreps = 3

# NOT YET IMPLEMENTED.
# Specify whether to use common random numbers across different design points.
# Default is to use CRN across design points since each design point is a
# ProblemSolver instance.
# crn_across_design_pts = True

# Specify the number of postreplications to take at each recommended solution
# from each macroreplication at each design point.
n_postreps = 100

# Specify the number of postreplications to take at x0 and x*.
n_postreps_init_opt = 200

# Specify the CRN control for postreplications.
crn_across_budget = True  # Default
crn_across_macroreps = False  # Default
crn_across_init_opt = True  # Default

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

# Compute the performance metrics at each design point and print to csv.
myDFMetaExperiment.report_statistics(solve_tols=[0.05, 0.10, 0.20, 0.50])
