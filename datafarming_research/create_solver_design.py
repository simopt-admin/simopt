"""
This script is intended to help with running a data-farming experiment on
a solver. It creates a design of solver factors and runs multiple
macroreplications at each version of the solver. Outputs are printed to a file.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))


from simopt.data_farming_base import DataFarmingMetaExperiment, DesignPoint


# Specify the name of the solver as it appears in directory.py
# solver_name = "RNDSRCH"
solver_name = "ASTRODF"

# Specify the names of the model factors (in order) that will be varied.
# solver_factor_headers = ["sample_size"]
solver_factor_headers = ["gamma_1", "gamma_2", "eta_1", "eta_2"]


#solver factors chosen for cross design, factor name followed by list containing factor values to cross design over
#cross_design_factors = {'crn_across_solns': [True, False] }#,'easy_solve': [True,False], 'reuse_points': [True, False]}

cross_design_factors = {}
# If creating the design, provide the name of a .txt file containing
# the following:
#    - one row corresponding to each solver factor being varied
#    - three columns:
#         - first column: lower bound for factor value
#         - second column: upper bound for factor value
#         - third column: (integer) number of digits for discretizing values
#                         (e.g., 0 corresponds to integral values for the factor)
solver_factor_settings_filename = "ASTRODF_factor_settings"
#solver_factor_settings_filename = None

# OR, if the design has been created, provide the name of a .text file
# containing the following:
#    - one row corresponding to each design point
#    - the number of columns equal to the number of factors being varied
#    - each value in the table gives the value of the factor (col index)
#      for the design point (row index)
# E.g., design_filename = "solver_factor_settings_design"
design_filename = None
# design_filename = "random_search_design"
#design_filename = "astrodf_design"
csv_filename = None
#csv_filename = './data_farming_experiments/astrodf_factor_setting_design.csv'

# OPTIONAL: Provide additional overrides for default solver/problem/model factors.
# If empty, default factor settings are used.
solver_fixed_factors = {}


# No code beyond this point needs to be edited.

# Create DataFarmingExperiment object.
myDFMetaExperiment = DataFarmingMetaExperiment(solver_name=solver_name,
                                               #problem_name=problem_name,
                                               solver_factor_headers=solver_factor_headers,
                                               solver_factor_settings_filename=solver_factor_settings_filename,
                                               design_filename=design_filename,
                                               solver_fixed_factors=solver_fixed_factors,
                                               #problem_fixed_factors=problem_fixed_factors,
                                               #model_fixed_factors=model_fixed_factors,
                                               cross_design_factors= cross_design_factors,
                                               csv_filename = csv_filename
                                               )


