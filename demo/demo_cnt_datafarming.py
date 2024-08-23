"""
This script is intended to help with running a data-farming experiment on
a solver. It creates a design of solver factors and runs multiple
macroreplications at each version of the solver. Outputs are printed to a file.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))


from simopt.data_farming_base import DataFarmingMetaExperiment, DesignPoint
from simopt.experiment_base import create_design, ProblemsSolvers



# Specify the name of the solver as it appears in directory.py
# solver_name = "RNDSRCH"
solver_name = "ASTRODF"

# Specify the names of the model factors (in order) that will be varied.
# solver_factor_headers = ["sample_size"]
solver_factor_headers = ["eta_1", "eta_2", "lambda_min" ]


#solver factors chosen for cross design, factor name followed by list containing factor values to cross design over
#cross_design_factors = {'crn_across_solns': [True, False] }#,'easy_solve': [True,False], 'reuse_points': [True, False]}

cross_design_factors = {'crn_across_solns': [True, False]}
# If creating the design, provide the name of a .txt file containing
# the following:
#    - one row corresponding to each solver factor being varied
#    - three columns:
#         - first column: lower bound for factor value
#         - second column: upper bound for factor value
#         - third column: (integer) number of digits for discretizing values
#                         (e.g., 0 corresponds to integral values for the factor)
solver_factor_settings_filename = "astrodf_testing"
#solver_factor_settings_filename = None

solver_list_no_design = [{"eta_1": .5, "eta_2": .75, "lambda_min": 50}, {"eta_1": .3, "eta_2": .4, "lambda_min": 80}, {'n_reps': 20, "n_loss": 3} ]
solver_names_no_design = ["ASTRODF","ASTRODF", "SPSA"]

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





# Create DataFarmingExperiment object.
solver_design_list = create_design(name=solver_name,
                factor_headers=solver_factor_headers,
                factor_settings_filename=solver_factor_settings_filename,
                fixed_factors=solver_fixed_factors,
                cross_design_factors= cross_design_factors,
                )
print(solver_design_list)

solver_names = []
for i in range(len(solver_design_list)): #this part will be done in GUI
    solver_names.append(solver_name)
    

# run with problem

problem_name = "CNTNEWS-1"
 
problem_factor_headers = ["budget"]

problem_factor_settings_filename = "testing_problem_cntnews"

problem_fixed_factors = {}

prob_no_design_list = [[{"budget": 1000},{"purchase_price": 10, "sales_price": 15, "order_quantity": 200}],
                       [{"budget": 200}, {"demand_mean": 2000, "lead_mean": 5 }]]

prob_names_no_design = ["CNTNEWS-1", "SSCONT-1"]

# Create DataFarmingExperiment object.
problem_design_list = create_design(name=problem_name,
                factor_headers=problem_factor_headers,
                factor_settings_filename=problem_factor_settings_filename,
                fixed_factors=problem_fixed_factors,
                #cross_design_factors= None,
                )
print(problem_design_list)

problem_names = []


# run for problem model values (will be assigned to model_fixed_factors using GUI)
model_name = "CNTNEWS"
 
model_factor_headers = ["purchase_price", "sales_price", "order_quantity"]

model_factor_settings_filename = "testing_model_cntnews"

model_fixed_factors = {"salvage_price": 5, "Burr_c": 1}

# Create DataFarmingExperiment object.
model_design_list = create_design(name=model_name,
                factor_headers=model_factor_headers,
                factor_settings_filename=model_factor_settings_filename,
                fixed_factors=model_fixed_factors,
                #cross_design_factors= None,
                )
print(model_design_list)


 
print(solver_names, problem_names)

# Combine model design with fixed problem factors
fixed_prob_model_design = []
fixed_prob_model_design_names = []

for index, prob_dp in enumerate(prob_no_design_list):
    current_prob_name = prob_names_no_design[index]
    for model_dp in model_design_list:
        dp = []
        dp.append(prob_dp)
        dp.append(model_dp)
        fixed_prob_model_design.append(dp)
        fixed_prob_model_design_names.append(current_prob_name)
    
    

# combine model and problem factors
prob_model_design_list = [] # list of lists, each inner list will hold a prob dp matched with a model dp
for prob_dp in problem_design_list:
    for model_dp in model_design_list:
        dp = []
        dp.append(prob_dp)
        dp.append(model_dp)
        prob_model_design_list.append(dp)
        
for i in range(len(prob_model_design_list)):
    problem_names.append(problem_name)
            
        
print('prob model list', prob_model_design_list)



experiment = ProblemsSolvers(solver_factors = solver_design_list,
                             #solver_factors = solver_list_no_design,
                              problem_factors = prob_no_design_list,
                              #problem_factors = fixed_prob_model_design,
                              solver_names = solver_names,
                              #solver_names = solver_names_no_design,
                              #problem_names = problem_names
                              problem_names = prob_names_no_design
                              #problem_names = fixed_prob_model_design_names
                              )

experiment.check_compatibility()
experiment.run(n_macroreps = 5)
experiment.post_replicate( n_postreps = 2)
experiment.post_normalize(n_postreps_init_opt = 100)
experiment.record_group_experiment_results()
experiment.log_group_experiment_results()
experiment.report_group_statistics()