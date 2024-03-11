"""
This script is intended to help with running a data-farming experiment on
a solver. It creates a design of solver factors and runs multiple
macroreplications at each version of the solver. Outputs are printed to a file.
"""
import os
import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
import pandas as pd
from simopt.experiment_base import ProblemsSolvers 

problem_name = 'CNTNEWS-1'


#fixed factors
# n_p = 3 # number of products
# n_m = 4 # number of materials
# mat_to_prod = [[1,2,3], [1,2,3],[1,2,3]] # maps materials to products (currently every product uses all materials)
# process_cost = [0.1, 0.1, 0.1] # processing cost per product unit
# order_cost = 20 # one time ordering cost
# purchase_yeild = [.9,.9,.9] # yeild rates for initially purchased materials
total_budget = 5000 # budget for all purchases
sales_price = [20,20,20,20] # sales price per product unit
# order_quantity = [20,20,20] # intial order quantity per material
# mean = [15,15,15] # mean parameter for poisson demand distribution

# problem factors
init_sol = [20, 20, 20, 20]
budget = 1000


# design options
n_stacks = 1

    
# turn design txt file into dataframe
df = pd.read_csv("expt1_test_inputs.txt", sep=' ', encoding="utf-8")


# get dps for each material type, adjust as needed to fit txt file format
c_m = df.iloc[:, :4]
c_r = df.iloc[:,4:8]
p_v = df.iloc[:,8:12]

print('c_m', c_m)
print('c_r', c_r)
print('p_v', p_v)
    
n_dp = len(df) # number of design points
problem_factors = [] # list to hold dictionary of problem factors for each dp of problem
problem_names = []

# concatinate dfs back into lists to match factor settings format
for row in range(n_dp):
    
    m_cost = [c_m.iloc[row,0], c_m.iloc[row,1], c_m.iloc[row,2], c_m.iloc[row,3]]
    r_cost = [c_r.iloc[row,0], c_r.iloc[row,1], c_r.iloc[row,2], c_r.iloc[row,2]]
    s_price = [p_v.iloc[row,0], p_v.iloc[row,1], p_v.iloc[row,2], p_v.iloc[row,3]]

    
    # create dictionary of all factor values at this dp
    dp_factors = {}
    prob_dp_factors = {} # holds fixed problem factor values
    dp = [] # list for current dp
    dp_factors["material_cost"] = m_cost
    dp_factors["recourse_cost"] = r_cost
    dp_factors["salvage_price"] = s_price
    # dp_factors["num_material"] = n_m
    # dp_factors["num_product"] = n_p
    # dp_factors["mat_to_prod"] = mat_to_prod
    # dp_factors["process_cost"] = process_cost 
    # dp_factors["order_cost"] = order_cost
    # dp_factors["purchase_yield"] = purchase_yeild
    # dp_factors["total_budget"] = total_budget
    # dp_factors["sales price"] = sales_price
    # dp_factors["order quantity"] = order_quantity
    # dp_factors["poi_mean"] = mean
    
    prob_dp_factors["initial_solution"] = init_sol
    prob_dp_factors["budget"] = budget
    dp.append(prob_dp_factors)
    dp.append(dp_factors)
    problem_factors.append(dp)
    problem_names.append(problem_name)

# solver options
solver_name = "RNDSRCH"

crn = True
#sample_size = 10

solver_factors = [{'crn_across_solns': crn, 'sample_size': 5}, {'crn_across_solns': crn, 'sample_size': 10}, {'crn_across_solns': crn, 'sample_size': 50} ]

solver_names = [solver_name, solver_name, solver_name]

print(problem_factors)


# call problemssolvers
experiment = ProblemsSolvers(problem_factors = problem_factors,
                              solver_factors = solver_factors,
                              problem_names = problem_names,
                              solver_names = solver_names)


# run experiment
n_macro = 10
n_post = 100
n_postnormal = 200

experiment.run(n_macroreps = n_macro)
experiment.post_replicate( n_postreps = n_post)
experiment.post_normalize(n_postreps_init_opt = n_postnormal)
experiment.record_group_experiment_results()
experiment.log_group_experiment_results()
experiment.report_group_statistics()






