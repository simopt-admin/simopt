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
n_p = 3 # number of products
n_m = 3 # number of materials
mat_to_prod = [[1,2,3], [1,2,3],[1,2,3]] # maps materials to products (currently every product uses all materials)
process_cost = [0.1, 0.1, 0.1] # processing cost per product unit
order_cost = 20 # one time ordering cost
purchase_yeild = [.9,.9,.9] # yeild rates for initially purchased materials
total_budget = 600 # budget for all purchases
sales_price = [12,12,12] # sales price per product unit
order_quantity = [20,20,20] # intial order quantity per material
mean = [15,15,15] # mean parameter for poisson demand distribution

# problem factors
init_sol = [40, 40, 100]
budget = 1000

# datafarming factors
max_m_cost = [5,5,5] # upper bound on material cost for each material type
min_m_cost = [1,2,3] # lower bound on material cost for each material type
dec_m_cost = [2,2,2] # number of decimals to vary in design for each material cost


max_r_cost = [10,10,10] # upper bound on recourse cost for each material type
min_r_cost = [7,7,7] # lower bound on recourse cost for each material type
dec_r_cost = [2,2,2] # number of decimals to vary in design for each recourse cost


max_s_price = [.6,.6,.6] # max salvage price per material unit
min_s_price = [.1,.1,.1] # min salvage price per material unit
dec_s_price = [2,2,2] # number of decimals to vary in design for each salvage price


# design options
n_stacks = 1



# create factor text file for each set of materials and call ruby to create design txt file
factor_settings_filename = 'factor_settings_file'
with open(f"{factor_settings_filename}.txt", 'w') as file:
        file.write("")
for i in range(n_m):
    with open(f"{factor_settings_filename}.txt", 'a') as file:
        file.write(f"{max_m_cost[i]}\t{min_m_cost[i]}\t{dec_m_cost[i]}\n") # write material cost factor settings to file
        file.write(f"{max_r_cost[i]}\t{min_r_cost[i]}\t{dec_r_cost[i]}\n") # write recourse cost factor settings to file
        file.write(f"{max_s_price[i]}\t{min_s_price[i]}\t{dec_s_price[i]}\n") # write salvage factor settings to file
        
# use ruby to create design txt file
command = f"stack_nolhs.rb -s {n_stacks} {factor_settings_filename}.txt > {factor_settings_filename}_design.txt"
os.system(command)
    
# turn design txt file into dataframe
df = pd.read_csv(f"{factor_settings_filename}_design.txt", header=None, delimiter="\t", encoding="utf-8")

# get dps for each material type
m_1 = df.iloc[:, :3]
m_2 = df.iloc[:,3:6]
m_3 = df.iloc[:,6:9]


    
n_dp = len(df) # number of design points
problem_factors = [] # list to hold dictionary of problem factors for each dp of problem
problem_names = []

# concatinate dfs back into lists to match factor settings format
for row in range(n_dp):
    
    m_cost = [m_1.iloc[row,0], m_2.iloc[row,0], m_3.iloc[row,0]]
    r_cost = [m_1.iloc[row,1], m_2.iloc[row,1], m_3.iloc[row,1]]
    s_price = [m_1.iloc[row,2], m_2.iloc[row,2], m_3.iloc[row,2]]

    
    # create dictionary of all factor values at this dp
    dp_factors = {}
    prob_dp_factors = {} # holds fixed problem factor values
    dp = [] # list for current dp
    dp_factors["material_cost"] = m_cost
    dp_factors["recourse_cost"] = r_cost
    dp_factors["salvage_price"] = s_price
    dp_factors["num_material"] = n_m
    dp_factors["num_product"] = n_p
    dp_factors["mat_to_prod"] = mat_to_prod
    dp_factors["process_cost"] = process_cost 
    dp_factors["order_cost"] = order_cost
    dp_factors["purchase_yield"] = purchase_yeild
    dp_factors["total_budget"] = total_budget
    dp_factors["sales price"] = sales_price
    dp_factors["order quantity"] = order_quantity
    dp_factors["poi_mean"] = mean
    
    prob_dp_factors["initial_solution"] = init_sol
    prob_dp_factors["budget"] = budget
    dp.append(prob_dp_factors)
    dp.append(dp_factors)
    problem_factors.append(dp)
    problem_names.append(problem_name)

# solver options
solver_name = "RNDSRCH"

crn = True
sample_size = 10

solver_factors = [{'crn_across_solns': crn, 'sample_size': sample_size}]

solver_names = [solver_name]

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






