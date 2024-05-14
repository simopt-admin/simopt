"""
This script is intended to help with debugging a problem.
It imports a problem, initializes a problem object with given factors,
sets up pseudorandom number generators, and runs multiple replications
at a given solution.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import random number generator.
from mrg32k3a.mrg32k3a import MRG32k3a

# Import the Solution class.
from simopt.base import Solution

import pandas as pd

# Import problem.

# from models.<filename> import <problem_class_name>
# Replace <filename> with name of .py file containing problem class.
# Replace <problem_class_name> with name of problem class.

# Fix factors of problem. Specify a dictionary of factors.

# fixed_factors = {}  # Resort to all default values.
# Look at Problem class definition to get names of factors.

# Initialize an instance of the specified problem class.

# myproblem = <problem_class_name>(fixed_factors=fixed_factors)
# Replace <problem_class_name> with name of problem class.

# Initialize a solution x corresponding to the problem.

# x = (,)
# Look at the Problem class definition to identify the decision variables.
# x will be a tuple consisting of the decision variables.

# The following line does not need to be changed.
# mysolution = Solution(x, myproblem)

# Working example for CntNVMaxProfit problem.
# -----------------------------------------------
from simopt.models.cntnv import CntNVMaxProfit

#fixed factors (uncomment any of these if you want to change from problem default)
# n_p = 3 # number of products
# n_m = 4 # number of materials
mat_to_prod = [[1, 1, 1, 0], [1, 2, 2, 0],[0, 0, 3, 3]] # maps materials to products (currently every product uses all materials)
# process_cost = [0.1, 0.1, 0.1,0.1] # processing cost per product unit
order_cost = 0 # one time ordering cost
purchase_yeild = [1,1,1,1] # yeild rates for initially purchased materials
total_budget = 1200 # budget for all purchases
sales_price = [6,12,20] # sales price per product unit
# order_quantity = [20,20,20,20] # intial order quantity per material
mean = [20,25,15] # mean parameter for poisson demand distribution

# problem factors (can change these as desired, comment out to use problem defaults)
init_sol = [20, 20, 20, 20]
budget = 500
 
# turn design txt file into dataframe
design_filename = "cnt_design.xlsx" # location of design file
df = pd.read_excel(design_filename) # may need to change sep depending on type of file uploaded

csv_filename = './demo_problem_results/results.csv'

solutions = [(45.0808, 77.2949, 119.5386, 50.0416),(48.3686, 73.9203, 106.1471, 52.428)]

headers = ['Problem Design Point', 'Solution', 'material_cost', 'recourse_cost', 'salvage_price', 'Mean Objective Estimate', 'Standard Error']
results = pd.DataFrame(columns = headers)
# concatinate tables back into arrays for each design point
problem_renames = []
for index, row in df.iterrows():
    # get problem factor arrays for each design point
    # will need to add an additional index to each array if increaseing the number of materials
    # will need to create an additional array if varying more than 3 problem factors
    cm_1 = row['cm1']
    cm_2 = row['cm2']
    cm_3 = row['cm3']
    cm_4 = row['cm4']
    rm_1 = row['rm1']
    rm_2 = row['rm2']
    rm_3 = row['rm3']
    rm_4 = row['rm4']
    sm_1 = row['sm1']
    sm_2 = row['sm2']
    sm_3 = row['sm3']
    sm_4 = row['sm4']
    m_cost = [round(cm_1,4), round(cm_2,4), round(cm_3,4), round(cm_4,4)]
    r_cost = [round(rm_1,4), round(rm_2,4), round(rm_3,4), round(rm_4,4)]
    s_price = [round(sm_1,4), round(sm_2,4), round(sm_3,4), round(sm_4,4)]


    
    # create dictionary of all factor values at this dp
    # uncomment correspoding fixed factor lines to change from default problem value
    dp_factors = {}
    prob_factors = {}
    dp_factors["material_cost"] = m_cost
    dp_factors["recourse_cost"] = r_cost
    dp_factors["salvage_price"] = s_price
    # dp_factors["num_material"] = n_m
    # dp_factors["num_product"] = n_p
    dp_factors["mat_to_prod"] = mat_to_prod
    # dp_factors["process_cost"] = process_cost 
    dp_factors["order_cost"] = order_cost
    dp_factors["purchase_yield"] = purchase_yeild
    dp_factors["total_budget"] = total_budget
    dp_factors["sales_price"] = sales_price
    # dp_factors["order quantity"] = order_quantity
    dp_factors["poi_mean"] = mean    
    prob_factors["initial_solution"] = init_sol
    prob_factors["budget"] = budget


    fixed_factors = prob_factors
    model_fixed_factors = dp_factors

    myproblem = CntNVMaxProfit(fixed_factors=fixed_factors, model_fixed_factors = model_fixed_factors)
    for sol in solutions:
        if sol == solutions[0]:
            dp_append = 'a'
        else:
            dp_append = 'b'
        dp = f'{index}_{dp_append}'
        x = sol
        mysolution = Solution(x, myproblem)
        # -----------------------------------------------
        
        # Another working example for FacilitySizingTotalCost problem. (Commented out)
        # This example has stochastic constraints.
        # -----------------------------------------------
        # from models.facilitysizing import FacilitySizingTotalCost
        # fixed_factors = {"epsilon": 0.1}
        # myproblem = FacilitySizingTotalCost(fixed_factors=fixed_factors)
        # x = (200, 200, 200)
        # mysolution = Solution(x, myproblem)
        # -----------------------------------------------
        
        
        # The rest of this script requires no changes.
        
        # Create and attach rngs to solution
        rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(myproblem.model.n_rngs)]
        mysolution.attach_rngs(rng_list, copy=False)
        
        # Simulate a fixed number of replications (n_reps) at the solution x.
        n_reps = 200
        myproblem.simulate(mysolution, m=n_reps)
        
        mean_estimate = mysolution.objectives_mean[0]
        stdev = mysolution.objectives_stderr[0]
        
        # add results to dataframe
        new_row = [dp, sol, m_cost, r_cost, s_price, mean_estimate, stdev ]
        
        results.loc[len(results)] = new_row
        
        # Print results to console.
        print(myproblem.factors)
        #print('new mean', mean_estimate)
        print(f"Ran {n_reps} replications of the {myproblem.name} problem at solution x = {x}.\n")
        print(f"The mean objective estimate was {round(mysolution.objectives_mean[0], 4)} with standard error {round(mysolution.objectives_stderr[0], 4)}.")
        print("The individual observations of the objective were:")
        for idx in range(n_reps):
            print(f"\t {round(mysolution.objectives[idx][0], 4)}")
        if myproblem.gradient_available:
            print("\nThe individual observations of the gradients of the objective were:")
            for idx in range(n_reps):
                print(f"\t {[round(g, 4) for g in mysolution.objectives_gradients[idx][0]]}")
        else:
            print("\nThis problem has no known gradients.")
        if myproblem.n_stochastic_constraints > 0:
            print(f"\nThis problem has {myproblem.n_stochastic_constraints} stochastic constraints of the form E[LHS] <= 0.")
            for stc_idx in range(myproblem.n_stochastic_constraints):
                print(f"\tFor stochastic constraint #{stc_idx + 1}, the mean of the LHS was {round(mysolution.stoch_constraints_mean[stc_idx], 4)} with standard error {round(mysolution.stoch_constraints_stderr[stc_idx], 4)}.")
                print("\tThe observations of the LHSs were:")
                for idx in range(n_reps):
                    print(f"\t\t {round(mysolution.stoch_constraints[idx][stc_idx], 4)}")
        else:
            print("\nThis problem has no stochastic constraints.")

results.to_csv(csv_filename, index = False)