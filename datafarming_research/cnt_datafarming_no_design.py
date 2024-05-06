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

def main():
    problem_name = 'CNTNEWS-1'
    
    
    #fixed factors (uncomment any of these if you want to change from problem default)
    n_p = 3 # number of products
    n_m = 4 # number of materials
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
    m_cost = [1,1,1.1,1.21]
    r_cost = [2,2,2.2,42]
    s_price = [.5,.5,.55,.605]
    # problem factors
    init_sol = [20, 20, 20, 20]
    budget = 1000
    
    
    problem_names = []
    dp =[]
    problem_factors = []
    # create dictionary of all factor values at this dp
    dp_factors = {}
    prob_dp_factors = {} # holds fixed problem factor values
    dp = [] # list for current dp
    dp_factors["material_cost"] = m_cost
    dp_factors["recourse_cost"] = r_cost
    dp_factors["salvage_price"] = s_price
    dp_factors["num_material"] = n_m
    dp_factors["num_product"] = n_p
    # dp_factors["mat_to_prod"] = mat_to_prod
    # dp_factors["process_cost"] = process_cost 
    # dp_factors["order_cost"] = order_cost
    # dp_factors["purchase_yield"] = purchase_yeild
    # dp_factors["total_budget"] = total_budget
    # dp_factors["sales price"] = sales_price
    # dp_factors["order quantity"] = order_quantity
    # dp_factors["poi_mean"] = mean
    
    dp_factors["initial_solution"] = init_sol
    dp_factors["budget"] = budget
    dp.append(dp_factors)
    problem_names.append(problem_name)
    
    print(dp)
    # solver options
    solver_name = "ASTRODF"
    
    crn = True
    #sample_size = 10
    
    solver_factors = [{'crn_across_solns': crn, 'sample_size': 5}]
    
    solver_names = [solver_name]
    
    #solver_factors = [{}]
    
    
    # call problemssolvers
    experiment = ProblemsSolvers(problem_factors = dp,
                                  solver_factors = solver_factors,
                                  problem_names = problem_names,
                                  solver_names = solver_names)
    
    
    # run experiment
    n_macro = 10
    n_post = 100
    n_postnormal = 200
    
    experiment.run(n_macroreps = n_macro)
    experiment.post_replicate( n_postreps = n_post, crn_across_macroreps = True)
    experiment.post_normalize(n_postreps_init_opt = n_postnormal)
    experiment.record_group_experiment_results()
    experiment.log_group_experiment_results()
    experiment.report_group_statistics()
    
if (__name__ == "__main__"):
    main()
   
    



