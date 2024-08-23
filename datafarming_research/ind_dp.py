"""
This script is intended to load a design over problem factors to be run on
the updated version of CNTNEWS-1.
"""
import os
import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
import pandas as pd
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles

def main():
    
    solver_name = 'ASTRODF' # name of solver
    
    # turn design txt file into dataframe
    design_filename = "ASTRODF_design.txt" # location of design file
    df = pd.read_csv(design_filename, sep='\t', encoding="utf-8", header=None) # may need to change sep depending on type of file uploaded
    print('solver df', df)
        
    n_dp = len(df) # number of design points(
    solver_factors = [] # list to hold dictionary of problem factors for each dp of problem (don't change)
    solver_names = [] # list to hold problem names (don't change)
    
    # add default version of ASTRODF to factor list
    solver_factors.append({'lambda_min' : 10})
    solver_names.append(solver_name)
    
    solver_renames = []
    solver_renames.append("ASTRODF_default")
    # concatinate tables back into arrays for each design point
    for row in range(n_dp):
        print(row)
        dp_factors = {}
        dp_factors["eta_1"] = float(df.iloc[row, 0])
        dp_factors["eta_2"] = float(df.iloc[row, 1])
        dp_factors["gamma_1"] = float(df.iloc[row, 2])
        dp_factors["gamma_2"] = float(df.iloc[row, 3])
        # temp
        dp_factors["lambda_min"] = 10
    
        solver_factors.append(dp_factors)
        solver_names.append(solver_name)
        solver_renames.append(f"{solver_name}_{row}")
        print(solver_factors)
    print(solver_renames)
    print('factors', solver_factors)
    # solver information
    problem_name = "CNTNEWS-1" # name of solver

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
    

        
    n_dp = len(df) # number of design points(
    problem_factors = [] # list to hold dictionary of problem factors for each dp of problem (don't change)
    problem_names = [] # list to hold problem names (don't change)
    
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
        dp_factors["initial_solution"] = init_sol
        dp_factors["budget"] = budget
    
        problem_factors.append(dp_factors)
        problem_names.append(problem_name)
        problem_renames.append(f"{problem_name}_{index}")
        
    print(solver_factors[12])

    # call problemssolvers
    experiment = ProblemsSolvers(problem_factors = [problem_factors[0]],
                                  solver_factors = [solver_factors[0]],
                                  problem_names = [problem_names[0]],
                                  solver_names = [solver_names[0]],
                                  problem_renames = [problem_renames[0]],
                                  solver_renames = [solver_renames[0]])
    
    
    # run experiment (can change any of these values as desired)
    n_macro = 10 # number of macro replications at each design point
    n_post = 200 # number of post replications of each macro rep
    n_postnormal = 200 # number of post replications at x0 and x*
    
    experiment.run(n_macroreps = n_macro)
    experiment.post_replicate( n_postreps = n_post, crn_across_macroreps =True)
    experiment.post_normalize(n_postreps_init_opt = n_postnormal)
    experiment.record_group_experiment_results()
    experiment.log_group_experiment_results()
    experiment.report_group_statistics()
    
    # print("Plotting results.")
    # # Produce basic plots of the solvers on the problems.
    # plot_solvability_profiles(experiments=experiment.experiments, plot_type="cdf_solvability")
    # # Plots will be saved in the folder experiments/plots.
    # print("Finished. Plots can be found in experiments/plots folder.")
    

if (__name__ == "__main__"):
    main()


