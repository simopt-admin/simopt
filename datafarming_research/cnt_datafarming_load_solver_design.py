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
    design_filename = "ASTRODF_factor_settings_design.txt" # location of design file
    df = pd.read_csv(design_filename, sep='\t', encoding="utf-8") # may need to change sep depending on type of file uploaded
    
        
    n_dp = len(df) # number of design points(
    solver_factors = [] # list to hold dictionary of problem factors for each dp of problem (don't change)
    solver_names = [] # list to hold problem names (don't change)
    
    # add default version of ASTRODF to factor list
    solver_factors.append({})
    solver_names.append(solver_name)
    
    # concatinate tables back into arrays for each design point
    for row in range(n_dp):
        dp_factors = {}
        dp_factors["gamma_1"] = float(df.iloc[row, 0])
        dp_factors["gamma_2"] = float(df.iloc[row, 1])
        dp_factors["eta_1"] = float(df.iloc[row, 2])
        dp_factors["eta_2"] = float(df.iloc[row, 3])
    
        solver_factors.append(dp_factors)
        solver_names.append(solver_name)
    print('factors', solver_factors)
    # solver information
    problem_name = "CNTNEWS-1" # name of solver

    #fixed factors (uncomment any of these if you want to change from problem default)
    # n_p = 3 # number of products
    # n_m = 4 # number of materials
    # mat_to_prod = [[1, 2, 1, 3], [1, 1, 3, 1],[2, 0, 4, 1]] # maps materials to products (currently every product uses all materials)
    # process_cost = [0.1, 0.1, 0.1,0.1] # processing cost per product unit
    # order_cost = 20 # one time ordering cost
    # purchase_yeild = [.9,.9,.9,.9] # yeild rates for initially purchased materials
    #total_budget = 5000 # budget for all purchases
    #sales_price = [20,20,20,20] # sales price per product unit
    # order_quantity = [20,20,20,20] # intial order quantity per material
    # mean = [15,15,15,15] # mean parameter for poisson demand distribution
    
    # problem factors (can change these as desired, comment out to use problem defaults)
    init_sol = [20, 20, 20, 20]
    budget = 1000
     
    # turn design txt file into dataframe
    design_filename = "expt1_test_inputs.txt" # location of design file
    df = pd.read_csv(design_filename, sep=' ', encoding="utf-8") # may need to change sep depending on type of file uploaded
    
    # get tables for each design factor depending on rows in design file
    # will need to add a new table if changing more than 3 problem factors
    # if more than 4 material types will need to adjust column numbers to match design file
    c_m = df.iloc[:, :4]
    c_r = df.iloc[:,4:8]
    p_v = df.iloc[:,8:12]
    
    # these are just used as validation, can remove if desired
    print('c_m', c_m)
    print('c_r', c_r)
    print('p_v', p_v)
        
    n_dp = len(df) # number of design points(
    problem_factors = [] # list to hold dictionary of problem factors for each dp of problem (don't change)
    problem_names = [] # list to hold problem names (don't change)
    
    # concatinate tables back into arrays for each design point
    for row in range(n_dp):
        # get problem factor arrays for each design point
        # will need to add an additional index to each array if increaseing the number of materials
        # will need to create an additional array if varying more than 3 problem factors
        m_cost = [c_m.iloc[row,0], c_m.iloc[row,1], c_m.iloc[row,2], c_m.iloc[row,3]]
        r_cost = [c_r.iloc[row,0], c_r.iloc[row,1], c_r.iloc[row,2], c_r.iloc[row,3]]
        s_price = [p_v.iloc[row,0], p_v.iloc[row,1], p_v.iloc[row,2], p_v.iloc[row,3]]
    
        
        # create dictionary of all factor values at this dp
        # uncomment correspoding fixed factor lines to change from default problem value
        dp_factors = {}
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
        dp_factors["initial_solution"] = init_sol
        dp_factors["budget"] = budget
    
        problem_factors.append(dp_factors)
        problem_names.append(problem_name)

    # # example of running solver using only default values
    # solver_factors = [{}]
    # solver_names = [solver_name]
    
    # # example of running a solver cross design over sample size values of 5, 10 and 50
    # solver_factors = [{'crn_across_solns': crn, 'sample_size': 5}, {'crn_across_solns': crn, 'sample_size': 10}, {'crn_across_solns': crn, 'sample_size': 50}] #include a dictionary for each version of solver
    # solver_names = [solver_name, solver_name, solver_name] # repeat solver name for every version of solver
    
    # call problemssolvers
    experiment = ProblemsSolvers(problem_factors = problem_factors,
                                  solver_factors = solver_factors,
                                  problem_names = problem_names,
                                  solver_names = solver_names)
    
    
    # run experiment (can change any of these values as desired)
    n_macro = 10 # number of macro replications at each design point
    n_post = 100 # number of post replications of each macro rep
    n_postnormal = 200 # number of post replications at x0 and x*
    
    experiment.run(n_macroreps = n_macro)
    experiment.post_replicate( n_postreps = n_post)
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


