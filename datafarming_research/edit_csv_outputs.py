# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:00:53 2024

@author: Owner
"""

import pandas as pd

results_filename = './post_and_norm_same/ASTRODF_on_CNTNEWS-1_results.csv' # location of csv file that you want to edit
problem_design_filename =  'expt1_test_inputs.txt' # location of file with  problem design points
solver_design_filename = 'ASTRODF_design.txt' #location of file with solver design points
new_save_name = 'same_postnorm_cnt_design_results_edited.csv' # save name of new edited file
problem_design = pd.read_csv(problem_design_filename, sep = ' ')
solver_design = pd.read_csv(solver_design_filename, sep = '\t', header=None)
results = pd.read_csv(results_filename)
prob_n_dp = len(problem_design)
solver_n_dp = len(solver_design)
print(solver_n_dp)

# create new columns in results
results['problem_dp'] = 0
results['solver_dp'] = 0


# seperate problem_design by factor
c_m = problem_design.iloc[:, :4]
c_r = problem_design.iloc[:,4:8]
p_v = problem_design.iloc[:,8:12]

# add col for problem dp
for dp in range(prob_n_dp):
    # problem design factors
    m_cost = str([c_m.iloc[dp,0], c_m.iloc[dp,1], c_m.iloc[dp,2], c_m.iloc[dp,3]])
    r_cost = str([c_r.iloc[dp,0], c_r.iloc[dp,1], c_r.iloc[dp,2], c_r.iloc[dp,3]])
    s_price = str([p_v.iloc[dp,0], p_v.iloc[dp,1], p_v.iloc[dp,2], p_v.iloc[dp,3]])

    
    for index, row in results.iterrows():
        current_c_m = str(row['material_cost'])
        current_c_r = str(row['recourse_cost'])
        current_p_v = str(row['salvage_price'])


        
        if m_cost == current_c_m and r_cost == current_c_r and s_price == current_p_v:

            results.at[index, 'problem_dp'] = dp 



# add col for solver dp
for dp in range(solver_n_dp):
    # solver desing factors
    gamma_1 = str(solver_design.iloc[dp,0])
    gamma_2 = str(solver_design.iloc[dp,1])
    eta_1 = str(solver_design.iloc[dp,2])
    eta_2 = str(solver_design.iloc[dp,3])
    for index, row in results.iterrows():
        current_g_1 = str(row['gamma_1'])
        current_g_2 = str(row['gamma_2'])
        current_e_1 = str(row['eta_1'])
        current_e_2 = str(row['eta_2'])
        
        if gamma_1 == current_g_1 and gamma_2 == current_g_2 and eta_1 == current_e_1 and eta_2 == current_e_2:

            results.at[index, 'solver_dp'] = dp
    
    

results.to_csv(new_save_name, index = False)
            
print(results)
            
        
    
    