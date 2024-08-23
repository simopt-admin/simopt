# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:00:53 2024

@author: Owner
"""

import pandas as pd
import ast

results_filename = './4-25_correct_gaps/ASTRODF_on_CNTNEWS-1_results.csv' # location of csv file that you want to edit
problem_design_filename =  'cnt_design.xlsx' # location of file with  problem design points
solver_design_filename = 'ASTRODF_design.txt' #location of file with solver design points
new_save_name = '5-6_subset_results.csv' # save name of new edited file
problem_design = pd.read_excel(problem_design_filename)
solver_design = pd.read_csv(solver_design_filename, sep = '\t', header=None)
results = pd.read_csv(results_filename)
prob_n_dp = len(problem_design)
solver_n_dp = len(solver_design)


# create new columns in results
results['problem_dp'] = 0
results['solver_dp'] = 'default'

def round_values(lst):
    new_list = []
    for item in lst:
        new_item = round(float(item),3)
        new_list.append(new_item)
    return new_list


# add col for problem dp
for prob_dp, row in problem_design.iterrows():
    # problem design factors
    m_cost = round_values([row['cm1'], row['cm2'],row['cm3'], row['cm4']])
    r_cost = round_values([row['rm1'], row['rm2'],row['rm3'], row['rm4']])
    s_price = round_values([row['sm1'], row['sm2'],row['sm3'], row['sm4']])
    

    
    for index, row in results.iterrows():
        current_c_m = round_values(ast.literal_eval(row['material_cost']))
        current_c_r = round_values(ast.literal_eval(row['recourse_cost']))
        current_p_v = round_values(ast.literal_eval(row['salvage_price']))
        if prob_dp == 1 and index ==11:
            print(m_cost, current_c_m)
        




        
        if m_cost == current_c_m and r_cost == current_c_r and s_price == current_p_v:
            
            print('true')

            results.at[index, 'problem_dp'] = prob_dp



# add col for solver dp
for dp in range(solver_n_dp):
    # solver desing factors
    gamma_1 = str(solver_design.iloc[dp,2])
    gamma_2 = str(solver_design.iloc[dp,3])
    eta_1 = str(solver_design.iloc[dp,0])
    eta_2 = str(solver_design.iloc[dp,1])
    for index, row in results.iterrows():
        current_g_1 = str(row['gamma_1'])
        current_g_2 = str(row['gamma_2'])
        current_e_1 = str(row['eta_1'])
        current_e_2 = str(row['eta_2'])
        
        if gamma_1 == current_g_1 and gamma_2 == current_g_2 and eta_1 == current_e_1 and eta_2 == current_e_2:

            results.at[index, 'solver_dp'] = dp

    
    

results.to_csv(new_save_name, index = False)
            
print(results)
            
        
    
    