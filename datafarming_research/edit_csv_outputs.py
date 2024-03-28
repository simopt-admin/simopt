# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:00:53 2024

@author: Owner
"""

import pandas as pd

results_filename = './gamma_1_4/gamma_1_4_results.csv' # location of csv file that you want to edit
problem_design_filename =  'expt1_test_inputs.txt' # location of file with  problem design points
new_save_name = 'gamma_1_4_edited_results.csv' # save name of new edited file
design = pd.read_csv(problem_design_filename, sep = ' ')
results = pd.read_csv(results_filename)
n_dp = len(design)

# create new columns in results
results['problem_dp'] = 0

# seperate design by factor
c_m = design.iloc[:, :4]
c_r = design.iloc[:,4:8]
p_v = design.iloc[:,8:12]


for dp in range(n_dp):
    m_cost = str([c_m.iloc[dp,0], c_m.iloc[dp,1], c_m.iloc[dp,2], c_m.iloc[dp,3]])
    r_cost = str([c_r.iloc[dp,0], c_r.iloc[dp,1], c_r.iloc[dp,2], c_r.iloc[dp,3]])
    s_price = str([p_v.iloc[dp,0], p_v.iloc[dp,1], p_v.iloc[dp,2], p_v.iloc[dp,3]])
    
    for index, row in results.iterrows():
        current_c_m = str(row['material_cost'])
        current_c_r = str(row['recourse_cost'])
        current_p_v = str(row['salvage_price'])

        
        if m_cost == current_c_m and r_cost == current_c_r and s_price == current_p_v:

            results.at[index, 'problem_dp'] = dp 

results.to_csv(new_save_name, index = False)
            
print(results)
            
        
    
    