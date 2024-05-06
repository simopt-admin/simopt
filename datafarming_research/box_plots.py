# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:52:50 2024

@author: Owner
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#default_results_filename = './default ASTRODF/default_ASTRODF_on_cnt_design.csv' # file location of csv results for problem design run on default version of ASTRODF
results_filename = 'small_experiment_results.csv' # file location of csv results for cross design of solver design & problem design

exp_data = pd.read_csv(results_filename)

problem_dp = 5

# Generating sample data for four groups
default = exp_data.loc[(exp_data['solver_dp'] == 'default') & (exp_data['problem_dp'] == problem_dp)]

measure = 'Final Relative Optimality Gap'

# Creating subplots
fig, axs = plt.subplots(2, 2, figsize = (25,12))

solvers = [9,14]

for prob_dp in range(0,16):
    fig, ax = plt.subplots()
    ax.boxplot(default[measure], positions=[0], widths=0.6, patch_artist=True, boxprops=dict(facecolor="skyblue"), showfliers=False)
    pos = 1
    for sol_dp in solvers:        
        print(sol_dp)
        # Creating box plots for each dataset
        data = exp_data.loc[(exp_data['solver_dp'] == str(sol_dp)) & (exp_data['problem_dp'] ==prob_dp)]
        ax.boxplot(data[measure], positions=[pos], widths=0.6, patch_artist=True, showfliers=False)
        pos +=1
        
    # Adding labels to x-axis
    ax.set_xticks(np.arange(0,len(solvers)+1))
    name_lst = ['D']
    for sol_dp in solvers:
        name_lst.append(f'{sol_dp}')
    ax.set_xticklabels(name_lst)
    
    # Adding title and labels
    ax.set_title(f'{measure} on problem{prob_dp}')
    ax.set_ylabel(f'{measure}')
    plt.savefig(f'./boxplots/{measure}_problem_{prob_dp}.png')
    
    # Displaying the plot
    plt.show()
        
        
        
    #     # Creating subplots
    #     fig, axs = plt.subplots(2, 2)
    #     data1 = exp_data.loc[(exp_data['solver_dp'] == str(i)) & (exp_data['problem_dp'] ==problem_dp)]
    #     data2 = exp_data.loc[(exp_data['solver_dp'] == str(i+1)) & (exp_data['problem_dp'] ==problem_dp)]
    #     data3 = exp_data.loc[(exp_data['solver_dp'] == str(i+2)) & (exp_data['problem_dp'] ==problem_dp)]
    #     print('data1', data1)
    
    #     # Creating box plots for each group
    #     axs[0, 0].boxplot(default[measure])
    #     axs[0, 0].set_title('Default')
        
    #     axs[0, 1].boxplot(data1[measure])
    #     axs[0, 1].set_title(f'solver_{i}')
        
    #     axs[1, 0].boxplot(data2[measure])
    #     axs[1, 0].set_title(f'solver_{i+1}')
        
    #     axs[1, 1].boxplot(data3[measure])
    #     axs[1, 1].set_title(f'solver_{i+2}')

    # # Adjusting layout
    # plt.tight_layout()
    
    # plt.subplots_adjust(top=.875)
    
    # fig.suptitle(f'{measure} on problem {problem_dp}')
    
    # plt.savefig(f'./boxplots/solver{i}-{i+2}_on_problem_{problem_dp}.png')
    
    # # Displaying the plot
    # plt.show()