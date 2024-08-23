# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:35:10 2024

@author: Owner
"""


import pandas as pd
import ast
import pickle
import numpy as np

file_names = []


file_names.append(f'./experiments/logs/ASTRODF_on_CNTNEWS-1_results.csv')

for sol_dp in range(49):
    for prob_dp in range(49):
        file_names.append(f'./experiments/logs/ASTRODF_{sol_dp}_on_CNTNEWS-1_{prob_dp}_results.csv')


data_frames = []

for file in file_names:
    df = pd.read_csv(file)
    data_frames.append(df)

results = pd.concat(data_frames, axis = 0)

results.to_csv('./4-30/default_is_10_raw_results.csv', index = False)
    
    

