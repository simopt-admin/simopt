# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:36:15 2026

@author: nikki
"""
import sys
from pathlib import Path

# Take the current directory, find the parent, and add it to the system path
sys.path.append(str(Path.cwd().parent))


# Specify the names of the solver(s) and problem(s) to test.
solver_abbr_names = [ "SQPASTRODF"]
problem_abbr_names = ["AMUSEMENTPARK-1"]
solver_factors = [{"easy_solve": False, "use_gradients":False}]
problem_factors = [{}]

num_macroreps = 1
num_postreps = 50
num_postreps_init_opt = 50

# Initialize an instance of the experiment class.
from simopt.experiment_base import ProblemsSolvers

mymetaexperiment = ProblemsSolvers(
    solver_names=solver_abbr_names, problem_names=problem_abbr_names, solver_factors = solver_factors, problem_factors = problem_factors
)

# Write to log file.
mymetaexperiment.log_group_experiment_results()

# Run a fixed number of macroreplications of each solver on each problem.
mymetaexperiment.run(n_macroreps=num_macroreps)