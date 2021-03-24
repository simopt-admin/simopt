#!/usr/bin/env python
"""
Summary
-------
Provide dictionary directories listing solvers, problems, and oracles.

Listing
-------
solver_directory : dictionary
problem_directory : dictionary
oracle_directory : dictionary
"""

# import solvers
from solvers.randomsearch import RandomSearch

# import problems
from problems.cntnv_max_profit import CntNVMaxProfit
from problems.mm1_min_mean_sojourn_time import MM1MinMeanSojournTime

# import oracles
from oracles.cntnv import CntNV
from oracles.mm1queue import MM1Queue

solver_directory = {
    "RNDSRCH": RandomSearch
}

problem_directory = {
    "CNTNEWS-1": CntNVMaxProfit,
    "MM1-1": MM1MinMeanSojournTime
}

oracle_directory = {
    "CNTNEWS": CntNV,
    "MM1": MM1Queue
}