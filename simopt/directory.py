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
from solvers.simannealing import SANE

# import problems
from problems.cntnv_max_profit import CntNVMaxProfit
from problems.mm1_min_mean_sojourn_time import MM1MinMeanSojournTime
from problems.facilitysizing_totalcost import FacilitySizingTotalCost
from problems.rmitd_maxrevenue import RMITDMaxRevenue
from problems.sscont_min_cost import SSContMinCost

# import oracles
from oracles.cntnv import CntNV
from oracles.mm1queue import MM1Queue
from oracles.facilitysizing import FacilitySize
from oracles.rmitd import RMITD
from oracles.sscont import SSCont

solver_directory = {
    "RNDSRCH": RandomSearch,
    "SANE": SANE,
}

problem_directory = {
    "CNTNEWS-1": CntNVMaxProfit,
    "MM1-1": MM1MinMeanSojournTime,
    "FACSIZE-1": FacilitySizingTotalCost,
    "RMITD-1": RMITDMaxRevenue,
    "SSCONT-1": SSContMinCost
}

oracle_directory = {
    "CNTNEWS": CntNV,
    "MM1": MM1Queue,
    "FACSIZE": FacilitySize,
    "RMITD": RMITD,
    "SSCONT": SSCont
}