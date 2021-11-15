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
from solvers.astrodf import ASTRODF
from solvers.randomsearch import RandomSearch
from solvers.simannealing import SANE
# import oracles and problems
from oracles.cntnv import CntNV, CntNVMaxProfit
from oracles.mm1queue import MM1Queue, MM1MinMeanSojournTime
from oracles.facilitysizing import FacilitySize, FacilitySizingTotalCost, FacilitySizingMaxService
from oracles.rmitd import RMITD, RMITDMaxRevenue
from oracles.sscont import SSCont, SSContMinCost
from oracles.contam import Contamination, ContaminationTotalCost
# directory dictionaries
solver_directory = {
    "ASTRODF": ASTRODF,
    "RNDSRCH": RandomSearch,
    "SANE": SANE,
}
problem_directory = {
    "CNTNEWS-1": CntNVMaxProfit,
    "MM1-1": MM1MinMeanSojournTime,
    "FACSIZE-1": FacilitySizingTotalCost,
    "FACSIZE-2": FacilitySizingMaxService,
    "RMITD-1": RMITDMaxRevenue,
    "SSCONT-1": SSContMinCost,
    "CONTAM-1": ContaminationTotalCost
}
oracle_directory = {
    "CNTNEWS": CntNV,
    "MM1": MM1Queue,
    "FACSIZE": FacilitySize,
    "RMITD": RMITD,
    "SSCONT": SSCont,
    "CONTAM": Contamination
}
