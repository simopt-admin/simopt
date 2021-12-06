#!/usr/bin/env python
"""
Summary
-------
Provide dictionary directories listing solvers, problems, and models.

Listing
-------
solver_directory : dictionary
problem_directory : dictionary
model_directory : dictionary
"""
# import solvers
from solvers.astrodf import ASTRODF
from solvers.randomsearch import RandomSearch
from solvers.simannealing import SANE
from solvers.strong import STRONG
# import models and problems
from models.cntnv import CntNV, CntNVMaxProfit
from models.mm1queue import MM1Queue, MM1MinMeanSojournTime
from models.facilitysizing import FacilitySize, FacilitySizingTotalCost, FacilitySizingMaxService
from models.rmitd import RMITD, RMITDMaxRevenue
from models.sscont import SSCont, SSContMinCost
from models.ironore import IronOre, IronOreMaxRev
from models.dynamnews import DynamNews, DynamNewsMaxProfit
from models.vehicleroute import VehicleRoute, VehicleRouteMinDist
# directory dictionaries
solver_directory = {
    "ASTRODF": ASTRODF,
    "RNDSRCH": RandomSearch,
    "SANE": SANE,
    "STRONG": STRONG,
}
problem_directory = {
    "CNTNEWS-1": CntNVMaxProfit,
    "MM1-1": MM1MinMeanSojournTime,
    "FACSIZE-1": FacilitySizingTotalCost,
    "FACSIZE-2": FacilitySizingMaxService,
    "RMITD-1": RMITDMaxRevenue,
    "SSCONT-1": SSContMinCost,
    "IRONORE-1": IronOreMaxRev,
    "DYNAMNEWS-1": DynamNewsMaxProfit,
    "VEHROUTE-1": VehicleRouteMinDist
}
model_directory = {
    "CNTNEWS": CntNV,
    "MM1": MM1Queue,
    "FACSIZE": FacilitySize,
    "RMITD": RMITD,
    "SSCONT": SSCont,
    "IRONORE": IronOre,
    "DYNAMNEWS": DynamNews,
    "VEHROUTE": VehicleRoute
}
