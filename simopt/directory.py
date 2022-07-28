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
from solvers.strong import STRONG
from solvers.neldmd import NelderMead
from solvers.adam import ADAM
from solvers.aloe import ALOE
from solvers.adam2 import ADAM2
from solvers.PGD_usimplex import PGD
from solvers.aloe2 import ALOE2
# import models and problems
from models.cntnv import CntNV, CntNVMaxProfit
from models.mm1queue import MM1Queue, MM1MinMeanSojournTime
from models.facilitysizing import FacilitySize, FacilitySizingTotalCost, FacilitySizingMaxService
from models.rmitd import RMITD, RMITDMaxRevenue
from models.sscont import SSCont, SSContMinCost
from models.ironore import IronOre, IronOreMaxRev, IronOreMaxRevCnt
from models.dynamnews import DynamNews, DynamNewsMaxProfit
from models.dualsourcing import DualSourcing, DualSourcingMinCost
from models.contam import Contamination, ContaminationTotalCostDisc, ContaminationTotalCostCont
from models.chessmm import ChessMatchmaking, ChessAvgDifference
from models.san import SAN, SANLongestPath
from models.hotel import Hotel, HotelRevenue
from models.tableallocation import TableAllocation, TableAllocationMaxRev
from models.paramesti import ParameterEstimation, ParamEstiMinLogLik
from models.covid import COVID, CovidMinInfect
from models.volunteer import Volunteer, VolunteerDist, VolunteerSurvival
from models.fake import Fake, FakeProblem
# directory dictionaries
solver_directory = {
    "ASTRODF": ASTRODF,
    "RNDSRCH": RandomSearch,
    "STRONG": STRONG,
    "NELDMD": NelderMead,
    "ADAM": ADAM,
    "ALOE": ALOE,
    "ADAM2": ADAM2,
    "PGD": PGD,
    "ALOE2": ALOE2
}
problem_directory = {
    "CNTNEWS-1": CntNVMaxProfit,
    "MM1-1": MM1MinMeanSojournTime,
    "FACSIZE-1": FacilitySizingTotalCost,
    "FACSIZE-2": FacilitySizingMaxService,
    "RMITD-1": RMITDMaxRevenue,
    "SSCONT-1": SSContMinCost,
    "IRONORE-1": IronOreMaxRev,
    "IRONORECONT-1": IronOreMaxRevCnt,
    "DYNAMNEWS-1": DynamNewsMaxProfit,
    "DUALSOURCING-1": DualSourcingMinCost,
    "CONTAM-1": ContaminationTotalCostDisc,
    "CONTAM-2": ContaminationTotalCostCont,
    "CHESS-1": ChessAvgDifference,
    "SAN-1": SANLongestPath,
    "HOTEL-1": HotelRevenue,
    "TABLEALLOCATION-1": TableAllocationMaxRev,
    "PARAMESTI-1": ParamEstiMinLogLik,
    "COVID-1": CovidMinInfect,
    "VOLUNTEER-1": VolunteerDist,
    "VOLUNTEER-2": VolunteerSurvival,
    "FAKE-1": FakeProblem
}
model_directory = {
    "CNTNEWS": CntNV,
    "MM1": MM1Queue,
    "FACSIZE": FacilitySize,
    "RMITD": RMITD,
    "SSCONT": SSCont,
    "IRONORE": IronOre,
    "DYNAMNEWS": DynamNews,
    "DUALSOURCING": DualSourcing,
    "CONTAM": Contamination,
    "CHESS": ChessMatchmaking,
    "SAN": SAN,
    "HOTEL": Hotel,
    "TABLEALLOCATION": TableAllocation,
    "PARAMESTI": ParameterEstimation,
    "COVID": COVID,
    "VOLUNTEER": Volunteer,
    "FAKE": Fake
}