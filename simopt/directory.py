#!/usr/bin/env python
"""
Summary
-------
Provide dictionary directories listing solvers, problems, and models.

Listing
-------
solver_directory : dictionary
solver_nonabbreviated_directory : dictionary
problem_directory : dictionary
problem_nonabbreviated_directory : dictionary
model_directory : dictionary
model_unabbreviated_directory : dictionary
"""
# import solvers
from .solvers.astrodf import ASTRODF
from .solvers.randomsearch import RandomSearch
from .solvers.neldmd import NelderMead
from .solvers.strong import STRONG
from .solvers.adam import ADAM
from .solvers.aloe import ALOE
from .solvers.pgd import PGD
from .solvers.active_set import ACTIVESET
from .solvers.pgdss import PGDSS
from .solvers.PGD_usimplex import PGD2
from .solvers.dsearch import DS
from .solvers.frankwolfe import FrankWolfe
# import models and problems
from .models.cntnv import CntNV, CntNVMaxProfit
from .models.mm1queue import MM1Queue, MM1MinMeanSojournTime
from .models.facilitysizing import FacilitySize, FacilitySizingTotalCost, FacilitySizingMaxService
from .models.rmitd import RMITD, RMITDMaxRevenue
from .models.sscont import SSCont, SSContMinCost
from .models.ironore import IronOre, IronOreMaxRev, IronOreMaxRevCnt
from .models.dynamnews import DynamNews, DynamNewsMaxProfit
from .models.dualsourcing import DualSourcing, DualSourcingMinCost
from .models.contam import Contamination, ContaminationTotalCostDisc, ContaminationTotalCostCont
from .models.chessmm import ChessMatchmaking, ChessAvgDifference
from .models.san import SAN, SANLongestPath, SANLongestPathConstr
from .models.hotel import Hotel, HotelRevenue
from .models.tableallocation import TableAllocation, TableAllocationMaxRev
from .models.paramesti import ParameterEstimation, ParamEstiMaxLogLik
from .models.fixedsan import FixedSAN, FixedSANLongestPath
from .models.voting import Voting, MinVotingMaxWaitTime
from .models.prodsys import ProdSys, ProdSysMinLeadTime
from .models.covid_old import COVID, CovidMinInfect
from .models.volunteer import Volunteer, VolunteerDist, VolunteerSurvival
from .models.fake2 import Fake2, FakeProblem2
from .models.smf import SMF, SMF_Max
from .models.cascade import Cascade, CascadeMax
# directory dictionaries
solver_directory = {
    "ASTRODF": ASTRODF,
    "RNDSRCH": RandomSearch,
    "NELDMD": NelderMead,
    "STRONG": STRONG,
    "ADAM": ADAM,
    "ALOE": ALOE,
    "PGD": PGD,
    "PGD-SS": PGDSS,
    "ACTIVESET": ACTIVESET,
    "PGD-usimplex": PGD2,
    "DS": DS,
    "FW": FrankWolfe
}
solver_nonabbreviated_directory = {
    "ASTRODF (SDCN)": ASTRODF,
    "RandomSearch": RandomSearch,
    "Projected Gradient Descent": PGD,
    "Projected Gradient Descent": PGDSS,
    "FrankWolfe": FrankWolfe,
    "Active Set": ACTIVESET,
    "Nelder-Mead": NelderMead,
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
    "SAN-2": SANLongestPathConstr,
    "HOTEL-1": HotelRevenue,
    "TABLEALLOCATION-1": TableAllocationMaxRev,
    "PARAMESTI-1": ParamEstiMaxLogLik,
    "FIXEDSAN-1": FixedSANLongestPath,
    "VOTING-1": MinVotingMaxWaitTime,
    "PRODSYS-1": ProdSysMinLeadTime,
    "COVID-1": CovidMinInfect,
    "VOLUNTEER-1": VolunteerDist,
    "VOLUNTEER-2": VolunteerSurvival,
    "FAKE2-1": FakeProblem2,
    "SMF-1": SMF_Max,
    "CASCADE-1": CascadeMax

}
problem_nonabbreviated_directory = {
    "Max Profit for Continuous Newsvendor (SBCG)": CntNVMaxProfit,
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": MM1MinMeanSojournTime,
    "Min Total Cost for Facility Sizing (SSCG)": FacilitySizingTotalCost,
    "Max Service for Facility Sizing (SDCN)": FacilitySizingMaxService,
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": RMITDMaxRevenue,
    "Min Total Cost for (s, S) Inventory (SBCN)": SSContMinCost,
    "Max Revenue for Iron Ore (SBDN)": IronOreMaxRev,
    "Max Profit for Dynamic Newsvendor (SBDG)": DynamNewsMaxProfit,
    "Min Cost for Dual Sourcing (SBDN)": DualSourcingMinCost,
    "Min Total Cost for Discrete Contamination (SSDN)": ContaminationTotalCostDisc,
    "Min Total Cost for Continuous Contamination (SSCN)": ContaminationTotalCostCont,
    "Min Avg Difference for Chess Matchmaking (SSCN)": ChessAvgDifference,
    "Min Mean Longest Path for Stochastic Activity Network (SBCN)": SANLongestPath,
    "Max Revenue for Hotel Booking (SBDN)": HotelRevenue,
    "Max Revenue for Restaurant Table Allocation (SDDN)": TableAllocationMaxRev,
    "Max Log Likelihood for Gamma Parameter Estimation (SBCG)": ParamEstiMaxLogLik,
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCN)": FixedSANLongestPath,
    "Minimize Maximum Waiting Time for Voting System (SDDN)": MinVotingMaxWaitTime,
    "Minimize Lead Time for Production System (SSDN)": ProdSysMinLeadTime
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
    "FIXEDSAN": FixedSAN,
    "VOTING": Voting,
    "PRODSYS": ProdSys,
    "COVID": COVID,
    "VOLUNTEER": Volunteer,
    "FAKE2": Fake2,
    "CASCADE": Cascade
}
model_unabbreviated_directory = {
    "Max Profit for Continuous Newsvendor (SBCG)": "CNTNEWS",
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": "MM1",
    "Min Total Cost for Facility Sizing (SSCG)": "FACSIZE",
    "Max Service for Facility Sizing (SDCN)": "FACSIZE",
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": "RMITD",
    "Min Total Cost for (s, S) Inventory (SBCN)": "SSCONT",
    "Max Revenue for Iron Ore (SBDN)": "IRONORE",
    "Max Profit for Dynamic Newsvendor (SBDG)": "DYNAMNEWS",
    "Min Cost for Dual Sourcing (SBDN)": "DUALSOURCING",
    "Min Total Cost for Discrete Contamination (SSDN)": "CONTAM",
    "Min Total Cost for Continuous Contamination (SSCN)": "CONTAM",
    "Min Avg Difference for Chess Matchmaking (SSCN)": "CHESS",
    "Min Mean Longest Path for Stochastic Activity Network (SBCN)": "SAN",
    "Max Revenue for Hotel Booking (SBDN)": "HOTEL",
    "Max Revenue for Restaurant Table Allocation (SDDN)": "TABLEALLOCATION",
    "Max Log Likelihood for Gamma Parameter Estimation (SBCG)": "PARAMESTI",
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCN)": "FIXEDSAN",
    "Minimize Maximum Waiting Time for Voting System (SDDN)": "VOTING",
    "Minimize Lead Time for Production System (SSDN)": "PRODSYS"
}
