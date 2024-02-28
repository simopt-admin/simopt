#!/usr/bin/env python
"""
Summary
-------
Provide dictionary directories listing solvers, problems, and models.
"""
# import solvers
from .solvers.astrodf import ASTRODF
from .solvers.randomsearch import RandomSearch
from .solvers.neldmd import NelderMead
from .solvers.strong import STRONG
from .solvers.spsa import SPSA
from .solvers.adam import ADAM
from .solvers.aloe import ALOE
from .solvers.gasso import GASSO
from .solvers.Boom_FrankWolfe import BoomFrankWolfe
from .solvers.Boom_ProxGD import BoomProxGD
from .solvers.active_set import ACTIVESET
from .solvers.dsearch import DS
from .solvers.pgdss import PGDSS
# import models and problems
from .models.cntnv import CntNV, CntNVMaxProfit
from .models.mm1queue import MM1Queue, MM1MinMeanSojournTime
from .models.facilitysizing import FacilitySize, FacilitySizingTotalCost, FacilitySizingMaxService
from .models.rmitd0 import RMITD, RMITDMaxRevenue
from .models.sscont import SSCont, SSContMinCost
from .models.ironore import IronOre, IronOreMaxRev, IronOreMaxRevCnt
from .models.dynamnews import DynamNews, DynamNewsMaxProfit
from .models.dualsourcing import DualSourcing, DualSourcingMinCost
from .models.contam import Contamination, ContaminationTotalCostDisc, ContaminationTotalCostCont
from .models.chessmm import ChessMatchmaking, ChessAvgDifference
# from .models.san_2 import SAN, SANLongestPath
# from .models.san_1 import SAN1, SANLongestPath1
from .models.san import SAN, SANLongestPath, SANLongestPathConstr
from .models.hotel import Hotel, HotelRevenue
from .models.tableallocation import TableAllocation, TableAllocationMaxRev
from .models.paramesti import ParameterEstimation, ParamEstiMaxLogLik
from .models.fixedsan import FixedSAN, FixedSANLongestPath
from .models.network import Network, NetworkMinTotalCost
from.models.fickleserver import FickleServer, FickleServerMinServiceRate
from .models.EOQ import EOQ, EOQ_Mincost
# from .models.vac import COVID_vac, CovidMinInfectVac  #orginal method
# from .models.vac_alias import COVID_vac, CovidMinInfectVac  #by multinomial 
from .models.covid import COVID_vac, CovidMinInfectVac  #combine case
from .models.ccbaby_old import BabyCC, CCBabyShift
from .models.smf import SMF, SMF_Max
from .models.openjackson import OpenJackson, OpenJacksonMinQueue   #6
from .models.cascade import Cascade, CascadeMax
from .models.testcase2 import QuadraticF, QuadraticF_Min
from .models.volunteer import Volunteer, VolunteerDist, VolunteerSurvival
from .models.smfcvx import SMFCVX0, SMFCVX_Max0
from .models.smf_cvx import CVXSMF, CVXSMF_Max
# from .models.smfcvx import SMFCVX, SMFCVX_Max


# directory dictionaries
solver_directory = {
    "ASTRODF": ASTRODF,
    "ACTIVESET": ACTIVESET,
    "RNDSRCH": RandomSearch,
    "NELDMD": NelderMead,
    "STRONG": STRONG,
    "SPSA": SPSA,
    "ADAM": ADAM,
    "ALOE": ALOE,
    "GASSO": GASSO,
    "Boom-PGD": BoomProxGD,
    "Boom-FW": BoomFrankWolfe,
    "DS": DS,
    "PGD-SS": PGDSS
}

solver_unabbreviated_directory = {
    "ASTRO-DF (SBCN)": ASTRODF,
    "Active Set (SBCN)": ACTIVESET,
    "Random Search (SSMN)": RandomSearch,
    "Nelder-Mead (SBCN)": NelderMead,
    "STRONG (SBCN)": STRONG,
    "SPSA (SBCN)": SPSA,
    "ADAM (SBCN)": ADAM,
    "Adaptive Line-search with Oracle Estimations (SBCN)": ALOE,
    "Boom-PGD (SBCN)": BoomProxGD,
    "Boom-FW (SBCN)": BoomFrankWolfe,
    "DS (SBCN)": DS,
    "PGD-SS (SBCN)": PGDSS
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
    "NETWORK-1": NetworkMinTotalCost,
    "FICKLE-1": FickleServerMinServiceRate,
    "EOQ-1": EOQ_Mincost,
    "VAC-1": CovidMinInfectVac,
    "CCBABY-1": CCBabyShift,
    "SMF-1": SMF_Max,
    "OPENJACKSON-1": OpenJacksonMinQueue,
    "CASCADE-1": CascadeMax,
    "QF-1": QuadraticF_Min,
    "VOLUNTEER-1": VolunteerDist,
    "VOLUNTEER-2": VolunteerSurvival,
    "CVXSMF-1": CVXSMF_Max,
    "SMFCVX-1": SMFCVX_Max0
}

problem_unabbreviated_directory = {
    "Max Profit for Continuous Newsvendor (SBCG)": CntNVMaxProfit,
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": MM1MinMeanSojournTime,
    "Min Total Cost for Facility Sizing (SSCG)": FacilitySizingTotalCost,
    "Max Service for Facility Sizing (SDCN)": FacilitySizingMaxService,
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": RMITDMaxRevenue,
    "Min Total Cost for (s, S) Inventory (SBCN)": SSContMinCost,
    "Max Revenue for Iron Ore (SBDN)": IronOreMaxRev,
    "Max Revenue for Continuous Iron Ore (SBCN)": IronOreMaxRevCnt,
    "Max Profit for Dynamic Newsvendor (SBDN)": DynamNewsMaxProfit,
    "Min Cost for Dual Sourcing (SBDN)": DualSourcingMinCost,
    "Min Total Cost for Discrete Contamination (SSDN)": ContaminationTotalCostDisc,
    "Min Total Cost for Continuous Contamination (SSCN)": ContaminationTotalCostCont,
    "Min Avg Difference for Chess Matchmaking (SSCN)": ChessAvgDifference,
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": SANLongestPath,
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": SANLongestPathConstr,
    "Max Revenue for Hotel Booking (SBDN)": HotelRevenue,
    "Max Revenue for Restaurant Table Allocation (SDDN)": TableAllocationMaxRev,
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": ParamEstiMaxLogLik,
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": FixedSANLongestPath,
    "Min Total Cost for Communication Networks System (SDCN)": NetworkMinTotalCost,
    "MIN Total cost...": FickleServerMinServiceRate,
    "Min Total Symptomatic Infected People()": CovidMinInfectVac,
    "Min Total Cost for Baby Call Center With Shift()": CCBabyShift,
    "Min Total Max Flow for SMF()": SMF_Max,
    "Min Total Queue": OpenJacksonMinQueue,
    "Min Total Cost for Volunteer": VolunteerDist,
    "Max Total Survival rate for Volunteer": VolunteerSurvival,
    "Min": CascadeMax,
    "Max Flow Problem Convex": CVXSMF_Max,
    "Max Flow Problem Convex": SMFCVX_Max0
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
    "NETWORK": Network,
    "FICKLE": FickleServer,
    "COVIDVAC": COVID_vac,
    "CCBABY": BabyCC,
    "SMF": SMF,
    "OPENJACKSON": OpenJackson,
    "VOLUNTEER": Volunteer,
    "CASCADE": Cascade,
    "CVXSMF": CVXSMF,
    "SMFCVX": SMFCVX0,
    "QF": QuadraticF
}
model_unabbreviated_directory = {
    "Max Profit for Continuous Newsvendor (SBCG)": "CNTNEWS",
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": "MM1",
    "Min Total Cost for Facility Sizing (SSCG)": "FACSIZE",
    "Max Service for Facility Sizing (SDCN)": "FACSIZE",
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": "RMITD",
    "Min Total Cost for (s, S) Inventory (SBCN)": "SSCONT",
    "Max Revenue for Iron Ore (SBDN)": "IRONORE",
    "Max Revenue for Continuous Iron Ore (SBCN)": "IRONORE",
    "Max Profit for Dynamic Newsvendor (SBDN)": "DYNAMNEWS",
    "Min Cost for Dual Sourcing (SBDN)": "DUALSOURCING",
    "Min Total Cost for Discrete Contamination (SSDN)": "CONTAM",
    "Min Total Cost for Continuous Contamination (SSCN)": "CONTAM",
    "Min Avg Difference for Chess Matchmaking (SSCN)": "CHESS",
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": "SAN",
    "Max Revenue for Hotel Booking (SBDN)": "HOTEL",
    "Max Revenue for Restaurant Table Allocation (SDDN)": "TABLEALLOCATION",
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": "PARAMESTI",
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": "FIXEDSAN",
    "Min Total Cost for Communication Networks System (SDCN)": "NETWORK",
    "Min Total Symptomatic Infected People()": "COVIDVAC",
    "Min Total Cost for Baby Call Center with Shift()": "CCBabyShift",
    "Min Total Max Flow for SMF()": "SMF_Max",
    "Min Total Queue()": "OPENJ",
    "Min Total distance()": "VOLUNTEER",
    "Max Total Flow()": "CVXSMF",
    "Max Total Flow()": "SMFCVX"
}
