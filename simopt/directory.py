#!/usr/bin/env python
"""
Summary
-------
Provide dictionary directories listing solvers, problems, and models.
"""
# import solvers
from .solvers.Boom_FrankWolfe import BoomFrankWolfe
from .solvers.Boom_FrankWolfe1 import BoomFrankWolfe1
from .solvers.Boom_FrankWolfe2 import BoomFrankWolfe2
from .solvers.FrankWolfe_SS import FrankWolfeSS
from .solvers.Boom_ProxGD import BoomProxGD
from .solvers.Boom_ProxGD1 import BoomProxGD1
from .solvers.Boom_ProxGD2 import BoomProxGD2
from .solvers.Boom_ProxGD3 import BoomProxGD3 #pgdss
from .solvers.active_set import ACTIVESET
from .solvers.active_set1 import ACTIVESET1
from .solvers.active_set2 import ACTIVESET2
from .solvers.active_set3 import ACTIVESET3
# import models and problems
from .models.san import SAN, SANLongestPath, SANLongestPathConstr
from .models.network import Network, NetworkMinTotalCost
from .models.smf import SMF, SMF_Max
from .models.openjackson import OpenJackson, OpenJacksonMinQueue
from .models.cascade import Cascade, CascadeMax
from .models.smfcvx import SMFCVX0, SMFCVX_Max0

# directory dictionaries
solver_directory = {
    "AS-B": ACTIVESET,
    "AS-I": ACTIVESET1,
    "AS-Z": ACTIVESET2,
    "AS-SS": ACTIVESET3,
    "PGD-B": BoomProxGD,
    "PGD-I": BoomProxGD1,
    "PGD-Z": BoomProxGD2,
    "PGD-SS": BoomProxGD3,
    "FW-B": BoomFrankWolfe,
    "FW-I": BoomFrankWolfe1,
    "FW-Z": BoomFrankWolfe2,
    "FW-SS": FrankWolfeSS,
}

solver_unabbreviated_directory = {
    "Active Set (SBCN)": ACTIVESET,
    "Active Set1 (SBCN)": ACTIVESET1,
    "Active Set2 (SBCN)": ACTIVESET2,
    "Active Set3 (SBCN)": ACTIVESET3,
    "PGD-backtracking (SBCN)": BoomProxGD,
    "PGD-interpolation (SBCN)": BoomProxGD1,
    "PGD-zoom (SBCN)": BoomProxGD2,
    "PGD-SS (SBCN)": BoomProxGD3,
    "FW-backtracking (SBCN)": BoomFrankWolfe,
    "FW-interpolation (SBCN)": BoomFrankWolfe1,
    "FW-zoom (SBCN)": BoomFrankWolfe2,
    "FW-SS (SBCN)": FrankWolfeSS,
}

problem_directory = {
    "SAN-1": SANLongestPath,
    "SAN-2": SANLongestPathConstr,
    "NETWORK-1": NetworkMinTotalCost,
    "SMF-1": SMF_Max,
    "OPENJACKSON-1": OpenJacksonMinQueue,
    "CASCADE-1": CascadeMax,
    "SMFCVX-1": SMFCVX_Max0
}

problem_unabbreviated_directory = {
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": SANLongestPath,
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": SANLongestPathConstr,
    "Min Total Cost for Communication Networks System (SDCN)": NetworkMinTotalCost,
    "Min Total Max Flow for SMF()": SMF_Max,
    "Min Total Queue": OpenJacksonMinQueue,
    "Min": CascadeMax,
    "Max Flow Problem Convex": SMFCVX_Max0
}
model_directory = {
    "SAN": SAN,
    "NETWORK": Network,
    "SMF": SMF,
    "OPENJACKSON": OpenJackson,
    "CASCADE": Cascade,
    "SMFCVX": SMFCVX0,
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
