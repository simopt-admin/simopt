#!/usr/bin/env python
"""
Summary
-------
Provide dictionary directories listing solvers, problems, and models.
"""
# import solvers
from .solvers.FW_B import BoomFrankWolfe
from .solvers.FW_I import BoomFrankWolfe1
from .solvers.FW_Z import BoomFrankWolfe2
from .solvers.FW_SS import FrankWolfeSS  #Boom_FrankWolfe3
from .solvers.PGD_B import BoomProxGD
from .solvers.PGD_I import BoomProxGD1
from .solvers.PGD_Z import BoomProxGD2
from .solvers.PGD_SS import BoomProxGD3 #pgdss
from .solvers.AS_B import ACTIVESET
from .solvers.AS_I import ACTIVESET1
from .solvers.AS_Z import ACTIVESET2
from .solvers.AS_SS import ACTIVESET3
# import models and problems
from .models.san_2 import SAN, SANLongestPathConstr
from .models.network import Network, NetworkMinTotalCost
from .models.smf_2 import SMF, SMF_Max
from .models.openjackson import OpenJackson, OpenJacksonMinQueue   #6
from .models.cascade_2 import Cascade, CascadeMax
from .models.smfcvx_2 import SMFCVX, SMFCVX_Max


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
    "FW-SS": FrankWolfeSS
}

solver_unabbreviated_directory = {
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
    "FW-SS (SBCN)": FrankWolfeSS
}

problem_directory = {
    "SAN-2": SANLongestPathConstr,
    "NW-1": NetworkMinTotalCost,
    "SMF-1": SMF_Max,
    "OPENJ-1": OpenJacksonMinQueue,
    "CC-1": CascadeMax,
    "SMFCVX-1": SMFCVX_Max
}

problem_unabbreviated_directory = {
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": SANLongestPathConstr,
    "Min Total Cost for Communication Networks System (SDCN)": NetworkMinTotalCost,
    "Min Total Max Flow for SMF()": SMF_Max,
    "Min Total Queue": OpenJacksonMinQueue,
    "Min": CascadeMax,
    "Max Flow Problem Convex": SMFCVX
}
model_directory = {
    "SAN": SAN,
    "NETWORK": Network,
    "SMF": SMF,
    "OPENJACKSON": OpenJackson,
    "CASCADE": Cascade,
    "SMFCVX": SMFCVX,
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
