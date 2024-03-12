#!/usr/bin/env python
"""
Summary
-------
Provide dictionary directories listing solvers, problems, and models.
"""
# import solvers
from .solvers.randomsearch import RandomSearch
from .solvers.Boom_FrankWolfe import BoomFrankWolfe
from .solvers.Boom_ProxGD import BoomProxGD
from .solvers.active_set import ACTIVESET
from .solvers.pgdss import PGDSS

# import models and problems
from .models.san import SAN, SANLongestPath, SANLongestPathConstr
from .models.network import Network, NetworkMinTotalCost
from .models.smf import SMF, SMF_Max
from .models.openjackson import OpenJackson, OpenJacksonMinQueue   #6
from .models.cascade import Cascade, CascadeMax
from .models.smfcvx import SMFCVX0, SMFCVX_Max0


# directory dictionaries
solver_directory = {
    "ACTIVESET": ACTIVESET,
    "RNDSRCH": RandomSearch,
    "Boom-PGD": BoomProxGD,
    "Boom-FW": BoomFrankWolfe,
    "PGD-SS": PGDSS
}

solver_unabbreviated_directory = {
    "Active Set (SBCN)": ACTIVESET,
    "Random Search (SSMN)": RandomSearch,
    "Boom-PGD (SBCN)": BoomProxGD,
    "Boom-FW (SBCN)": BoomFrankWolfe,
    "PGD-SS (SBCN)": PGDSS
}

problem_directory = {
    "SAN-1": SANLongestPath,
    "SAN-2": SANLongestPathConstr,
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
