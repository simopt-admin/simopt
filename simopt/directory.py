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
# import models and problems
from .models.example import ExampleModel, ExampleProblem
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
from .models.san import SAN, SANLongestPath
from .models.hotel import Hotel, HotelRevenue
from .models.tableallocation import TableAllocation, TableAllocationMaxRev
from .models.paramesti import ParameterEstimation, ParamEstiMaxLogLik
from .models.fixedsan import FixedSAN, FixedSANLongestPath
from .models.network import Network, NetworkMinTotalCost
from .models.amusementpark import AmusementPark, AmusementParkMinDepart
# directory dictionaries
solver_directory = {
    "ASTRODF": ASTRODF,
    "RNDSRCH": RandomSearch,
    "NELDMD": NelderMead,
    "STRONG": STRONG,
    "SPSA": SPSA,
    "ADAM": ADAM,
    "ALOE": ALOE
}

solver_unabbreviated_directory = {
    "ASTRO-DF (SBCN)": ASTRODF,
    "Random Search (SSMN)": RandomSearch,
    "Nelder-Mead (SBCN)": NelderMead,
    "STRONG (SBCN)": STRONG,
    "SPSA (SBCN)": SPSA,
    "ADAM (SBCN)": ADAM,
    "ALOE (SBCN)": ALOE
}

problem_directory = {
    "EXAMPLE-1": ExampleProblem,
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
    "PARAMESTI-1": ParamEstiMaxLogLik,
    "FIXEDSAN-1": FixedSANLongestPath,
    "NETWORK-1": NetworkMinTotalCost,
    "AMUSEMENTPARK-1": AmusementParkMinDepart
}

problem_unabbreviated_directory = {
    "Min Deterministic Function + Noise (SUCG)": ExampleProblem,
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
    "Max Revenue for Hotel Booking (SBDN)": HotelRevenue,
    "Max Revenue for Restaurant Table Allocation (SDDN)": TableAllocationMaxRev,
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": ParamEstiMaxLogLik,
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": FixedSANLongestPath,
    "Min Total Cost for Communication Networks System (SDCN)": NetworkMinTotalCost,
    "Min Total Departed Visitors for Amusement Park (SDDN)": AmusementParkMinDepart
}
model_directory = {
    "EXAMPLE": ExampleModel,
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
    "AMUSEMENTPARK": AmusementPark
}
model_problem_unabbreviated_directory = {
    "Min Deterministic Function + Noise (SUCG)": "EXAMPLE",
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
    "Min Total Departed Visitors for Amusement Park (SDDN)": "AMUSEMENTPARK"
}
model_problem_class_directory = {
    "Min Deterministic Function + Noise (SUCG)": ExampleModel,
    "Max Profit for Continuous Newsvendor (SBCG)": CntNV,
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": MM1Queue,
    "Min Total Cost for Facility Sizing (SSCG)": FacilitySize,
    "Max Service for Facility Sizing (SDCN)": FacilitySize,
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": RMITD,
    "Min Total Cost for (s, S) Inventory (SBCN)": SSCont,
    "Max Revenue for Iron Ore (SBDN)": IronOre,
    "Max Revenue for Continuous Iron Ore (SBCN)": IronOre,
    "Max Profit for Dynamic Newsvendor (SBDN)": DynamNews,
    "Min Cost for Dual Sourcing (SBDN)": DualSourcing,
    "Min Total Cost for Discrete Contamination (SSDN)": Contamination,
    "Min Total Cost for Continuous Contamination (SSCN)": Contamination,
    "Min Avg Difference for Chess Matchmaking (SSCN)": ChessMatchmaking,
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": SAN,
    "Max Revenue for Hotel Booking (SBDN)": Hotel,
    "Max Revenue for Restaurant Table Allocation (SDDN)": TableAllocation,
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": ParameterEstimation,
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": FixedSAN,
    "Min Total Cost for Communication Networks System (SDCN)": Network,
    "Min Total Departed Visitors for Amusement Park (SDDN)": AmusementPark
}
model_unabbreviated_directory= {
	"Deterministic Function + Noise": ExampleModel,
	"Continuous Newsvendor": CntNV,
	"MM1 Queue": MM1Queue,
	"Facility Sizing": FacilitySize,
	"Revenue Management Temporal Demand": RMITD,
	"(s, S) Inventory": SSCont,
	"Iron Ore": IronOre,
	"Dynamic Newsvendor": DynamNews,
	"Dual Sourcing": DualSourcing,
	"Contamination": Contamination,
	"Chess Matchmaking": ChessMatchmaking,
	"Stochastic Activity Network": SAN,
	"Hotel Booking": Hotel,
	"Restaurant Table Allocation": TableAllocation,
	"Gamma Parameter Estimation": ParameterEstimation,
	"Fixed Stochastic Activity Network": FixedSAN,
	"Communication Networks System": Network,
	"Amusement Park (SDDN)": AmusementPark
}
