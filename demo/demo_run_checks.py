"""
This script is intended to help with running a data-farming experiment on
a solver. It creates a design of solver factors and runs multiple
macroreplications at each version of the solver. Outputs are printed to a file.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))


from simopt.data_farming_base import DataFarmingMetaExperiment, DesignPoint
from simopt.experiment_base import create_design, ProblemsSolvers

from simopt.models.cntnv import CntNVMaxProfit


# run checks on model

problem = CntNVMaxProfit()

factor_names = ["initial_solution", "budget"]
problem.run_all_checks(factor_names)
