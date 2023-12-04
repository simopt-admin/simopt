# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:33:47 2023

@author: Owner
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))


from simopt.data_farming_base import DataFarmingMetaExperiment, DesignPoint
from simopt.experiment_base import read_group_experiment_results


print(read_group_experiment_results(".\experiments\outputs\ASTRODF_on_CNTNEWS-1.pickle"))

