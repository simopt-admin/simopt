import unittest
import sys
import os.path as o
import numpy as np
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))


from rng.mrg32k3a import MRG32k3a
from oracles.sscont import SSCont

class TestSSContOracle(unittest.TestCase):
    
    def test_replicate(self):
        myoracle = SSCont()
        rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(myoracle.n_rngs)]
        responses, gradients = myoracle.replicate(rng_list)
        self.assertTrue(responses["avg_order"] >= myoracle.factors["S"] - myoracle.factors["s"])
        self.assertTrue((0 <= responses["order_rate"]) & (responses["order_rate"] <= 1))
        self.assertTrue((0 <= responses["on_time_rate"]) & (responses["on_time_rate"] <= 1))
        self.assertTrue((0 <= responses["stockout_rate"]) & (responses["stockout_rate"] <= 1))
        self.assertTrue(0 <= responses["avg_stockout"])
        self.assertTrue(0 <= responses["avg_backorder_costs"])
        self.assertTrue(0 <= responses["avg_order_costs"])
        self.assertTrue(0 <= responses["avg_holding_costs"])

if __name__ == '__main__':
    unittest.main()