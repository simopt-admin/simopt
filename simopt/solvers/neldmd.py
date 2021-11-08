"""
Summary
-------
Nelder-Mead
The algorithm maintains a simplex of points that moves around the feasible 
region according to certain geometric operations: reflection, expansion, 
scontraction, and shrinking.
"""
from base import Solver
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

class NELDMD(Solver):
    """
    REFERENCE		
    Russell R. Barton, John S. Ivey, Jr., (1996)
    Nelder-Mead Simplex Modifications for Simulation 
    Optimization. Management Science 42(7):954-973.

    Attributes
    ----------
    name : string
        name of solver
    objective_type : string
        description of objective types:
            "single" or "multi"
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_needed : bool
        indicates if gradient of objective function is needed
    factors : dict
        changeable factors (i.e., parameters) of the solver
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    rng_list : list of rng.MRG32k3a objects
        list of RNGs used for the solver's internal purposes

    Arguments
    ---------
    name : str
        user-specified name for solver
    fixed_factors : dict
        fixed_factors of the solver

    See also
    --------
    base.Solver
    """
    def __init__(self, name="NELDMD", fixed_factors={}):
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "Use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "r": {
                "description": "number of replications taken at each solution",
                "datatype": float,
                "default": 30.
            },
            "alpha": {
                "description": "reflection coefficient > 0",
                "datatype": float,
                "default": 1.
            },
            "gammap": {
                "description": "expansion coefficient > 1",
                "datatype": float,
                "default": 2.
            },
            "betap": {
                "description": "contraction coefficient > 0, < 1",
                "datatype": float,
                "default": 0.5
            },
            "delta": {
                "description": "shrink factor > 0, < 1",
                "datatype": float,
                "default": 0.5
            },
            "sensitivity": {
                "description": "shrinking scale for VarBds (bounds)",
                "datatype": float,
                "default": 10**(-7)
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "alpha": self.check_alpha,
            "gammap": self.check_gammap,
            "betap": self.check_betap,
            "delta": self.check_delta,
            "sensitivity": self.check_sensitivity
        }
        super().__init__(fixed_factors)

    def check_r(self):
        return self.factors["r"] > 0

    def check_alpha(self):
        return self.factors["alpha"] > 0

    def check_gammap(self):
        return self.factors["gammap"] > 1

    def check_betap(self):
        return (self.factors["betap"] > 0) & (self.factors["betap"] < 1)

    def check_delta(self):
        return (self.factors["delta"] > 0) & (self.factors["delta"] < 1)

    def check_sensitivity(self):
        return self.factors["sensitivity"] > 0
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
