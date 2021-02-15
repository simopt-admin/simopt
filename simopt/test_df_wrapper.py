import numpy as np
from rng.mrg32k3a import MRG32k3a
from oracles.mm1queue import MM1Queue
from base import DesignPoint

oracle_fixed_factors = {'mu': 5} # default overrides from GUI, others set as defaults
# Create an oracle object
myoracle = MM1Queue(fixed_factors=oracle_fixed_factors)

# Create a design point
mydesigndict = {'lambda': 1} # extracted from row of design matrix
mydesignpt = DesignPoint(design_oracle_factors=mydesigndict, oracle=myoracle)

# Run a single replication of the oracle at the design_point
mydesignpt.replicate(m=10)

print('I ran this.')