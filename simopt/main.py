import numpy as np

from rng.mrg32k3a import MRG32k3a
from oracles.gig1queue import GIG1Queue
from base import Solution

myoracle = GIG1Queue()
myoracle = GIG1Queue(params={"lambd": 10})
print(myoracle.params)

rng_list = [MRG32k3a() for _ in range(myoracle.n_rngs)]
# print(rng_list)
myoracle.attach_rngs(rng_list)
# print(myoracle.rng_list)

print('For x = (1,), is_simulatable should be True and is {}'.format(myoracle.check_simulatable(x=(1,))))
print('For x = [1], is_simulatable should be True(?) and is {}'.format(myoracle.check_simulatable(x=(1,))))
print('For x = (-1,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable(x=(-1,))))
print('For x = (0,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable(x=(0,))))
print('For x = (1,2), is_simulatable should be False and is {}'.format(myoracle.check_simulatable(x=(1,2))))
print('For x = "hi", is_simulatable should be False and is {}'.format(myoracle.check_simulatable(x='hi')))

mysoln = Solution(x=(1.5,))
response, gradient = myoracle.replicate(mysoln.x)
print('For a single replication:')
print('The responses are {}'.format(response))
print('The gradients are {}'.format(gradient))
for rng in myoracle.rng_list:
    rng.reset_substream()
myoracle.simulate(mysoln, m=3)

print('For a batch of 3 replications:')
print('The responses are {}'.format(mysoln.responses))
print('The mean responses are {}'.format(mysoln.response_mean()))
print('The variances of the responses are {}'.format(mysoln.response_var()))
print('The standard errors of the responses are {}'.format(mysoln.response_std_error()))
print('The covariances of the responses are {}'.format(mysoln.response_cov()))

print('The gradients are {}'.format(mysoln.gradients))
#print('The mean gradients are {}'.format(mysoln.gradient_mean()))
#print('The variances of the gradients are {}'.format(mysoln.gradient_var()))
#print('The standard errors of the gradients are {}'.format(mysoln.gradient_std_error()))
#print('The covariances of the gradients are {}'.format(mysoln.gradient_cov()))