import numpy as np

from rng.mrg32k3a import MRG32k3a
from oracles.mm1queue import MM1Queue
from base import Solution

myoracle = MM1Queue()
myoracle = MM1Queue(params={"lambd": 1.5})
print(myoracle.params)

rng_list = [MRG32k3a() for _ in range(myoracle.n_rngs)]
# print(rng_list)
myoracle.attach_rngs(rng_list)
# print(myoracle.rng_list)

print('For x = (1,), is_simulatable should be True and is {}'.format(myoracle.check_simulatable_x(x=(1,))))
print('For x = [1], is_simulatable should be True(?) and is {}'.format(myoracle.check_simulatable_x(x=(1,))))
print('For x = (-1,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_x(x=(-1,))))
print('For x = (0,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_x(x=(0,))))
print('For x = (1,2), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_x(x=(1,2))))
print('For x = "hi", is_simulatable should be False and is {}'.format(myoracle.check_simulatable_x(x='hi')))

mysoln = Solution(x=(3,))
response, gradient = myoracle.replicate(mysoln.x)
print('For a single replication:')
print('The responses are {}'.format(response))
print('The gradients are {}'.format(gradient))

for rng in myoracle.rng_list:
    rng.reset_substream()
myoracle.simulate(mysoln, m=3)

all_resp = [True, True]
is_obj = [True, False]

print('\nFor a batch of 3 replications:\n')
print('The responses are {}'.format(mysoln.responses))
print('The mean responses are {}'.format(mysoln.response_mean(which=all_resp)))
print('The first mean response is {}'.format(mysoln.response_mean(which=is_obj)))
print('The variances of the responses are {}'.format(mysoln.response_var(which=all_resp)))
print('The variance of the first response is {}'.format(mysoln.response_var(which=is_obj)))
print('The standard errors of the responses are {}'.format(mysoln.response_std_error(which=all_resp)))
print('The standard error of the first responses is {}'.format(mysoln.response_std_error(which=is_obj)))
print('The covariances of the responses are {}'.format(mysoln.response_cov(which=all_resp)))
print('The variance of the first response is (again) {}'.format(mysoln.response_cov(which=is_obj)))
print('')
print('The gradients are {}'.format(mysoln.gradients))
print('The mean of the first gradient is {}'.format(mysoln.gradient_mean(which=is_obj)))
print('The variance of the first gradients is {}'.format(mysoln.gradient_var(which=is_obj)))
print('The standard error of the first gradient is {}'.format(mysoln.gradient_std_error(which=is_obj)))
print('The variance of the first gradients is (again) {}'.format(mysoln.gradient_cov(which=is_obj)))