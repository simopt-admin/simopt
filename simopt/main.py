import numpy as np

from rng.mrg32k3a import MRG32k3a
from problems.mm1_min_mean_sojourn_time import MM1MinMeanSojournTime
from oracles.mm1queue import MM1Queue
from base import Solution

fixed_factors = {
    "lambda": 1.0
    #"lambda": 1.5,
    #"warmup": 20,
    #"people": 50
}
myproblem = MM1MinMeanSojournTime(fixed_factors)
#print(myProblem.dim)
print(myproblem.oracle.factors)

rng_list = [MRG32k3a() for _ in range(myproblem.oracle.n_rngs)]
#print(rng_list)
myproblem.oracle.attach_rngs(rng_list)
#print(myproblem.oracle.rng_list)

mysoln = Solution(x=[3.0], problem=myproblem)
print(mysoln.det_objectives)
print(mysoln.det_objectives_gradients)
#print(mysoln.x)
#print(mysoln.decision_factors)

myproblem.oracle.factors.update(mysoln.decision_factors)
responses, gradients = myproblem.oracle.replicate()
print('For a single replication:')
print('The responses are {}'.format(responses))
print('The gradients are {}'.format(gradients))

for rng in myproblem.oracle.rng_list:
    rng.reset_substream()

#myproblem.simulate(mysoln, m=1)
#myproblem.simulate(mysoln, m=1)
#myproblem.simulate(mysoln, m=1)
#print(myproblem.oracle.factors) 
myproblem.simulate(mysoln, m=3)
#print(myproblem.oracle.factors) 

# all_resp = [True, True]
# is_obj = [True, False]

# print('\nFor a batch of 3 replications:\n')
#print('The responses are {}'.format(mysoln.responses))
# print('The mean responses are {}'.format(mysoln.response_mean(which=all_resp)))
# print('The first mean response is {}'.format(mysoln.response_mean(which=is_obj)))
# print('The variances of the responses are {}'.format(mysoln.response_var(which=all_resp)))
# print('The variance of the first response is {}'.format(mysoln.response_var(which=is_obj)))
# print('The standard errors of the responses are {}'.format(mysoln.response_std_error(which=all_resp)))
# print('The standard error of the first response is {}'.format(mysoln.response_std_error(which=is_obj)))
# print('The covariances of the responses are {}'.format(mysoln.response_cov(which=all_resp)))
# print('The variance of the first response is (again) {}'.format(mysoln.response_cov(which=is_obj)))
# print('')
#print('The gradients are {}'.format(mysoln.gradients))
# print('The mean of the first gradient is {}'.format(mysoln.gradient_mean(which=is_obj)))
# print('The variance of the first gradients is {}'.format(mysoln.gradient_var(which=is_obj)))
# print('The standard error of the first gradient is {}'.format(mysoln.gradient_std_error(which=is_obj)))
# print('The variance of the first gradients is (again) {}'.format(mysoln.gradient_cov(which=is_obj)))

print('The objectives are {}'.format(mysoln.objectives[:mysoln.n_reps]))
print('The gradients of the objectives are {}'.format(mysoln.objectives_gradients[:mysoln.n_reps]))
print('The stochastic constraint LHSs are {}'.format(mysoln.stoch_constraints[:mysoln.n_reps]))
print('The gradients of the stochastic constraint LHSs are {}'.format(mysoln.stoch_constraints_gradients[:mysoln.n_reps]))

# Print summary statistics
print('The mean of the objectives are {}'.format(mysoln.objectives_mean))
print('The variance of the objectives are {}'.format(mysoln.objectives_var))
print('The standard error of the objectives are {}'.format(mysoln.objectives_stderr))
print('The covariance matrix of the objectives is {}'.format(mysoln.objectives_cov))
print('The mean of the objectives gradients are {}'.format(mysoln.objectives_gradients_mean))
print('The variance of the objectives gradients are {}'.format(mysoln.objectives_gradients_var))
print('The standard error of the objectives gradients are {}'.format(mysoln.objectives_gradients_stderr))
print('The covariance matrix of the objectives gradients is {}'.format(mysoln.objectives_gradients_cov))
print('The mean of the stochastic constraints are {}'.format(mysoln.stoch_constraints_mean))
print('The variance of the stochastic constraints are {}'.format(mysoln.stoch_constraints_var))
print('The standard error of the stochastic constraints are {}'.format(mysoln.stoch_constraints_stderr))
print('The covariance matrix of the stochastic constraints is {}'.format(mysoln.stoch_constraints_cov))
print('The mean of the stochastic constraints gradients are {}'.format(mysoln.stoch_constraints_gradients_mean))
print('The variance of the stochastic constraints gradients are {}'.format(mysoln.stoch_constraints_gradients_var))
print('The standard error of the stochastic constraints gradients are {}'.format(mysoln.stoch_constraints_gradients_stderr))
print('The covariance matrix of the stochastic constraints gradients is {}'.format(mysoln.stoch_constraints_gradients_cov))

myproblem.simulate(mysoln, m=300)


# noise_factors = {
#     "lambda": 1.5,
#     "warmup": 20,
#     "people": 50
# }
# myneworacle = MM1Queue(noise_factors)
# for key in noise_factors:
#     print(key, myneworacle.check_simulatable_factor(key))

# OLD VERSION OF DECISION VARIABLES & FACTORS

# myoracle = MM1Queue()
# myoracle = MM1Queue(params={"lambd": 1.5})
# print(myoracle.params)

# rng_list = [MRG32k3a() for _ in range(myoracle.n_rngs)]
# # print(rng_list)
# myoracle.attach_rngs(rng_list)
# # print(myoracle.rng_list)

# print('For x = (1,), is_simulatable should be True and is {}'.format(myoracle.check_simulatable_x(x=(1,))))
# print('For x = [1], is_simulatable should be True(?) and is {}'.format(myoracle.check_simulatable_x(x=(1,))))
# print('For x = (-1,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_x(x=(-1,))))
# print('For x = (0,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_x(x=(0,))))
# print('For x = (1,2), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_x(x=(1,2))))
# print('For x = "hi", is_simulatable should be False and is {}'.format(myoracle.check_simulatable_x(x='hi')))

# mysoln = Solution(x=(3,))
# response, gradient = myoracle.replicate(mysoln.x)
# print('For a single replication:')
# print('The responses are {}'.format(response))
# print('The gradients are {}'.format(gradient))

# for rng in myoracle.rng_list:
#     rng.reset_substream()
# myoracle.simulate(mysoln, m=3)