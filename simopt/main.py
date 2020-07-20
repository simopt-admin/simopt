import numpy as np

from rng.mrg32k3a import MRG32k3a
from oracles.gig1queue import GIG1Queue
from base import aggregate

myoracle = GIG1Queue()
myoracle = GIG1Queue(params={"lambd": 10})
print(myoracle.params)

rng_list = [MRG32k3a() for _ in range(myoracle.n_rngs)]
# print(rng_list)
myoracle.attach_rngs(rng_list)
# print(myoracle.rng_list)

# print(myoracle.check_simulatable(x=[1]))
# print(myoracle.check_simulatable(x=[-1]))
# print(myoracle.check_simulatable(x=[0]))
# print(myoracle.check_simulatable(x=[1,2]))
# print(myoracle.check_simulatable(x='hi'))

# output = myoracle.simulate(x=[1.5])
# print(output["response"])
# print(output["gradient"])

# response, gradient = myoracle.simulate(x=[1.5])
# print(response)
# print(gradient)

responses, gradients = myoracle.batch(x=[1.5], m=10)
# print(responses)
# print(gradients)

response_mean, response_cov, gradient_mean, gradient_cov = aggregate(responses, gradients)
print(response_mean)
print(response_cov)
# print(gradient_mean)
# print(gradient_cov)

# responses = np.array([[1,2],[4,8],[16,32]])
# gradients = np.array([[[1,2,3,4],[10,9,8,7]],[[2,3,4,5],[9,8,7,6]],[[3,4,5,6],[8,7,6,5]]])
# response_mean, response_cov, gradient_mean, gradient_cov = aggregate(responses, gradients)
# print(response_mean)
# print(response_cov)
# print(gradient_mean)
# print(gradient_cov)