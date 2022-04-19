#random_num_generator
import random
import numpy as np

# n = number of factors
# M = summation
 
M = 4
n = 2
x = [0]
for i in range(n-1):
    random_num = random.randint(1,M-1)
    while random_num in x:
        random_num = random.randint(1,M-1)
    x.append(random_num)
x.append(M)
x = np.sort(x)
y = []
for i in range(1,n+1):
    num = x[i] - x[i-1]
    y.append(num)

print("Random Vector: ", y)
print("")
print("Number of factors: ", len(y))
print("n: ", n)
print("")
print("Sum: ", sum(y))
print("M: ", M)
