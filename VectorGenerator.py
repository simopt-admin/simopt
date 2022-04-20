#random_num_generator
import random
from re import A
import numpy as np
import matplotlib.pyplot as plt

# n = number of factors
# M = summation
a = []
b = []
c = []



sum = 4
n_factors = 2
x = [0]
for i in range(n_factors-1):
    random_num = random.randint(1,sum-1)
    while random_num in x:
        random_num = random.randint(1,sum-1)
    x.append(random_num)
x.append(sum)
x = np.sort(x)
y = []
for i in range(1,n_factors+1):
    num = x[i] - x[i-1]
    y.append(num)
print(y)

    # print("Random Vector: ", y)
    # print("")
    # print("Number of factors: ", len(y))
    # print("n: ", n)
    # print("")
    # print("Sum: ", sum(y))
    # print("M: ", M)

#     if y == [3,1]:
#         a.append(1)
#     elif y == [1,3]:
#         b.append(2)
#     elif y == [2,2]:
#         c.append(3)
# lst = a+b+c
# print(len(a))
# print(len(b))
# print(len(c))

# print(len(lst))
# print(lst)

# count, bins, ignored = plt.hist(lst, 3, facecolor='green') 

# plt.xlabel('X~U[0,1]')
# plt.ylabel('Count')
# plt.title("Uniform Distribution Histogram (Bin size 3)")
# plt.show() 