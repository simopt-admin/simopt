"""
This script is the user interface for generating multiple random problem instances and
solve them by specified solvers.
It create problem-solver groups and runs multiple
macroreplications of each problem-solver pair. To run the file, user need
to import the solver and probelm they want to build random instances at the beginning,
and also provide an input file, which include the information needed to 
build random instances (the name of problem, number of random instances to 
generate, and some overriding factors).
"""

import sys
import os.path as o
import os
import re
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the ProblemsSolvers class and other useful functions
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles
from rng.mrg32k3a import MRG32k3a
from simopt.base import Solution
from simopt.models.smf import SMF_Max
from simopt.models.rmitd import RMITDMaxRevenue
from simopt.models.san_2 import SANLongestPath, SANLongestPathConstr
from simopt.models.mm1queue import MM1MinMeanSojournTime


# !! When testing a new solver/problem, first import problems from the random code file,
# Then create a test_input.txt file in your computer.
# There you should add the import statement and an entry in the file
# You need to specify name of solvers and problems you want to test in the file by 'solver_name'
# And specify the problem related informations by problem = [...]
# All lines start with '#' will be counted as commend and will not be implemented
# See the following example for more details.

# Ex:
# To create two random instance of SAN and three random instances of SMF:
# In the demo_user.py, modify:
# from simopt.models.smf import SMF_Max
# from simopt.models.san_2 import SANLongestPath
# In the input information file (test_input.txt), include the following lines:
# solver_names = ["RNDSRCH", "ASTRODF", "NELDMD"]
# problem1 = [SANLongestPath, 2, {'num_nodes':8, 'num_arcs':12}]
# problem2 = [SMF_Max, 3, {'num_nodes':7, 'num_arcs':16}]

# Grab information from the input file
def get_info(path):
    L = []
    with open(path) as file:
        lines = [line.rstrip() for line in file]
        for line in lines:
            if not line.startswith("#") and line:
                L.append(line)
    lines = L
    command_lines = []
    problem_sets = []
    for line in lines:
        if 'import' in line:
            command_lines.append(line)
        elif 'solver_names' in line:
            solver_names = line
        else:
            problem_sets.append(line)

    for i in command_lines:
        exec(i)
    
    problems = []
    solver_names = eval(re.findall(r'\[.*?\]', solver_names)[0])
    for l in problem_sets:
        o = re.findall(r'\[.*?\]', l)[0]
        problems.append(eval(o))
    
    problem_sets = [p[0] for p in problems]
    L_num = [p[1] for p in problems]
    L_para = [p[2] for p in problems]
    
    return solver_names, problem_sets, L_num, L_para

# Read input file and process information
path = input('Please input the path of the input file: ')
if "'" in path:  # If the input path already has quotation marks
    path = path.replace("'", "")
    
solver_names, problem_set, L_num, L_para = get_info(path)
rands = [True for i in range(len(problem_set))]

# Check whether the input file is valid
if len(L_num) != len(problem_set) or len(L_para) != len(problem_set):
    print('Invalid input. The input number of random instances does not match with the number of problems you want.')
    print('Please check your input file')

def rebase(random_rng, n):
    new_rngs = []
    for rng in random_rng:
        stream_index = rng.s_ss_sss_index[0]
        substream_index = rng.s_ss_sss_index[1]
        subsubstream_index = rng.s_ss_sss_index[2]
        new_rngs.append(MRG32k3a(s_ss_sss_index=[stream_index, substream_index + n, subsubstream_index]))
    random_rng = new_rngs
    return random_rng

myproblems = problem_set

# Check whether the problem is random
for i in range(len(problem_set)):
    if L_num[i] == 0:
        L_num[i] = 1
        rands[i] = False
    else:
        rands[i] = True

problems = []
problem_names = []

def generate_problem(i, myproblems, rands, problems, L_num, L_para):
    print('For problem ', myproblems[i]().name, ':')  
    model_fixed_factors = L_para[i]
    
    name = myproblems[i]
    myproblem = name(model_fixed_factors=model_fixed_factors, random=rands[i])
    random_rng = [MRG32k3a(s_ss_sss_index=[2, 4 + L_num[i], ss]) for ss in range(myproblem.n_rngs)]
    rng_list2 = [MRG32k3a(s_ss_sss_index=[2, 4, ss]) for ss in range(myproblem.model.n_random)]
    
    if rands[i] == False:  # Determinant case
        problems.append(myproblem)
        myproblem.name = str(myproblem.model.name) + str(0)
        problem_names.append(myproblem.name)
        print('')
    
    else:
        for j in range(L_num[i]):
            random_rng = rebase(random_rng, 1)  # Advance the substream for different instances
            rng_list2 = rebase(rng_list2, 1)
            name = myproblems[i]
            myproblem = name(model_fixed_factors=model_fixed_factors, random=rands[i], random_rng=rng_list2)
            myproblem.attach_rngs(random_rng)
            # myproblem.name = str(myproblem.model.name) + str(j)
            myproblem.name = str(myproblem.name) + '-' + str(j)
            problems.append(myproblem)
            problem_names.append(myproblem.name)
            print('')
    
    return problems, problem_names
   
# Generate problems
for i in range(len(L_num)):
        problems, problem_names = generate_problem(i, myproblems, rands, problems, L_num, L_para)

# Initialize an instance of the experiment class.
mymetaexperiment = ProblemsSolvers(solver_names=solver_names, problems = problems)

# Run a fixed number of macroreplications of each solver on each problem.
mymetaexperiment.run(n_macroreps=3)

print("Post-processing results.")
# Run a fixed number of postreplications at all recommended solutions.
mymetaexperiment.post_replicate(n_postreps=20)
# Find an optimal solution x* for normalization.
mymetaexperiment.post_normalize(n_postreps_init_opt=20)

print("Plotting results.")
# Produce basic plots of the solvers on the problems.
plot_solvability_profiles(experiments=mymetaexperiment.experiments, plot_type="cdf_solvability")

# Plots will be saved in the folder experiments/plots.
print("Finished. Plots can be found in experiments/plots folder.")