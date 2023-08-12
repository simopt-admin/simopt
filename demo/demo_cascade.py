"""
This script is intended to help with debugging problems and solvers.
It create a problem-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import ProblemSolver, read_experiment_results, post_normalize, plot_terminal_progress, plot_terminal_scatterplots, plot_progress_curves, plot_solvability_cdfs

# solver_names = {"RNDSRCH", "ASTRODF", "NELDMD"}
solver_names = {"RNDSRCH", "NELDMD", "ASTRODF", "PGD-SS", "FW", "ACTIVESET"}

# problem_names = {"SAN-2"} # CNTNEWS-1"} #, "SAN-1"}
# problem_names = {"CASCADE-1"} # CNTNEWS-1"} #, "SAN-1"}
problem_names = {"CASCADE-1"} # CNTNEWS-1"} #, "SAN-1"}

# solver_name = "RNDSRCH"  # Random search solver
# problem_name = "CNTNEWS-1"  # Continuous newsvendor problem
# solver_name = <solver_name>
# problem_name = <problem_name>
# Open the pickle file in binary mode
with open('runtime3.pickle', 'rb') as file:
    # Load the object from the pickle file
    times = pickle.load(file)

opt_seed_size_dict = {}
opt_seed_set_dict = {}

for problem_name in problem_names:

    problem_experiments = []
    for solver_name in solver_names:
        print(f"Testing solver {solver_name} on problem {problem_name}.")
        # Initialize an instance of the experiment class.
        myexperiment = ProblemSolver(solver_name, problem_name)

        file_name_path = "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"

        # If the solver runs have already been performed, uncomment the
        # following pair of lines (and uncommmen the myexperiment.run(...)
        # line above) to read in results from a .pickle file.
        myexperiment = read_experiment_results(file_name_path)

        # Get the index of the maximum value
        max_value = float('-inf')  # Initialize with negative infinity
        max_index = None

        # Iterate over each sublist and find the maximum value
        for i, sublist in enumerate(myexperiment.all_est_objectives):
            for j, value in enumerate(sublist):
                if value > max_value:
                    max_value = value
                    max_index = (i, j)
        xstar =  np.array(myexperiment.all_recommended_xs)[max_index[0]][max_index[1]]
        opt_seed_set_dict[solver_name] = xstar
        print(xstar)
        opt_seed_size = np.ceil(np.sum(xstar) * 3)
        opt_seed_size_dict[solver_name] = opt_seed_size

        # print("Post-processing results.")
        # # Run a fixed number of postreplications at all recommended solutions.
        # myexperiment.post_replicate(n_postreps=100)


        # problem_experiments.append(myexperiment)

    # # Find an optimal solution x* for normalization.
    # post_normalize(problem_experiments, n_postreps_init_opt=100)
print(times)
print(opt_seed_size_dict)


opt_seed_size_dict = dict(sorted(opt_seed_size_dict.items(), key=lambda x: x[0]))
# plt.bar(opt_seed_size_dict.keys(), opt_seed_size_dict.values())
# plt.savefig('experiments/plots/cascade_seed_size.png')

# times = dict(list(times.items())[1:])
times = dict(sorted(times.items(), key=lambda x: x[0]))

# plt.bar(times.keys(), times.values(), width= 0.5)
# plt.xlabel('Algorithms')
# plt.ylabel('CPU Run Time (Secs)')

# plt.savefig('experiments/plots/cascadetime2_runtime.png',bbox_inches='tight')

# Create lists to store location, value, and algorithm name
locations = []
values = []
algorithms = []
opt_seed_set_dict = dict(sorted(opt_seed_set_dict.items(), key=lambda x: x[0]))
# Extract data from the dictionary
for algorithm, results in opt_seed_set_dict.items():
    locations.extend(range(1, 31))  # Locations from 1 to 30
    values.extend(np.array(results))
    algorithms.extend([algorithm] * 30)  # Repeat the algorithm name 30 times

# Create a DataFrame from the extracted data
results_df = pd.DataFrame({'Location': locations, 'Value': values, 'Algorithm': algorithms})

# Normalize the values to the range [0, 1]
min_value = np.min(values)
max_value = np.max(values)
normalized_values = (values - min_value) / (max_value - min_value)

# Generate scatter plot using seaborn
sns.scatterplot(x='Location', y='Value', hue='Algorithm', style = 'Algorithm', alpha=normalized_values, data=results_df)

# Add labels
plt.xlabel('Node Numbers')
plt.ylabel('Activation Probabilities')
plt.savefig('experiments/plots/cascade_scatter.png',bbox_inches='tight')


# # Re-compile problem-solver results.
# myexperiments = []
# for solver_name in solver_names:
#     #solver_experiments = []
#     for problem_name in problem_names:
#         file_name_path = "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
#         myexperiment = read_experiment_results(file_name_path)
#     myexperiments.append(myexperiment)
#    solver_experiments.append(myexperiment)
    # myexperiments.append(solver_experiments)

# print("Plotting results.")
# # Produce basic plots.
# plot_terminal_progress(experiments=myexperiments, plot_type="box", normalize=False)
# plot_terminal_progress(experiments=myexperiments, plot_type="box", normalize=True)
# plot_terminal_progress(experiments=myexperiments, plot_type="violin", normalize=False, all_in_one=False)
# plot_terminal_progress(experiments=myexperiments, plot_type="violin", normalize=True)
# plot_progress_curves(experiments=myexperiments, plot_type="all", normalize=False)
# plot_progress_curves(experiments=myexperiments, plot_type="mean", normalize=False)
# plot_progress_curves(experiments=myexperiments, plot_type="quantile", beta=0.90, normalize=False)
# plot_solvability_cdfs(experiments=myexperiments, solve_tol=0.1)
# #plot_terminal_scatterplots(experiments = myexperiments, all_in_one=False)

# # Plots will be saved in the folder experiments/plots.
# print("Finished. Plots can be found in experiments/plots folder.")
