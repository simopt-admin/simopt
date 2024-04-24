"""
This script is intended to help with debugging problems and solvers.
It create a problem-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

# Import the ProblemSolver class and other useful functions
from simopt.directory import problem_directory
from simopt.experiment_base import ProblemsSolvers, read_experiment_results, post_normalize, plot_terminal_progress, plot_terminal_scatterplots, plot_area_scatterplots, plot_progress_curves, plot_solvability_profiles
from mrg32k3a.mrg32k3a import MRG32k3a
import matplotlib.pyplot as plt

def rebase(random_rngs, n):
    '''
    Advance substream of each rng in random_rngs by n steps
    '''
    new_rngs = []
    for rng in random_rngs:
        stream_index = rng.s_ss_sss_index[0]
        substream_index = rng.s_ss_sss_index[1]
        subsubstream_index = rng.s_ss_sss_index[2]
        new_rngs.append(MRG32k3a(s_ss_sss_index=[stream_index, substream_index + n, subsubstream_index]))
    random_rngs = new_rngs
    return random_rngs

solver_names = ["PGD-B", "PGD-I", "PGD-Z", "PGD-SS", "AS-B", "AS-I", "AS-Z", "AS-SS", "FW-B", "FW-I", "FW-Z", "FW-SS"]

#Create a list for each problem
problem_names = ["OPENJ-1", "SAN-2", "SMF-1", "SMFCVX-1", "CC-1"]
num_random_instances = [5, 5, 5, 5, 5] # Number of random instances
all_problem_fixed_factors = [{}, {}, {}, {}, {}] # Fixed problem factors
all_model_fixed_factors = [{}, {}, {}, {}, {}] # Fixed model factors
problem_renames = ["OPENJ", "SAN", "SMF", "SMFCVX", "Cascade"] # Prefix of random problem names


rand_problems = []
# Generate random problems
for problem_idx in range(len(problem_names)):        
    default_rand_problem = problem_directory[problem_names[problem_idx]](random=True)
    rnd_problem_factors_rngs = [MRG32k3a(s_ss_sss_index=[2, 4 + num_random_instances[problem_idx], ss]) 
                    for ss in range(default_rand_problem.n_rngs)]
    rnd_model_factors_rngs = [MRG32k3a(s_ss_sss_index=[2, 4, ss]) for ss in range(default_rand_problem.model.n_random)]
    

    for rand_inst_idx in range(num_random_instances[problem_idx]):
        rnd_problem_factors_rngs = rebase(rnd_problem_factors_rngs, 1)  # Advance the substream for different instances
        rnd_model_factors_rngs = rebase(rnd_model_factors_rngs, 1)

        # Generate random problem and model factors
        rand_problem = problem_directory[problem_names[problem_idx]](name = problem_renames[problem_idx]+ f'-R{rand_inst_idx+1}',
                                                                        fixed_factors = all_problem_fixed_factors[problem_idx], 
                                                                        model_fixed_factors = all_model_fixed_factors[problem_idx],
                                                                        random=True, random_rng=rnd_model_factors_rngs)
        rand_problem.attach_rngs(rnd_problem_factors_rngs)

        fixed_factors = {}  # Will hold problem factor values for current random problem.
        model_fixed_factors = {}  # Will hold model factor values for current random problem.

        # Retrieve the generated random factor values.
        for factor in rand_problem.factors:
            fixed_factors[factor] = rand_problem.factors[factor]
        for factor in rand_problem.model.factors:
            model_fixed_factors[factor] = rand_problem.model.factors[factor]
        
        # Create random problem based on the generated factor values.
        rand_problem = problem_directory[problem_names[problem_idx]](name = problem_renames[problem_idx]+ f'-R{rand_inst_idx+1}',
                                                                        fixed_factors = fixed_factors, 
                                                                        model_fixed_factors = model_fixed_factors)

        rand_problems.append(rand_problem)



# Initialize an instance of the experiment class.
experiment_name = 'RAND_EXP1'
mymetaexperiment = ProblemsSolvers(solver_names=solver_names, problems = rand_problems, file_name_path = f"./experiments/outputs/group_{experiment_name}.pickle")

# Write to log file
mymetaexperiment.log_group_experiment_results()

# Run a fixed number of macroreplications of each solver on each problem.
mymetaexperiment.run(n_macroreps=30)

print("Post-processing results.")
# Run a fixed number of postreplications at all recommended solutions.
mymetaexperiment.post_replicate(n_postreps=50)
# Find an optimal solution x* for normalization.
mymetaexperiment.post_normalize(n_postreps_init_opt=50)

print("Plotting results.")
# color_palette = [
#     "#00429d",
#     "#2558ac",
#     "#3f71b3",
#     "#568aba",
#     "#6ba2c1",
#     "#80bac8",
#     "#97d2cf",
#     "#afead6",
#     "#c9e2dd",
#     "#e3fbe4",
#     "#fdffbc",
#     "#f5c25c"
# ]
cmap = plt.get_cmap('tab20')
# Generate 20 distinct colors from the colormap
color_palette = [cmap(i) for i in range(cmap.N)]

# Produce basic plots of the solvers on the problems.
plot_solvability_profiles(mymetaexperiment.experiments, plot_type="cdf_solvability", color_palette = color_palette)
plot_solvability_profiles(mymetaexperiment.experiments, plot_type="quantile_solvability", beta = 0.9, color_palette = color_palette)
plot_solvability_profiles(mymetaexperiment.experiments, plot_type="diff_cdf_solvability", color_palette = color_palette)
plot_solvability_profiles(mymetaexperiment.experiments, plot_type="diff_quantile_solvability", beta = 0.9, color_palette = color_palette)

plot_terminal_scatterplots(mymetaexperiment.experiments)

plot_area_scatterplots(mymetaexperiment.experiments)

plot_terminal_progress(mymetaexperiment.experiments, plot_type="box", normalize=False)
plot_terminal_progress(mymetaexperiment.experiments, plot_type="box", normalize=True)
plot_terminal_progress(mymetaexperiment.experiments, plot_type="violin", normalize=False, all_in_one=False)
plot_terminal_progress(mymetaexperiment.experiments, plot_type="violin", normalize=True)

n_solvers = len(mymetaexperiment.experiments)
n_problems = len(mymetaexperiment.experiments[0])
CI_param = True


for i in range(n_problems):
    plot_progress_curves([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type = 'mean', normalize = False, all_in_one = True, 
                            plot_CIs = CI_param, print_max_hw = True, color_palette=color_palette)
    plot_progress_curves([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type = 'quantile', beta = 0.9, normalize = False, all_in_one = True, 
                            plot_CIs = CI_param, print_max_hw = True, color_palette=color_palette)

    plot_progress_curves([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type = 'mean', normalize = True, all_in_one = True, 
                            plot_CIs = CI_param, print_max_hw = True, color_palette=color_palette)
    plot_progress_curves([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type = 'quantile', beta = 0.9, normalize = True, all_in_one = True, 
                            plot_CIs = CI_param, print_max_hw = True, color_palette=color_palette)

# Plots will be saved in the folder experiments/plots.
print("Finished. Plots can be found in experiments/plots folder.")