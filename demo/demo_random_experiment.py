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
import argparse
import matplotlib.pyplot as plt

sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the ProblemsSolvers class and other useful functions
from simopt.directory import problem_directory
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles, plot_progress_curves
from mrg32k3a.mrg32k3a import MRG32k3a
import warnings
warnings.filterwarnings("ignore")
# !! When testing a new solver/problem, first import problems from the random code file,
# Then create a test_input.txt file in your computer.
# There you should add the import statement and an entry in the file
# You need to specify name of solvers and problems you want to test in the file by 'solver_name'
# And specify the problem related informations by problem = [...]
# All lines start with '#' will be counted as commend and will not be implemented
# See the following example for more details.


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


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add arguments with default values
    parser.add_argument('--n_macroreps', type=int, default=30, help='Number of macrorepetitions')
    parser.add_argument('--n_postreps', type=int, default=50, help='Number of postrepetitions')

    # Parse the arguments
    args = parser.parse_args()

    print(f'n_macroreps: {args.n_macroreps}, n_postreps: {args.n_postreps}')

    # Create a list named "solver_names"
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

    # Write to log file.
    mymetaexperiment.log_group_experiment_results()

    # Run a fixed number of macroreplications of each solver on each problem.
    mymetaexperiment.run(n_macroreps=args.n_macroreps)

    print("Post-processing results.")
    # Run a fixed number of postreplications at all recommended solutions.
    mymetaexperiment.post_replicate(n_postreps=args.n_postreps)
    # Find an optimal solution x* for normalization.
    mymetaexperiment.post_normalize(n_postreps_init_opt=args.n_postreps)

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

    n_solvers = len(mymetaexperiment.experiments)
    n_problems = len(mymetaexperiment.experiments[0])
    CI_param = True


    for i in range(n_problems):
        plot_progress_curves([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type = 'mean', normalize = False, all_in_one = True, 
                             plot_CIs = CI_param, print_max_hw = True, color_palette=color_palette)
        plot_progress_curves([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type = 'quantile', beta = 0.9, normalize = False, all_in_one = True, 
                             plot_CIs = CI_param, print_max_hw = True, color_palette=color_palette)


    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")

if __name__ == "__main__":
    main()