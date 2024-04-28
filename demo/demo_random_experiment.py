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
import seaborn as sns
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the ProblemsSolvers class and other useful functions
from simopt.directory import problem_directory
from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles, plot_progress_curves, plot_terminal_progress, plot_terminal_scatterplots, plot_area_scatterplots, plot_progress_curves, plot_solvability_profiles
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
    # solver_names = ["PGD-B", "AS-B", "AS-I", "AS-SS", "FW-I"]

    # Create a list for each problem
    problem_names = ["SAN-2", "SMF-1", "SMFCVX-1", "CC-1"]
    num_random_instances = [5, 5, 5, 5] # Number of random instances
    all_problem_fixed_factors = [{}, {}, {}, {}] # Fixed problem factors
    all_model_fixed_factors = [{}, {}, {}, {}] # Fixed model factors
    problem_renames = ["SAN", "SMF", "SMFCVX", "Cascade"] # Prefix of random problem names

    # problem_names = ["SAN-2", "SMF-1", "SMFCVX-1"]
    # num_random_instances = [2, 2, 2] # Number of random instances
    # all_problem_fixed_factors = [{}, {}, {}, {}] # Fixed problem factors
    # all_model_fixed_factors = [{}, {}, {}, {}] # Fixed model factors
    # problem_renames = ["SAN", "SMF", "SMFCVX"] # Prefix of random problem names

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
    experiment_name = 'RAND_EXP_4P_12S'
    mymetaexperiment = ProblemsSolvers(solver_names=solver_names, problems = rand_problems, file_name_path = f"./experiments/outputs/group_{experiment_name}.pickle")

    # Write to log file.
    mymetaexperiment.log_group_experiment_results()

    # Run a fixed number of macroreplications of each solver on each problem.
    mymetaexperiment.run(n_macroreps=args.n_macroreps)

    print("Post-processing results.")
    # Run a fixed number of postreplications at all recommended solutions.
    mymetaexperiment.post_replicate(n_postreps=args.n_postreps, crn_across_macroreps=True)
    # Find an optimal solution x* for normalization.
    mymetaexperiment.post_normalize(n_postreps_init_opt=args.n_postreps)

    print("Plotting results.")

    color_palette = [
        "#ff0000",  # Red
        "#00ff00",  # Green
        "#0000ff",  # Blue
        "#FFD700",  # Gold
        "#ff00ff",  # Magenta
        "#00ffff",  # Cyan
        "#000000",  # Black
        "#808080",  # Gray (Mid Gray)
        "#880000",  # Dark Red
        "#008800",  # Dark Green
        "#000088",  # Dark Blue
        "#888800"   # Olive
    ]

    # tab20b_r_palette = sns.color_palette("tab20b_r", n_colors=12)

    # # Convert the RGB colors to Hex format
    # tab20b_r_hex_colors = tab20b_r_palette.as_hex()


    # # cmap = plt.get_cmap('tab20')
    # # Generate 20 distinct colors from the colormap
    # color_palette = [c for c in tab20b_r_hex_colors]

    # Produce basic plots of the solvers on the problems.
    plot_solvability_profiles(mymetaexperiment.experiments, plot_type="cdf_solvability", print_max_hw=True, solve_tol=0.2, color_palette = color_palette)

    plot_solvability_profiles(mymetaexperiment.experiments, plot_type="cdf_solvability", print_max_hw=True, solve_tol=0.1, color_palette = color_palette)

    # plot_terminal_scatterplots(mymetaexperiment.experiments)

    # plot_area_scatterplots(mymetaexperiment.experiments, plot_CIs=False, print_max_hw=True)

    n_solvers = len(mymetaexperiment.experiments)
    n_problems = len(mymetaexperiment.experiments[0])

    for i in range(n_problems):
        plot_terminal_progress([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="box", normalize=False)
        plot_terminal_progress([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="box", normalize=True)
        plot_terminal_progress([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="violin", normalize=False)
        plot_terminal_progress([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="violin", normalize=True)

        plot_progress_curves([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type = 'mean', normalize = False, all_in_one = True, 
                             plot_CIs = True, print_max_hw = True, color_palette=color_palette)

        plot_progress_curves([mymetaexperiment.experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type = 'mean', normalize = True, all_in_one = True, 
                                plot_CIs = True, print_max_hw = True, color_palette=color_palette)


    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")

if __name__ == "__main__":
    main()