"""
This script is intended to help with debugging problems and solvers.
It create a problem-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o
import matplotlib.pyplot as plt  # Add matplotlib import

sys.path.append(
    o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import (
    ProblemSolver,
    post_normalize,
    plot_progress_curves,
    plot_solvability_cdfs,
    plot_feasibility,
    ProblemsSolvers,
    plot_terminal_progress,
    plot_feasibility_progress

)

from simopt.models.san import SANLongestPathStochastic, SAN, SANLongestPath
from simopt.solvers.randomsearch import RandomSearch
from simopt.solvers.csa_lp import CSA_LP as csa_v2
from simopt.solvers.csa import CSA
from simopt.solvers.csa_lp_old import CSA_LP as csa_v1
from simopt.solvers.csa_lp_v0 import CSA_LP as csa_v0
from simopt.solvers.csa_lp_v0_all_sol import CSA_LP as csa_v0_all
from simopt.solvers.csa_lp_v1_all_sol import CSA_LP as csa_v1_all
from simopt.solvers.csa_lp_all_sol import CSA_LP as csa_v2_all
from simopt.solvers.csa_lp_v1a_all_sol import CSA_LP as csa_v1a_all
from simopt.solvers.csa_lp_v1b_all_sol import CSA_LP as csa_v1b_all
from simopt.solvers.csa_lp_v2a_all_sol import CSA_LP as csa_v2a_all
from simopt.solvers.csa_lp__v2b_all_sol import CSA_LP as csa_v2b_all
from simopt.solvers.csa_lp_v3a_all_sol import CSA_LP as csa_v3a_all
from simopt.solvers.csa_all_solns import CSA
from simopt.solvers.csa_lp_v2a import CSA_LP as csa_v2a_recommended

"""
csa_v0: original version of csa_lp
csa_v1 : Added gradient to lp, no adjustments to step-size, d, or recomnendation criteria
csa_v2 : Most updated version of csa. d is normalized and all solutions until feasible are recommended. 
        After only feasilible solutions that improve objective are recommended
"""


def main() -> None:
    # constraint_nodes = [6]
    #
    # max_length_to_node = [5]
    #
    initial = (5,) * 13
    #
    # fixed_factors = {"constraint_nodes": constraint_nodes, "length_to_node_constraint": max_length_to_node, "initial_solution": initial, "budget": 15000}
    # prob_1_const = SANLongestPathStochastic(fixed_factors=fixed_factors, name="SAN_1")

    constraint_nodes = [6, 8]

    max_length_to_node = [5, 5]

    fixed_factors = {"constraint_nodes": constraint_nodes, "length_to_node_constraint": max_length_to_node,
                     "initial_solution": initial, "budget": 15000}
    prob_2_const = SANLongestPathStochastic(fixed_factors=fixed_factors, name="SAN")

    # adjust step sizes

    solvers = [RandomSearch()]

    problems = [prob_2_const]
    # problem.upper_bounds = 13*(100,)
    # problem.lower_bounds = 13*(0.1,)
    # problem = SANLongestPath()

    # -----------------------------------------------

    # Specify file path name for storing experiment outputs in .pickle file.

    # Initialize an instance of the experiment class.
    myexperiment = ProblemsSolvers(solvers=solvers, problems=problems)

    # Run a fixed number of macroreplications of the solver on the problem.
    myexperiment.run(n_macroreps=10)

    # If the solver runs have already been performed, uncomment the
    # following pair of lines (and uncommmen the myexperiment.run(...)
    # line above) to read in results from a .pickle file.
    # myexperiment = read_experiment_results(file_name_path)

    # Run a fixed number of postreplications at all recommended solutions.
    myexperiment.post_replicate(n_postreps=100)
    # Find an optimal solution x* for normalization.
    myexperiment.post_normalize(100, ignore_feasibility=False)

    exp_plot_list = []
    # concat experiment lists
    for exp in myexperiment.experiments:
        exp_plot_list = exp_plot_list + exp

    # plot feas & objective violin
    plot_feasibility(experiments=myexperiment.experiments, plot_type="objective_violin")
    plt.show()


if __name__ == "__main__":
    main()
