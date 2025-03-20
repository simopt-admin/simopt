"""
This script is intended to help with debugging problems and solvers.
It create a problem-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o

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
from simopt.solvers.csa_lp import CSA_LP


def main() -> None:
    
    constraint_nodes = [5,7]
    
    max_length_to_node = [5,5]
    
    initial = (5,)*13
    
    fixed_factors = {"constraint_nodes": constraint_nodes, "max_length_to_node": max_length_to_node, "initial_solution": initial}
    
    solvers = [CSA_LP(), RandomSearch()]

    problem =  SANLongestPathStochastic(fixed_factors=fixed_factors)   
    #problem = SANLongestPath()
    
    
    # -----------------------------------------------

    

    # Specify file path name for storing experiment outputs in .pickle file.


    # Initialize an instance of the experiment class.
    myexperiment = ProblemsSolvers(solvers=solvers, problems=[problem])

    # Run a fixed number of macroreplications of the solver on the problem.
    myexperiment.run(n_macroreps=10)

    # If the solver runs have already been performed, uncomment the
    # following pair of lines (and uncommmen the myexperiment.run(...)
    # line above) to read in results from a .pickle file.
    # myexperiment = read_experiment_results(file_name_path)


    # Run a fixed number of postreplications at all recommended solutions.
    myexperiment.post_replicate(n_postreps=200)
    # Find an optimal solution x* for normalization.
    myexperiment.post_normalize(200)
     
    #myexperiment.log_group_experiment_results()
    #myexperiment.report_group_statistics()
    #print(myexperiment.all_intermediate_budgets)
    #print(myexperiment.all_stoch_constraints[0])


    
    # plot feasibility scores
    #myexperiment.experiments[1][0].compute_feasibility_score()
    #print(len(myexperiment.experiments[1][0].all_intermediate_budgets[0]) == len(myexperiment.experiments[1][0].compute_feasibility_score()[0]))

    #plot_terminal_progress(experiments=myexperiment.experiments)
    #print(myexperiment.experiments[1][0].feasibility_curves)
    #plot_feasibility(myexperiment.experiments, plot_type= "scatter")
    #plot_progress_curves(myexperiment.experiments[0] + myexperiment.experiments[1], "all", normalize=False)
    #plot_feasibility(myexperiment.experiments,  all_in_one=True, two_sided=True)
    plot_feasibility(myexperiment.experiments, "contour",  all_in_one=True, color_fill = False, two_sided=True, plot_conf_ints=False)
    plot_feasibility(myexperiment.experiments, "scatter",  all_in_one=True, two_sided=True)
    plot_feasibility(myexperiment.experiments, "violin",  all_in_one=True, two_sided=True)
    #plot_feasibility(myexperiment.experiments, "violin",score_type= "inf_norm", norm_degree=2,  all_in_one=False)
    #plot_feasibility_progress(myexperiment.experiments,plot_type =  "all", score_type = "inf_norm", all_in_one=True)
    #plot_feasibility_progress(myexperiment.experiments, plot_type = "mean",all_in_one=True)
    #plot_feasibility_progress(myexperiment.experiments, plot_type = "quantile", all_in_one=True)

    # print("Plotting results.")
    # # Produce basic plots of the solver on the problem.
    # plot_progress_curves(
    #     experiments=[myexperiment], plot_type="all", normalize=False
    # )
    # plot_progress_curves(
    #     experiments=[myexperiment], plot_type="mean", normalize=False
    # )
    # plot_progress_curves(
    #     experiments=[myexperiment],
    #     plot_type="quantile",
    #     beta=0.90,
    #     normalize=False,
    # )
    # plot_solvability_cdfs(experiments=[myexperiment], solve_tol=0.1)

    # # Plots will be saved in the folder experiments/plots.
    # print("Finished. Plots can be found in experiments/plots folder.")


if __name__ == "__main__":
    main()
