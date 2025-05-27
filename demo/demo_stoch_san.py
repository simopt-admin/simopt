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
from simopt.solvers.csa_lp import CSA_LP
from simopt.solvers.csa import CSA

def main() -> None:



    constraint_nodes = [6, 8]
    
    max_length_to_node = [5, 5]
    
    initial = (5,)*13
    
    fixed_factors = {"constraint_nodes": constraint_nodes, "length_to_node_constraint": max_length_to_node, "initial_solution": initial, "budget": 20000}
    
    csa = CSA(fixed_factors={"crn_across_solns":False})
    csa_lp = CSA_LP( fixed_factors={"ratio":.8, "tolerance": .1})
    csa_lp_1 = CSA_LP(name = "tol_.1", fixed_factors={"ratio":.8, "tolerance": .1})
    solvers = [ csa_lp, RandomSearch()]
    
    problem =  SANLongestPathStochastic(fixed_factors=fixed_factors)   
    #problem.upper_bounds = 13*(100,)
    #problem.lower_bounds = 13*(0.1,)
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
    myexperiment.post_replicate(n_postreps=100)
    # Find an optimal solution x* for normalization.
    myexperiment.post_normalize(100, ignore_feasibility=True)
    
    exp_plot_list = []
    #concat experiment lists
    for exp in myexperiment.experiments:
        exp_plot_list = exp_plot_list + exp
     
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
    #plot_progress_curves(exp_plot_list , "all", normalize=False, all_in_one=True, print_max_hw=False, plot_optimal=False, save_as_pickle=True)
    plot_progress_curves(exp_plot_list , "mean", normalize=False, all_in_one=True, print_max_hw=False, plot_optimal=False, save_as_pickle=True)
    #plot_progress_curves(exp_plot_list , "quantile", normalize=False, all_in_one=True, print_max_hw=False, plot_optimal=False, save_as_pickle=True)
    #plot_feasibility(myexperiment.experiments,  all_in_one=True, two_sided=True, plot_optimal=False, solver_set_name="RNDSRCH")
    #plot_feasibility(myexperiment.experiments, "contour",  all_in_one=True, color_fill = False, two_sided=True, plot_conf_ints=False, save_as_pickle=True)
    #plot_feasibility(myexperiment.experiments, "contour",  all_in_one=False, color_fill = False, two_sided=True, plot_conf_ints=False)
    #plot_feasibility(myexperiment.experiments, "contour",  all_in_one=False, color_fill = True, two_sided=True, plot_conf_ints=False)
    #plot_feasibility(myexperiment.experiments, "scatter",  all_in_one=True, two_sided=True, plot_optimal=True,save_as_pickle=True) 
    #plot_feasibility(myexperiment.experiments, "violin",  all_in_one=True, two_sided=True,save_as_pickle=True)
    #plot_feasibility(myexperiment.experiments, "violin",score_type= "inf_norm", norm_degree=2,  all_in_one=False)
    #plot_feasibility_progress(myexperiment.experiments,plot_type =  "all", score_type = "inf_norm", all_in_one=True, two_sided=True, save_as_pickle=True)
    #plot_feasibility_progress(myexperiment.experiments,plot_type =  "mean", score_type = "inf_norm", all_in_one=True, two_sided=True, save_as_pickle=True)
    #plot_feasibility_progress(myexperiment.experiments,plot_type =  "quantile", score_type = "inf_norm", all_in_one=True, two_sided=True, save_as_pickle=True)
    #plot_terminal_progress(exp_plot_list, plot_type = 'violin',  normalize=False, plot_optimal=True, save_as_pickle=True)
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
    #print("Finished. Plots have been displayed and saved in experiments/plots folder.")




if __name__ == "__main__":
    main()
