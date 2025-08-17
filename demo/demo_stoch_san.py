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

from simopt.experiment_base import post_normalize as post_normalize_problem
from simopt.models.san import SANLongestPathStochastic, SAN, SANLongestPath
from simopt.solvers.randomsearch import RandomSearch
from simopt.solvers.csa_lp import CSA_LP as csa_v2
from simopt.solvers.csa import CSA as CSA_recommended
from simopt.solvers.csa_lp_old import CSA_LP as csa_v1
from simopt.solvers.csa_lp_v0 import CSA_LP as csa_v0
from simopt.solvers.csa_lp_v0_all_sol import CSA_LP as csa_v0_all
from simopt.solvers.csa_lp_v1_all_sol import CSA_LP as csa_v1_all
from simopt.solvers.csa_lp_all_sol import CSA_LP as csa_v2_all
from simopt.solvers.csa_lp_v1a_all_sol import CSA_LP as csa_v1a_all
from simopt.solvers.csa_lp_v1a import CSA_LP as csa_v1a_recommended
from simopt.solvers.csa_lp_v1b_all_sol import CSA_LP as csa_v1b_all
from simopt.solvers.csa_lp_v2a_all_sol import CSA_LP as csa_v2a_all
from simopt.solvers.csa_lp__v2b_all_sol import CSA_LP as csa_v2b_all
from simopt.solvers.csa_lp_v3a_all_sol import CSA_LP as csa_v3a_all
from simopt.solvers.csa_lp_v4_all_sol import CSA_LP as csa_v4a_all
from simopt.solvers.csa_all_solns import CSA
from simopt.solvers.csa_all_solns_unnormalized import CSA as CSA_unnormalized
from simopt.solvers.csa_lp_v2a import CSA_LP as csa_v2a_recommended
from simopt.solvers.csa_normal import CSA as csa_normal_reccomended
from simopt.solvers.fcsa import CSA_LP as fcsa_recommended
from simopt.solvers.csa_unnormalized import CSA as csa_unnormalized
from simopt.solvers.fcsa_merge import FCSA as fcsa_merge



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
    initial = (5,)*13
    #
    # fixed_factors = {"constraint_nodes": constraint_nodes, "length_to_node_constraint": max_length_to_node, "initial_solution": initial, "budget": 15000}
    # prob_1_const = SANLongestPathStochastic(fixed_factors=fixed_factors, name="SAN_1")

    constraint_nodes = [6, 8]

    max_length_to_node = [5, 5]

    fixed_factors = {"constraint_nodes": constraint_nodes, "length_to_node_constraint": max_length_to_node,
                     "initial_solution": initial, "budget": 10000}
    prob_2_const = SANLongestPathStochastic(fixed_factors=fixed_factors)

    no_CRN = {"crn_across_solns": False}
    rand = RandomSearch(fixed_factors={"sample_size":int(5)}, name="RandomSearch")
    v0 = csa_v0(name="csa_lp_v0")
    v1 = csa_v1(name="csa_lp_v1") 
    v2 = csa_v2(name="csa_lp_v2")
    v0_all = csa_v0_all(name="v0")
    v1_all = csa_v1_all(name="csa_lp_v1")
    v2_all = csa_v2_all(name="csa_lp_v2")
    v1a = csa_v1a_all(name="CSA-MC")
    v1a_recommended = csa_v1a_recommended(name="CSA-MC")
    v1b = csa_v1b_all(name="c")
    v2a = csa_v2a_all(name="FCSA")
    v2b = csa_v2b_all(name="v2b")
    v3a = csa_v3a_all(name="csa_v3")
    v4 = csa_v4a_all(name="FCSA-0.1")
    csa_normal = CSA(name="CSA-N") #csa with no maximization problem, only follows most violated constraint
    csa = CSA_unnormalized(name="CSA")
    csa_recommended = CSA_recommended(name="CSA")
    v2a_recommended = csa_v2a_recommended(name="csa_v2")
    fcsa = fcsa_recommended(name="FCSA")
    csa_n_reccomended = csa_normal_reccomended(name="CSA-N")
    fcsa_1 = fcsa_recommended(name="FCSA-1", fixed_factors={"feas_const": 1.0})
    fcsa_10 = fcsa_recommended(name="FCSA-10", fixed_factors={"feas_const": 10.0})
    
    def step_f( k: int) -> float:
        """
        take in the current iteration k
        """
        return .2
    csa_test = fcsa_merge(fixed_factors={"search_direction": "CSA"}, name="CSA")
    csa_lp_test = fcsa_merge(fixed_factors={"search_direction": "CSA-LP"}, name="CSA-LP")
    fcsa_test = fcsa_merge(fixed_factors={"search_direction": "FCSA", "feas_const": float(1), "feas_score": 2}, name="FCSA")
    #test = fcsa_merge(fixed_factors=fixed_factors)

    #adjust step sizes


    
    solvers = [ csa_test, fcsa_test]
    
    problems = [prob_2_const]
    #problem.upper_bounds = 13*(100,)
    #problem.lower_bounds = 13*(0.1,)
    #problem = SANLongestPath() 
    
    
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
    myexperiment.post_normalize(100)
    
    exp_plot_list = []
    #concat experiment lists
    for exp in myexperiment.experiments:
        exp_plot_list = exp_plot_list + exp
    #post_normalize_problem(exp_plot_list, 100, proxy_opt_x=(20,)*13)
    #myexperiment.log_group_experiment_results()
    #myexperiment.report_group_statistics()
    #print(myexperiment.all_intermediate_budgets)
    #print(myexperiment.all_stoch_constraints[0])

    # plot feasibility scores
    #myexperiment.experiments[1][0].compute_feasibility_score()
    #print(len(myexperiment.experiments[1][0].all_intermediate_budgets[0]) == len(myexperiment.experiments[1][0].compute_feasibility_score()[0]))

    #plot_terminal_progress(experiments=myexperiment.experiments)
    #print(myexperiment.experiments[1][0].feasibility_curves)
    #plot_feasibility(myexperiment.experiments, plot_type= "scatter", plot_conf_ints=True, save_as_pickle=True, plot_optimal=False)
    #plt.show()
    #plot_feasibility(myexperiment.experiments, plot_type= "scatter", plot_conf_ints=False, save_as_pickle=True,plot_optimal=False)
    #plot_feasibility(myexperiment.experiments, plot_type="contour")
    #plot_progress_curves(exp_plot_list , "all", normalize=False, all_in_one=True, print_max_hw=False, plot_optimal=True, save_as_pickle=True)
    #plt.show()
    # plot_progress_curves([myexperiment.experiments[0][0]], "all", normalize=False, all_in_one=True, print_max_hw=False, plot_optimal=False,
    #                      save_as_pickle=True)
    # plt.show()
    # plot_progress_curves([myexperiment.experiments[0][1]], "all", normalize=False, all_in_one=True, print_max_hw=False,
    #                      plot_optimal=False,
    #                      save_as_pickle=True)
    # plt.show()
    #plot_progress_curves(exp_plot_list , "mean", normalize=False, all_in_one=True, print_max_hw=False, plot_optimal=True, save_as_pickle=True)
    #plt.show()
    #plot_progress_curves(exp_plot_list , "quantile", normalize=False, all_in_one=True, print_max_hw=False, plot_optimal=False, save_as_pickle=True)
    #plot_feasibility(myexperiment.experiments,  all_in_one=True, two_sided=True, plot_optimal=False, solver_set_name="RNDSRCH")
    #plot_feasibility(myexperiment.experiments, "contour",  all_in_one=True, color_fill = True, two_sided=True, plot_conf_ints=False, save_as_pickle=True)
    #plot_feasibility(myexperiment.experiments, "contour",  all_in_one=False, color_fill = False, two_sided=True, plot_conf_ints=False)
    #plot_feasibility(myexperiment.experiments, "contour",  all_in_one=False, color_fill = True, two_sided=True, plot_conf_ints=False)
    #plot_feasibility(myexperiment.experiments, "scatter",  all_in_one=True, two_sided=True, plot_optimal=True,save_as_pickle=True)
    #plot_feasibility(myexperiment.experiments, "violin",  all_in_one=True, two_sided=True,save_as_pickle=True)
    #plt.show()
    #plot_feasibility(myexperiment.experiments, "violin",score_type= "inf_norm", norm_degree=2,  all_in_one=False)
    # plot_feasibility_progress(myexperiment.experiments,plot_type =  "all", score_type = "inf_norm", all_in_one=True, two_sided=True, print_max_hw = False, save_as_pickle=True)
    # plt.show()
    # plot_feasibility_progress(myexperiment.experiments,plot_type =  "all", score_type = "inf_norm", all_in_one=False, two_sided=True, print_max_hw = False, save_as_pickle=True)
    # plt.show()
    #plot_feasibility_progress(myexperiment.experiments,plot_type =  "mean", score_type = "norm", norm_degree=2, all_in_one=True, two_sided=False, print_max_hw = False, save_as_pickle=True)
    #plt.show()
    # plot_feasibility_progress(myexperiment.experiments,plot_type =  "quantile", beta = .9, score_type = "inf_norm", all_in_one=True, two_sided=False, print_max_hw = False, save_as_pickle=True)
    # plt.show()
    plot_feasibility_progress(myexperiment.experiments,plot_type =  "quantile", beta = .9, score_type = "norm", all_in_one=False, two_sided=False, print_max_hw = False, save_as_pickle=True)
    plt.show()
    #plot_feasibility_progress(myexperiment.experiments,plot_type =  "mean", score_type = "inf_norm", all_in_one=True, two_sided=False, save_as_pickle=True)
    #plt.show()
    #plot_feasibility_progress(myexperiment.experiments,plot_type =  "quantile", score_type = "inf_norm", all_in_one=True, two_sided=True, save_as_pickle=True)
    #plot_terminal_progress(exp_plot_list, plot_type = 'violin',  normalize=False, plot_optimal=True, save_as_pickle=True, )
    #.show()
    # plot_terminal_progress([myexperiment.experiments[0][0]], plot_type='violin', normalize=False, plot_optimal=True, save_as_pickle=True)
    # plt.show()
    # plot_terminal_progress([myexperiment.experiments[0][1]], plot_type='violin', normalize=False, plot_optimal=True,
    #                        save_as_pickle=True)
    # plt.show()
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
