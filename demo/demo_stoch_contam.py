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
    plot_feasibility_scatterplots

)

from simopt.models.san import SANLongestPathStochastic, SAN
from simopt.solvers.randomsearch import RandomSearch


def main() -> None:
    
    constraint_nodes = [2,5]
    
    max_length_to_node = [2, 2]
    
    fixed_factors = {"constraint_nodes": constraint_nodes, "max_length_to_node": max_length_to_node}
    
    solver = RandomSearch()

    problem =  SANLongestPathStochastic(fixed_factors=fixed_factors)   
    
    
    
    # -----------------------------------------------

    print(f"Testing solver {solver.name} on problem {problem.name}.")

    # Specify file path name for storing experiment outputs in .pickle file.


    # Initialize an instance of the experiment class.
    myexperiment = ProblemSolver(solver=solver, problem=problem)

    # Run a fixed number of macroreplications of the solver on the problem.
    myexperiment.run(n_macroreps=10)

    # If the solver runs have already been performed, uncomment the
    # following pair of lines (and uncommmen the myexperiment.run(...)
    # line above) to read in results from a .pickle file.
    # myexperiment = read_experiment_results(file_name_path)

    print("Post-processing results.")
    # Run a fixed number of postreplications at all recommended solutions.
    myexperiment.post_replicate(n_postreps=200)
    # Find an optimal solution x* for normalization.
    post_normalize([myexperiment], n_postreps_init_opt=200)
    
 

    print(myexperiment.compute_feasibility_score("inf_norm")) 
    #print(myexperiment.all_intermediate_budgets)
    #print(myexperiment.all_stoch_constraints[0])

    # Log results.
    myexperiment.log_experiment_results()
    
    # plot feasibility scores
    plot_feasibility_scatterplots([[myexperiment]], "feasibility")

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
