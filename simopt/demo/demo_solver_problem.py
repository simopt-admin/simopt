import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
# os.chdir('../')

from wrapper_base import Experiment, read_experiment_results

solver_name = "RNDSRCH" # random search solver
problem_name = "CNTNEWS-1"
myexperiment = Experiment(solver_name, problem_name, solver_fixed_factors={"sample_size": 50})
#print(myexperiment.problem.check_problem_factor("initial_solution"))
myexperiment.run(n_macroreps=10)
#print("Here")
#file_name_path = "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
#myexperiment = read_experiment_results(file_name_path)
myexperiment.post_replicate(n_postreps=200, crn_across_budget=True, crn_across_macroreps=False)
#print("Now here.")
# myexperiment.plot_progress_curves(plot_type="all", normalize=False)
# myexperiment.plot_progress_curves(plot_type="all", normalize=True)
# #print("Finally here.")
#myexperiment.plot_progress_curves(plot_type="mean", normalize=True, plot_CIs=True)
# # myexperiment.plot_progress_curves(plot_type="quantile", normalize=True)
#myexperiment.plot_solvability_curves(solve_tols=[0.2])