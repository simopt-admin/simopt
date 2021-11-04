import numpy as np
import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from rng.mrg32k3a import MRG32k3a
from base import Solver, Problem, Model, Solution
from wrapper_base import Experiment, read_experiment_results, MetaExperiment

mymetaexperiment = MetaExperiment(solver_names=["RNDSRCH"], problem_names=["MM1-1", "CNTNEWS-1", "FACSIZE-1"], fixed_factors_filename="all_factors")
mymetaexperiment.run(n_macroreps=2, crn_across_solns=True)
mymetaexperiment.post_replicate(n_postreps=20, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)
# mymetaexperiment.plot_area_scatterplot(plot_CIs=True, all_in_one=False)
# mymetaexperiment.plot_solvability_profiles(solve_tol=0.1)

# myexperiment = Experiment(solver_name="RNDSRCH", problem_name="MM1-1")
# # myexperiment = Experiment(solver_name="RNDSRCH", problem_name="CNTNEWS-1")
# myexperiment.run(n_macroreps=5, crn_across_solns=True)
# myexperiment.post_replicate(n_postreps=20, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)

# myexperiment3 = read_experiment_results(file_name="RNDSRCH_on_CNTNEWS-1")
# myexperiment.post_replicate(n_postreps=20, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)
# # myexperiment3.compute_area_stats()
# myexperiment3.plot_solvability_curves(solve_tols=[0.1])
# myexperiment3.compute_solvability_quantiles(beta=0.5)
# print(myexperiment3.solve_time_quantiles)
# myexperiment3.plot_progress_curves(plot_type="all")
# myexperiment3.plot_progress_curves(plot_type="mean")
# myexperiment3.plot_progress_curves(plot_type="quantile")
# myexperiment3.plot_progress_curves(plot_type="all", normalize=False)
# myexperiment3.plot_progress_curves(plot_type="mean", normalize=False)
# myexperiment3.plot_progress_curves(plot_type="quantile", normalize=False)

print('I ran this.')
