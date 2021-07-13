import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from wrapper_base import MetaExperiment

print("here")
mymetaexperiment = MetaExperiment(solver_names=["RNDSRCH10", "RNDSRCH20", "RNDSRCH30", "RNDSRCH40", "RNDSRCH50"], problem_names=["CNTNEWS-1", "FACSIZE-2"], fixed_factors_filename="all_factors")
print("now here")
#mymetaexperiment.post_replicate(n_postreps=200, n_postreps_init_opt=200, crn_across_budget=True, crn_across_macroreps=False)
mymetaexperiment.plot_solvability_profiles(solve_tol=0.1, beta=0.5, ref_solver="RNDSRCH30")
#mymetaexperiment.plot_area_scatterplot(plot_CIs=False, all_in_one=False)
#mymetaexperiment.plot_progress_curves(plot_type="quantile", beta=0.90, normalize=True)
#mymetaexperiment.plot_solvability_curves(solve_tols=[0.1, 0.2])