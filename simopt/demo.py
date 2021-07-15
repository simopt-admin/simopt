from wrapper_base import Experiment, post_normalize, plot_progress_curves, plot_solvability_cdfs
from rng.mrg32k3a import MRG32k3a

exp1 = Experiment("RNDSRCH", "CNTNEWS-1")
exp1.run(10)
exp1.post_replicate(100)

exp2 = Experiment("ASTRODF", "CNTNEWS-1")
exp2.run(10)
exp2.post_replicate(100)

post_normalize([exp1, exp2], 200)
#bootstrap_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
#bootstrap_curves = bootstrap_sample([[exp1, exp2]], bootstrap_rng)

#plot_progress_curves([exp1, exp2], plot_type="all", all_in_one=False)
#plot_progress_curves([exp1, exp2], plot_type="mean", all_in_one=False)
#plot_progress_curves([exp1, exp2], plot_type="quantile", beta=0.9, all_in_one=False)
#plot_progress_curves([exp1, exp2], plot_type="mean", normalize=False, all_in_one=False)
#plot_progress_curves([exp1, exp2], plot_type="mean", normalize=True, all_in_one=True, print_max_hw=True)
#plot_progress_curves([exp1, exp2], plot_type="quantile", beta=0.9, normalize=True, all_in_one=True, print_max_hw=True)

plot_solvability_cdfs([exp1, exp2], solve_tol=0.2, all_in_one=True, plot_CIs=True, print_max_hw=True)
plot_solvability_cdfs([exp1, exp2], solve_tol=0.2, all_in_one=False, plot_CIs=True, print_max_hw=True)

