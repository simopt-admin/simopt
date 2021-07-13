from wrapper_base import Experiment, post_normalize, plot_progress_curves

exp1 = Experiment("RNDSRCH", "CNTNEWS-1")
exp1.run(10)
exp1.post_replicate(20)

exp2 = Experiment("ASTRODF", "CNTNEWS-1")
exp2.run(10)
exp2.post_replicate(20)

post_normalize([exp1, exp2], 50)
plot_progress_curves([exp1, exp2], plot_type="all", all_in_one=False)
plot_progress_curves([exp1, exp2], plot_type="mean", all_in_one=False)
plot_progress_curves([exp1, exp2], plot_type="quantile", beta=0.9, all_in_one=False)
plot_progress_curves([exp1, exp2], plot_type="mean", normalize=False, all_in_one=False)
