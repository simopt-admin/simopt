from wrapper_base import Experiment, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles
from rng.mrg32k3a import MRG32k3a

# exp1 = Experiment("RNDSRCH", "CNTNEWS-1", solver_rename="rndsrch10", problem_rename="cnt5", solver_fixed_factors={"sample_size":10}, oracle_fixed_factors={"purchase_price":5.0})
# exp1.run(10)
# exp1.post_replicate(100)

# exp2 = Experiment("RNDSRCH", "CNTNEWS-1", solver_rename="rndsrch20", problem_rename="cnt5", solver_fixed_factors={"sample_size":20}, oracle_fixed_factors={"purchase_price":5.0})
# exp2.run(10)
# exp2.post_replicate(100)

# exp3 = Experiment("RNDSRCH", "CNTNEWS-1", solver_rename="rndsrch10", problem_rename="cnt7", solver_fixed_factors={"sample_size":10}, oracle_fixed_factors={"purchase_price":7.0})
# exp3.run(10)
# exp3.post_replicate(100)

# exp4 = Experiment("RNDSRCH", "CNTNEWS-1", solver_rename="rndsrch20", problem_rename="cnt7", solver_fixed_factors={"sample_size":20}, oracle_fixed_factors={"purchase_price":7.0})
# exp4.run(10)
# exp4.post_replicate(100)

# exp5 = Experiment("RNDSRCH", "CNTNEWS-1", solver_rename="rndsrch30", problem_rename="cnt5", solver_fixed_factors={"sample_size":30}, oracle_fixed_factors={"purchase_price":5.0})
# exp5.run(10)
# exp5.post_replicate(100)

# exp6 = Experiment("RNDSRCH", "CNTNEWS-1", solver_rename="rndsrch30", problem_rename="cnt7", solver_fixed_factors={"sample_size":30}, oracle_fixed_factors={"purchase_price":7.0})
# exp6.run(10)
# exp6.post_replicate(100)


#post_normalize([exp1, exp2, exp5], 200)
#post_normalize([exp3, exp4, exp6], 200)
#bootstrap_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
#bootstrap_curves = bootstrap_sample([[exp1, exp2]], bootstrap_rng)

#plot_progress_curves([exp1, exp2], plot_type="all", all_in_one=False)
#plot_progress_curves([exp1, exp2], plot_type="mean", all_in_one=False)
#plot_progress_curves([exp1, exp2], plot_type="quantile", beta=0.9, all_in_one=False)
#plot_progress_curves([exp1, exp2], plot_type="mean", normalize=False, all_in_one=False)
#plot_progress_curves([exp1, exp2], plot_type="mean", normalize=True, all_in_one=True, print_max_hw=True)
#plot_progress_curves([exp1, exp2], plot_type="quantile", beta=0.9, normalize=True, all_in_one=True, print_max_hw=True)

#plot_solvability_cdfs([exp1, exp2], solve_tol=0.2, all_in_one=True, plot_CIs=True, print_max_hw=True)
#plot_solvability_cdfs([exp1, exp2], solve_tol=0.2, all_in_one=False, plot_CIs=True, print_max_hw=True)

exp1 = read_experiment_results("experiments/outputs/rndsrch10_on_cnt5.pickle")
exp2 = read_experiment_results("experiments/outputs/rndsrch20_on_cnt5.pickle")
exp3 = read_experiment_results("experiments/outputs/rndsrch10_on_cnt7.pickle")
exp4 = read_experiment_results("experiments/outputs/rndsrch20_on_cnt7.pickle")
exp5 = read_experiment_results("experiments/outputs/rndsrch30_on_cnt5.pickle")
exp6 = read_experiment_results("experiments/outputs/rndsrch30_on_cnt7.pickle")

#plot_area_scatterplots([[exp1, exp3], [exp2, exp4]], all_in_one=True, plot_CIs=True)
#plot_area_scatterplots([[exp1, exp3], [exp2, exp4]], all_in_one=False, plot_CIs=False)

plot_solvability_profiles([[exp1, exp3], [exp2, exp4], [exp5, exp6]], plot_type="cdf_solvability", all_in_one=True, plot_CIS=True, print_max_hw=True, solve_tol=0.1)
plot_solvability_profiles([[exp1, exp3], [exp2, exp4], [exp5, exp6]], plot_type="quantile_solvability", all_in_one=True, plot_CIS=True, print_max_hw=True, solve_tol=0.1, beta=0.5)
plot_solvability_profiles([[exp1, exp3], [exp2, exp4], [exp5, exp6]], plot_type="diff_cdf_solvability", all_in_one=True, plot_CIS=True, print_max_hw=True, solve_tol=0.1, ref_solver="rndsrch10")
plot_solvability_profiles([[exp1, exp3], [exp2, exp4], [exp5, exp6]], plot_type="diff_quantile_solvability", all_in_one=True, plot_CIS=True, print_max_hw=True, solve_tol=0.1, beta=0.5, ref_solver="rndsrch10")

