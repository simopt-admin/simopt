import matplotlib.pyplot as plt

from simopt.analysis.solvability_cdf import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from simopt.plots.solvability_cdf import plot_solvability_cdfs
from tests.utils import assert_logs_match, capture_log_data


def test_solvability_cdf(experiment):
    solve_tolerance = 0.1

    crn_options = CrnOptions(
        across_budget=experiment.crn_across_budget,
        across_macroreps=experiment.crn_across_macroreps,
        across_x0_xstar=experiment.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
    )

    desired = capture_log_data(
        plot_solvability_cdfs,
        experiments=[experiment],
        solve_tol=solve_tolerance,
        all_in_one=False,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
    )

    analysis_input = convert(experiment)
    analysis_result = analyze(
        analysis_input,
        ci_options,
        crn_options,
        solve_tolerance=solve_tolerance,
        normalize=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(plot, ax, analysis_result)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == 1


def test_solvability_cdf_same_problem_experiments(plot_experiments):
    same_problem_experiments = [
        experiment for group in zip(*plot_experiments, strict=True) for experiment in group
    ]
    solve_tolerance = 0.1
    reference = same_problem_experiments[0]

    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
    )

    desired = []
    for group in zip(*plot_experiments, strict=True):
        desired.extend(
            capture_log_data(
                plot_solvability_cdfs,
                experiments=list(group),
                solve_tol=solve_tolerance,
                all_in_one=False,
                n_bootstraps=ci_options.n_bootstraps,
                conf_level=ci_options.confidence_level,
                plot_conf_ints=True,
                print_max_hw=False,
                plot_title=None,
                legend_loc=None,
                ext=".png",
                save_as_pickle=False,
                solver_set_name="SOLVER_SET",
            )
        )

    analysis_inputs = [convert(experiment) for experiment in same_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        ci_options,
        crn_options,
        solve_tolerance=solve_tolerance,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == len(same_problem_experiments)
