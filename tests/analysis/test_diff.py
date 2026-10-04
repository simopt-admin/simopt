from simopt.analysis.diff import analyze_many, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from simopt.plot_type import PlotType
from simopt.plots.solvability_profile import plot_solvability_profiles
from tests.utils import assert_logs_match, capture_log_data


def test_diff_solvability_cdf(plot_experiments):
    same_problem_experiments = [experiment for row in plot_experiments for experiment in row]
    reference_solver = plot_experiments[-1][0].solver.name
    reference = same_problem_experiments[0]
    ci_options = ConfidenceIntervalOptions(n_bootstraps=10, confidence_level=0.95)
    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )

    desired = capture_log_data(
        plot_solvability_profiles,
        experiments=plot_experiments,
        plot_type=PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        solve_tol=0.1,
        beta=0.5,
        ref_solver=reference_solver,
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
        problem_set_name="PROBLEM_SET",
    )

    analysis_inputs = [convert(experiment) for experiment in same_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        same_problem_experiments,
        "cdf",
        reference_solver,
        ci_options,
        crn_options,
        normalize=True,
        solve_tolerance=0.1,
        beta=0.5,
    )
    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert len(analysis_results) == len(plot_experiments) - 1
    if analysis_results:
        assert_logs_match(actual, desired)
    else:
        # A single reference solver has no differences to plot.
        assert actual == desired == []
    assert len(actual) == len(desired) == len(plot_experiments) - 1


def test_diff_solvability_quantile(plot_experiments):
    same_problem_experiments = [experiment for row in plot_experiments for experiment in row]
    reference_solver = plot_experiments[-1][0].solver.name
    reference = same_problem_experiments[0]
    ci_options = ConfidenceIntervalOptions(n_bootstraps=10, confidence_level=0.95)
    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )

    desired = capture_log_data(
        plot_solvability_profiles,
        experiments=plot_experiments,
        plot_type=PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        solve_tol=0.1,
        beta=0.5,
        ref_solver=reference_solver,
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
        problem_set_name="PROBLEM_SET",
    )

    analysis_inputs = [convert(experiment) for experiment in same_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        same_problem_experiments,
        "quantile",
        reference_solver,
        ci_options,
        crn_options,
        normalize=True,
        solve_tolerance=0.1,
        beta=0.5,
    )
    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert len(analysis_results) == len(plot_experiments) - 1
    if analysis_results:
        assert_logs_match(actual, desired)
    else:
        # A single reference solver has no differences to plot.
        assert actual == desired == []
    assert len(actual) == len(desired) == len(plot_experiments) - 1
