from simopt.analysis.diff import analyze_many, plot_many
from simopt.options import ConfidenceIntervalOptions
from tests.utils import assert_logs_match, capture_log_data, split_pairs


def test_diff_solvability_cdf(plot_experiments, run_crn_options, plot_baseline):
    same_problem_experiments, analysis_inputs = split_pairs(
        [pair for row in plot_experiments for pair in row]
    )
    reference_solver = plot_experiments[-1][0][0].solver.name
    ci_options = ConfidenceIntervalOptions(n_bootstraps=10, confidence_level=0.95)
    crn_options = run_crn_options

    desired = plot_baseline["logs"]

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


def test_diff_solvability_quantile(plot_experiments, run_crn_options, plot_baseline):
    same_problem_experiments, analysis_inputs = split_pairs(
        [pair for row in plot_experiments for pair in row]
    )
    reference_solver = plot_experiments[-1][0][0].solver.name
    ci_options = ConfidenceIntervalOptions(n_bootstraps=10, confidence_level=0.95)
    crn_options = run_crn_options

    desired = plot_baseline["logs"]

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
