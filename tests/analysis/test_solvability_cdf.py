import matplotlib.pyplot as plt

from simopt.analysis.solvability_cdf import analyze, analyze_many, plot, plot_many
from simopt.options import ConfidenceIntervalOptions
from tests.utils import PICKLE_CRN_OPTIONS, assert_logs_match, capture_log_data, split_pairs


def test_solvability_cdf(experiment_pair, plot_baseline):
    solve_tolerance = 0.1

    crn_options = PICKLE_CRN_OPTIONS
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
    )

    desired = plot_baseline["logs"]

    _, analysis_input = experiment_pair
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


def test_solvability_cdf_same_problem_experiments(plot_experiments, run_crn_options, plot_baseline):
    same_problem_experiments, analysis_inputs = split_pairs(
        [pair for group in zip(*plot_experiments, strict=True) for pair in group]
    )
    solve_tolerance = 0.1

    crn_options = run_crn_options
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
    )

    desired = plot_baseline["logs"]

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
