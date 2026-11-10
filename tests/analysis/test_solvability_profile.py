import matplotlib.pyplot as plt
import pytest

from simopt.analysis.solvability_profile import analyze, analyze_many, plot, plot_many
from simopt.options import ConfidenceIntervalOptions
from tests.utils import PICKLE_CRN_OPTIONS, assert_logs_match, capture_log_data, split_pairs


@pytest.mark.parametrize(
    ("plot_type", "beta"), [("cdf", 0.5), ("quantile", 0.1), ("quantile", 0.5), ("quantile", 0.9)]
)
def test_solvability_profile(experiment_pair, plot_type, beta, plot_baseline):
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
        plot_type,
        ci_options,
        crn_options,
        solve_tolerance=solve_tolerance,
        normalize=True,
        beta=beta,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(
        plot,
        ax,
        analysis_result,
    )

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == 1


@pytest.mark.parametrize("plot_type", ["cdf", "quantile"])
def test_solvability_profile_same_problem_experiments(
    plot_experiments, plot_type, run_crn_options, plot_baseline
):
    same_problem_experiments, analysis_inputs = split_pairs(
        [pair for row in plot_experiments for pair in row]
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
        same_problem_experiments,
        plot_type,
        ci_options,
        crn_options,
        solve_tolerance=solve_tolerance,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments, plot_type)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == len(plot_experiments)


@pytest.mark.parametrize("plot_type", ["cdf", "quantile"])
def test_solvability_profile_different_problem_experiments(
    different_problem_experiments,
    plot_type,
    run_crn_options,
    plot_baseline,
):
    solve_tolerance = 0.1
    crn_options = run_crn_options
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
    )

    desired = plot_baseline["logs"]

    experiments, analysis_inputs = split_pairs(different_problem_experiments)
    analysis_results = analyze_many(
        analysis_inputs,
        experiments,
        plot_type,
        ci_options,
        crn_options,
        solve_tolerance=solve_tolerance,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, experiments, plot_type)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired)
