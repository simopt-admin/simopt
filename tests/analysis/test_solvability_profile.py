import matplotlib.pyplot as plt
import pytest

from simopt.analysis.solvability_profile import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from tests.utils import assert_logs_match, capture_log_data


@pytest.mark.parametrize(
    ("plot_type", "beta"), [("cdf", 0.5), ("quantile", 0.1), ("quantile", 0.5), ("quantile", 0.9)]
)
def test_solvability_profile(experiment, plot_type, beta, plot_baseline):
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

    desired = plot_baseline["logs"]

    analysis_input = convert(experiment)
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
def test_solvability_profile_same_problem_experiments(plot_experiments, plot_type, plot_baseline):
    same_problem_experiments = [experiment for row in plot_experiments for experiment in row]
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

    desired = plot_baseline["logs"]

    analysis_inputs = [convert(experiment) for experiment in same_problem_experiments]
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
    plot_baseline,
):
    solve_tolerance = 0.1
    reference = different_problem_experiments[0]
    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
    )

    desired = plot_baseline["logs"]

    analysis_inputs = [convert(experiment) for experiment in different_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        different_problem_experiments,
        plot_type,
        ci_options,
        crn_options,
        solve_tolerance=solve_tolerance,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, different_problem_experiments, plot_type)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired)
