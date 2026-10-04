import matplotlib.pyplot as plt
import pytest

from simopt.analysis.progress_curve import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from tests.utils import assert_logs_match, capture_log_data


@pytest.mark.parametrize(
    ("agg", "beta"), [("mean", 0.5), ("quantile", 0.1), ("quantile", 0.5), ("quantile", 0.9)]
)
@pytest.mark.parametrize("normalize", [False, True])
def test_progress_curve(experiment, agg, beta, normalize, plot_baseline):
    crn_options = CrnOptions(
        across_budget=experiment.crn_across_budget,
        across_macroreps=experiment.crn_across_macroreps,
        across_x0_xstar=experiment.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=100,
        confidence_level=0.95,
        bias_correction=True,
    )

    desired = plot_baseline["logs"]

    analysis_input = convert(experiment)
    analysis_result = analyze(
        analysis_input,
        agg,
        ci_options,
        crn_options,
        normalize=normalize,
        beta=beta,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(plot, ax, analysis_result)

    assert_logs_match(actual, desired)


def test_progress_curve_same_problem_experiments(plot_experiments, plot_baseline):
    same_problem_experiments = [
        experiment for group in zip(*plot_experiments, strict=True) for experiment in group
    ]
    reference = same_problem_experiments[0]

    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
        bias_correction=True,
    )

    desired = plot_baseline["logs"]

    analysis_inputs = [convert(experiment) for experiment in same_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        "mean",
        ci_options,
        crn_options,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == len(same_problem_experiments)
