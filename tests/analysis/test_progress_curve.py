import matplotlib.pyplot as plt
import pytest

from simopt.analysis.progress_curve import analyze, analyze_many, plot, plot_many
from simopt.options import ConfidenceIntervalOptions
from tests.utils import PICKLE_CRN_OPTIONS, assert_logs_match, capture_log_data, split_pairs


@pytest.mark.parametrize(
    ("agg", "beta"), [("mean", 0.5), ("quantile", 0.1), ("quantile", 0.5), ("quantile", 0.9)]
)
@pytest.mark.parametrize("normalize", [False, True])
def test_progress_curve(experiment_pair, agg, beta, normalize, plot_baseline):
    crn_options = PICKLE_CRN_OPTIONS
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=100,
        confidence_level=0.95,
        bias_correction=True,
    )

    desired = plot_baseline["logs"]

    _, analysis_input = experiment_pair
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


def test_progress_curve_same_problem_experiments(plot_experiments, run_crn_options, plot_baseline):
    same_problem_experiments, analysis_inputs = split_pairs(
        [pair for group in zip(*plot_experiments, strict=True) for pair in group]
    )

    crn_options = run_crn_options
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
        bias_correction=True,
    )

    desired = plot_baseline["logs"]

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
