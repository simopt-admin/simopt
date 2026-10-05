import matplotlib.pyplot as plt
import pytest

from simopt.analysis.feasibility_progress import analyze, analyze_many, plot, plot_many
from simopt.options import ConfidenceIntervalOptions
from tests.utils import PICKLE_CRN_OPTIONS, assert_logs_match, capture_log_data, split_pairs


@pytest.mark.parametrize("experiment_pair", ["SAN2_FCSA.pickle.zst"], indirect=True)
@pytest.mark.parametrize(
    ("agg", "beta"), [("mean", 0.5), ("quantile", 0.1), ("quantile", 0.5), ("quantile", 0.9)]
)
@pytest.mark.parametrize("score_type", ["inf_norm", "norm"])
def test_feasibility_progress(experiment_pair, agg, beta, score_type, plot_baseline):
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=5,
        confidence_level=0.95,
        bias_correction=True,
    )
    crn_options = PICKLE_CRN_OPTIONS

    desired = plot_baseline["logs"]

    _, analysis_input = experiment_pair
    analysis_result = analyze(
        analysis_input,
        agg,
        ci_options,
        crn_options,
        two_sided=False,
        score_type=score_type,
        beta=beta,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(
        plot,
        ax,
        analysis_result,
    )

    assert_logs_match(actual, desired)


def test_feasibility_progress_plot_many(
    stochastic_plot_experiments, run_crn_options, plot_baseline
):
    san2_experiments, analysis_inputs = split_pairs(
        [pair for group in zip(*stochastic_plot_experiments, strict=True) for pair in group]
    )

    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=5,
        confidence_level=0.95,
        bias_correction=True,
    )
    crn_options = run_crn_options

    desired = plot_baseline["logs"]

    analysis_results = analyze_many(
        analysis_inputs,
        "mean",
        ci_options,
        crn_options,
        two_sided=False,
    )

    actual = capture_log_data(
        plot_many,
        analysis_results,
        san2_experiments,
        plot_zero=False,
    )

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == len(san2_experiments)


def test_feasibility_progress_different_problem_experiments(
    different_problem_experiments_stochastic_constraints,
    run_crn_options,
    plot_baseline,
):
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=5,
        confidence_level=0.95,
        bias_correction=True,
    )
    crn_options = run_crn_options

    desired = plot_baseline["logs"]

    experiments, analysis_inputs = split_pairs(different_problem_experiments_stochastic_constraints)
    analysis_results = analyze_many(
        analysis_inputs,
        "mean",
        ci_options,
        crn_options,
        two_sided=False,
    )

    actual = capture_log_data(
        plot_many,
        analysis_results,
        experiments,
        plot_zero=False,
    )

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired)
