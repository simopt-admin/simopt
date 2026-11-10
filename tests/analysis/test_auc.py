import matplotlib.pyplot as plt

from simopt.analysis.auc import analyze, analyze_many, plot, plot_many
from simopt.options import ConfidenceIntervalOptions
from tests.utils import PICKLE_CRN_OPTIONS, assert_logs_match, capture_log_data, split_pairs


def test_auc(experiment_pair, plot_baseline):
    crn_options = PICKLE_CRN_OPTIONS
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.9,
        bias_correction=True,
    )
    desired = plot_baseline["logs"]

    _, analysis_input = experiment_pair
    analysis_result = analyze(
        analysis_input,
        ci_options,
        crn_options,
        normalize=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(plot, ax, analysis_result, "C0", "o")

    assert_logs_match(actual, desired)


def test_auc_same_problem_experiments(plot_experiments, run_crn_options, plot_baseline):
    same_problem_experiments, analysis_inputs = split_pairs(
        [pair for row in plot_experiments for pair in row]
    )

    crn_options = run_crn_options
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.9,
        bias_correction=True,
    )
    desired = plot_baseline["logs"]

    analysis_results = analyze_many(
        analysis_inputs,
        ci_options,
        crn_options,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == len(same_problem_experiments)


def test_auc_different_problem_experiments(
    different_problem_experiments, run_crn_options, plot_baseline
):
    crn_options = run_crn_options
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.9,
        bias_correction=True,
    )
    desired = plot_baseline["logs"]

    experiments, analysis_inputs = split_pairs(different_problem_experiments)
    analysis_results = analyze_many(
        analysis_inputs,
        ci_options,
        crn_options,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, experiments)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired)
