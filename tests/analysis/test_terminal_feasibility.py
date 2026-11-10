import matplotlib.pyplot as plt
import pytest

from simopt.analysis.terminal_feasibility import analyze, analyze_many, plot, plot_many
from simopt.options import ConfidenceIntervalOptions
from tests.utils import PICKLE_CRN_OPTIONS, assert_logs_match, capture_log_data, split_pairs


@pytest.mark.parametrize("experiment_pair", ["SAN2_FCSA.pickle.zst"], indirect=True)
@pytest.mark.parametrize("plot_conf_ints", [False, True])
@pytest.mark.parametrize("bias_correction", [False, True])
def test_terminal_feasibility(experiment_pair, plot_conf_ints, bias_correction, plot_baseline):
    crn_options = PICKLE_CRN_OPTIONS
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
        bias_correction=bias_correction,
    )

    desired = plot_baseline["logs"]

    _, analysis_input = experiment_pair
    analysis_result = analyze(
        analysis_input,
        ci_options if plot_conf_ints else None,
        crn_options,
        score_type="norm",
        norm_degree=2,
        two_sided=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(plot, ax, analysis_result, "C0", "o")

    assert_logs_match(actual, desired)


def test_terminal_feasibility_same_problem_experiments(
    stochastic_plot_experiments, run_crn_options, plot_baseline
):
    san2_experiments, analysis_inputs = split_pairs(
        [pair for group in zip(*stochastic_plot_experiments, strict=True) for pair in group]
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
        ci_options,
        crn_options,
        score_type="norm",
        norm_degree=2,
        two_sided=True,
    )

    actual = capture_log_data(plot_many, analysis_results, san2_experiments)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == len(san2_experiments)


def test_terminal_feasibility_different_problem_experiments(
    different_problem_experiments_stochastic_constraints,
    run_crn_options,
    plot_baseline,
):
    crn_options = run_crn_options
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
        bias_correction=True,
    )

    desired = plot_baseline["logs"]

    experiments, analysis_inputs = split_pairs(different_problem_experiments_stochastic_constraints)
    analysis_results = analyze_many(
        analysis_inputs,
        ci_options,
        crn_options,
        score_type="norm",
        norm_degree=2,
        two_sided=True,
    )

    actual = capture_log_data(
        plot_many,
        analysis_results,
        experiments,
    )

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired)
