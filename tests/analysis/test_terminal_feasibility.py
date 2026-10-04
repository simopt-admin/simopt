import matplotlib.pyplot as plt
import pytest

from simopt.analysis.terminal_feasibility import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from tests.utils import assert_logs_match, capture_log_data


@pytest.mark.parametrize("experiment", ["SAN2_FCSA.pickle.zst"], indirect=True)
@pytest.mark.parametrize("plot_conf_ints", [False, True])
@pytest.mark.parametrize("bias_correction", [False, True])
def test_terminal_feasibility(experiment, plot_conf_ints, bias_correction, plot_baseline):
    crn_options = CrnOptions(
        across_budget=experiment.crn_across_budget,
        across_macroreps=experiment.crn_across_macroreps,
        across_x0_xstar=experiment.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
        bias_correction=bias_correction,
    )

    desired = plot_baseline["logs"]

    analysis_input = convert(experiment)
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


def test_terminal_feasibility_same_problem_experiments(stochastic_plot_experiments, plot_baseline):
    san2_experiments = [
        experiment
        for group in zip(*stochastic_plot_experiments, strict=True)
        for experiment in group
    ]
    reference = san2_experiments[0]
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

    analysis_inputs = [convert(experiment) for experiment in san2_experiments]
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
    plot_baseline,
):
    reference = different_problem_experiments_stochastic_constraints[0]
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

    analysis_inputs = [
        convert(experiment) for experiment in different_problem_experiments_stochastic_constraints
    ]
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
        different_problem_experiments_stochastic_constraints,
    )

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired)
