import matplotlib.pyplot as plt

from simopt.analysis.terminal_scatterplot import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from tests.utils import assert_logs_match, capture_log_data


def test_terminal_scatterplot(experiment, plot_baseline):
    desired = plot_baseline["logs"]

    analysis_input = convert(experiment)
    analysis_result = analyze(
        analysis_input,
        normalize=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(plot, ax, analysis_result, color="C0", marker="o")

    assert_logs_match(actual, desired)


def test_terminal_scatterplot_same_problem_experiments(plot_experiments, plot_baseline):
    same_problem_experiments = [experiment for row in plot_experiments for experiment in row]
    desired = plot_baseline["logs"]

    analysis_inputs = [convert(experiment) for experiment in same_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == len(same_problem_experiments)


def test_terminal_scatterplot_different_problem_experiments(
    different_problem_experiments,
    plot_baseline,
):
    desired = plot_baseline["logs"]

    analysis_inputs = [convert(experiment) for experiment in different_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, different_problem_experiments)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired)
