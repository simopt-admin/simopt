import matplotlib.pyplot as plt

from simopt.analysis.terminal_scatterplot import analyze, analyze_many, plot, plot_many
from tests.utils import assert_logs_match, capture_log_data, split_pairs


def test_terminal_scatterplot(experiment_pair, plot_baseline):
    desired = plot_baseline["logs"]

    _, analysis_input = experiment_pair
    analysis_result = analyze(
        analysis_input,
        normalize=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(plot, ax, analysis_result, color="C0", marker="o")

    assert_logs_match(actual, desired)


def test_terminal_scatterplot_same_problem_experiments(plot_experiments, plot_baseline):
    same_problem_experiments, analysis_inputs = split_pairs(
        [pair for row in plot_experiments for pair in row]
    )
    desired = plot_baseline["logs"]

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

    experiments, analysis_inputs = split_pairs(different_problem_experiments)
    analysis_results = analyze_many(
        analysis_inputs,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, experiments)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired)
