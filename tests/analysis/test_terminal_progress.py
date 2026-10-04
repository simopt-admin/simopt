import matplotlib.pyplot as plt
import pytest

from simopt.analysis.terminal_progress import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from tests.utils import assert_logs_match, capture_log_data


@pytest.mark.parametrize("plot_type", ["box", "violin"])
@pytest.mark.parametrize("normalize", [False, True])
def test_terminal_progress(experiment, plot_type, normalize, plot_baseline):
    desired = plot_baseline["logs"]

    analysis_input = convert(experiment)
    analysis_result = analyze(
        analysis_input,
        normalize=normalize,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(
        plot,
        ax,
        analysis_result,
        plot_type=plot_type,
    )

    assert_logs_match(actual, desired)


def test_terminal_progress_same_problem_experiments(plot_experiments, plot_baseline):
    same_problem_experiments = [
        experiment for group in zip(*plot_experiments, strict=True) for experiment in group
    ]
    desired = plot_baseline["logs"]

    analysis_inputs = [convert(experiment) for experiment in same_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        normalize=True,
    )

    actual = capture_log_data(
        plot_many,
        analysis_results,
        same_problem_experiments,
        plot_type="violin",
    )

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == len(same_problem_experiments)
