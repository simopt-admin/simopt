import matplotlib.pyplot as plt
import pytest

from simopt.analysis.terminal_progress import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.plot_type import PlotType
from simopt.plots.terminal_progress import plot_terminal_progress
from tests.utils import assert_logs_match, capture_log_data


@pytest.mark.parametrize("plot_type", ["box", "violin"])
@pytest.mark.parametrize("normalize", [False, True])
def test_terminal_progress(experiment, plot_type, normalize):
    desired = capture_log_data(
        plot_terminal_progress,
        experiments=[experiment],
        plot_type=PlotType(plot_type),
        normalize=normalize,
        all_in_one=False,
        plot_title=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
    )

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


def test_terminal_progress_same_problem_experiments(plot_experiments):
    same_problem_experiments = [
        experiment for group in zip(*plot_experiments, strict=True) for experiment in group
    ]
    desired = []
    for group in zip(*plot_experiments, strict=True):
        desired.extend(
            capture_log_data(
                plot_terminal_progress,
                experiments=list(group),
                plot_type=PlotType.VIOLIN,
                normalize=True,
                all_in_one=False,
                plot_title=None,
                ext=".png",
                save_as_pickle=False,
                solver_set_name="SOLVER_SET",
            )
        )

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
