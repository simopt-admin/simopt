import matplotlib.pyplot as plt
import numpy as np

from simopt.analysis.terminal_progress import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.plot_type import PlotType
from simopt.plots.terminal_progress import plot_terminal_progress
from test.utils import capture_log_data, to_analysis_input


def test_terminal_progress(experiment):
    desired = capture_log_data(
        plot_terminal_progress,
        experiments=[experiment],
        plot_type=PlotType.VIOLIN,
        normalize=True,
        all_in_one=False,
        plot_title=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
    )

    result = convert(experiment)
    analysis_input = to_analysis_input(result)
    analysis_result = analyze(
        analysis_input,
        normalize=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(
        plot,
        ax,
        analysis_result,
        plot_type="violin",
    )

    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)


def test_terminal_progress_same_problem_experiments(same_problem_experiments):
    desired = capture_log_data(
        plot_terminal_progress,
        experiments=same_problem_experiments,
        plot_type=PlotType.VIOLIN,
        normalize=True,
        all_in_one=False,
        plot_title=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
    )

    analysis_inputs = [
        to_analysis_input(convert(experiment))
        for experiment in same_problem_experiments
    ]
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

    assert len(actual) == len(desired)
    for a, d in zip(actual, desired, strict=True):
        np.testing.assert_allclose(a, d, rtol=1e-5, atol=1e-5)
