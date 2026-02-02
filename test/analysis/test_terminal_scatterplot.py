import matplotlib.pyplot as plt
import numpy as np

from simopt.analysis.terminal_scatterplot import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.plots.terminal_scatterplot import plot_terminal_scatterplots
from test.utils import capture_log_data


def test_terminal_scatterplot(experiment):
    desired = capture_log_data(
        plot_terminal_scatterplots,
        experiments=[[experiment]],
        all_in_one=True,
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
        problem_set_name="PROBLEM_SET",
    )

    analysis_input = convert(experiment)
    analysis_result = analyze(
        analysis_input,
        normalize=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(plot, ax, analysis_result, color="C0", marker="o")

    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)


def test_terminal_scatterplot_same_problem_experiments(same_problem_experiments):
    desired = capture_log_data(
        plot_terminal_scatterplots,
        experiments=[[experiment] for experiment in same_problem_experiments],
        all_in_one=True,
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
        problem_set_name="PROBLEM_SET",
    )

    analysis_inputs = [convert(experiment) for experiment in same_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert len(actual) == len(desired)
    for a, d in zip(actual, desired, strict=True):
        np.testing.assert_allclose(a, d, rtol=1e-5, atol=1e-5)


def test_terminal_scatterplot_different_problem_experiments(
    different_problem_experiments,
):
    desired = capture_log_data(
        plot_terminal_scatterplots,
        experiments=[[experiment] for experiment in different_problem_experiments],
        all_in_one=True,
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
        problem_set_name="PROBLEM_SET",
    )

    analysis_inputs = [
        convert(experiment) for experiment in different_problem_experiments
    ]
    analysis_results = analyze_many(
        analysis_inputs,
        normalize=True,
    )

    actual = capture_log_data(
        plot_many, analysis_results, different_problem_experiments
    )

    assert len(actual) == len(desired)
    for a, d in zip(actual, desired, strict=True):
        np.testing.assert_allclose(a, d, rtol=1e-5, atol=1e-5)
