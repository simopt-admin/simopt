import matplotlib.pyplot as plt
import numpy as np

from simopt.analysis.progress_curve import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from simopt.plot_type import PlotType
from simopt.plots.progress_curve import plot_progress_curves
from test.utils import capture_log_data, to_analysis_input


def test_progress_curve(experiment):
    # TODO: Add test for quantile plot
    crn_options = CrnOptions(
        across_budget=experiment.crn_across_budget,
        across_macroreps=experiment.crn_across_macroreps,
        across_x0_xstar=experiment.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=100,
        confidence_level=0.95,
        bias_correction=True,
    )

    desired = capture_log_data(
        plot_progress_curves,
        experiments=[experiment],
        plot_type=PlotType.MEAN,
        beta=0.50,
        normalize=True,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
    )

    result = convert(experiment)
    analysis_input = to_analysis_input(result)
    analysis_result = analyze(
        analysis_input,
        "mean",
        ci_options,
        crn_options,
        normalize=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(plot, ax, analysis_result)

    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)


def test_progress_curve_same_problem_experiments(same_problem_experiments):
    reference = same_problem_experiments[0]

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

    desired = capture_log_data(
        plot_progress_curves,
        experiments=same_problem_experiments,
        plot_type=PlotType.MEAN,
        beta=0.50,
        normalize=True,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        plot_title=None,
        legend_loc=None,
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
        "mean",
        ci_options,
        crn_options,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert len(actual) == len(desired)
    for a, d in zip(actual, desired, strict=True):
        np.testing.assert_allclose(a, d, rtol=1e-5, atol=1e-5)
