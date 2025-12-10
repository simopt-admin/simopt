import matplotlib.pyplot as plt
import numpy as np
import pytest

from simopt.analysis.feasibility_progress import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from simopt.plot_type import PlotType
from simopt.plots.feasibility_progress import plot_feasibility_progress
from test.utils import capture_log_data, to_analysis_input


@pytest.mark.parametrize(
    "experiment", ["test/expected_results/SAN2_FCSA.pickle.zst"], indirect=True
)
def test_feasibility_progress(experiment):
    # TODO: Add test for bias_correction = False
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=5,
        confidence_level=0.95,
        bias_correction=True,
    )
    crn_options = CrnOptions(
        across_budget=experiment.crn_across_budget,
        across_macroreps=experiment.crn_across_macroreps,
        across_x0_xstar=experiment.crn_across_init_opt,
    )

    desired = capture_log_data(
        plot_feasibility_progress,
        experiments=[[experiment]],
        plot_type=PlotType.MEAN_FEASIBILITY_PROGRESS,
        score_type="inf_norm",
        norm_degree=1,
        two_sided=False,
        plot_zero=False,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        beta=0.5,
        solver_set_name="SOLVER_SET",
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
    )

    result = convert(experiment)
    analysis_input = to_analysis_input(result)
    analysis_result = analyze(
        analysis_input,
        "mean",
        ci_options,
        crn_options,
        two_sided=False,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(
        plot,
        ax,
        analysis_result,
    )

    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)


def test_feasibility_progress_plot_many(san2_experiments):
    reference = san2_experiments[0]

    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=5,
        confidence_level=0.95,
        bias_correction=True,
    )
    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )

    desired = capture_log_data(
        plot_feasibility_progress,
        experiments=[san2_experiments],
        plot_type=PlotType.MEAN_FEASIBILITY_PROGRESS,
        score_type="inf_norm",
        norm_degree=1,
        two_sided=False,
        plot_zero=False,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        beta=0.5,
        solver_set_name="SOLVER_SET",
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
    )

    analysis_inputs = [
        to_analysis_input(convert(experiment)) for experiment in san2_experiments
    ]
    analysis_results = analyze_many(
        analysis_inputs,
        "mean",
        ci_options,
        crn_options,
        two_sided=False,
    )

    actual = capture_log_data(
        plot_many,
        analysis_results,
        san2_experiments,
        plot_zero=False,
    )

    assert len(actual) == len(desired)
    for a, d in zip(actual, desired, strict=True):
        np.testing.assert_allclose(a, d, rtol=1e-5, atol=1e-5)


def test_feasibility_progress_different_problem_experiments(
    different_problem_experiments_stochastic_constraints,
):
    reference = different_problem_experiments_stochastic_constraints[0]

    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=5,
        confidence_level=0.95,
        bias_correction=True,
    )
    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )

    desired = capture_log_data(
        plot_feasibility_progress,
        experiments=[
            [experiment]
            for experiment in different_problem_experiments_stochastic_constraints
        ],
        plot_type=PlotType.MEAN_FEASIBILITY_PROGRESS,
        score_type="inf_norm",
        norm_degree=1,
        two_sided=False,
        plot_zero=False,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        beta=0.5,
        solver_set_name="SOLVER_SET",
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
    )

    analysis_inputs = [
        to_analysis_input(convert(experiment))
        for experiment in different_problem_experiments_stochastic_constraints
    ]
    analysis_results = analyze_many(
        analysis_inputs,
        "mean",
        ci_options,
        crn_options,
        two_sided=False,
    )

    actual = capture_log_data(
        plot_many,
        analysis_results,
        different_problem_experiments_stochastic_constraints,
        plot_zero=False,
    )

    assert len(actual) == len(desired)
    for a, d in zip(actual, desired, strict=True):
        np.testing.assert_allclose(a, d, rtol=1e-5, atol=1e-5)
