import matplotlib.pyplot as plt
import numpy as np

from simopt.analysis.solvability_profile import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from simopt.plot_type import PlotType
from simopt.plots.solvability_profile import plot_solvability_profiles
from test.utils import capture_log_data


def test_solvability_profile(experiment):
    solve_tolerance = 0.1
    crn_options = CrnOptions(
        across_budget=experiment.crn_across_budget,
        across_macroreps=experiment.crn_across_macroreps,
        across_x0_xstar=experiment.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
    )

    desired = capture_log_data(
        plot_solvability_profiles,
        experiments=[[experiment]],
        # TODO: test quantile solvability
        plot_type=PlotType.CDF_SOLVABILITY,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        solve_tol=solve_tolerance,
        beta=0.5,
        ref_solver=None,
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
        "cdf",
        ci_options,
        crn_options,
        solve_tolerance=solve_tolerance,
        normalize=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(
        plot,
        ax,
        analysis_result,
    )

    assert len(actual) == len(desired) == 1
    for actual_value, expected_value in zip(actual[0], desired[0], strict=True):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)


def test_solvability_profile_same_problem_experiments(same_problem_experiments):
    solve_tolerance = 0.1
    reference = same_problem_experiments[0]
    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
    )

    desired = capture_log_data(
        plot_solvability_profiles,
        experiments=[[experiment] for experiment in same_problem_experiments],
        # TODO: test quantile solvability
        plot_type=PlotType.CDF_SOLVABILITY,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        solve_tol=solve_tolerance,
        beta=0.5,
        ref_solver=None,
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
        same_problem_experiments,
        "cdf",
        ci_options,
        crn_options,
        solve_tolerance=solve_tolerance,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments, "cdf")

    assert len(actual) == len(desired)
    for a1, d1 in zip(actual, desired, strict=True):
        for a2, d2 in zip(a1, d1, strict=True):
            np.testing.assert_allclose(a2, d2, rtol=1e-5, atol=1e-5)


def test_solvability_profile_different_problem_experiments(
    different_problem_experiments,
):
    solve_tolerance = 0.1
    reference = different_problem_experiments[0]
    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
    )

    desired = capture_log_data(
        plot_solvability_profiles,
        experiments=[different_problem_experiments],
        # TODO: test quantile solvability
        plot_type=PlotType.CDF_SOLVABILITY,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        solve_tol=solve_tolerance,
        beta=0.5,
        ref_solver=None,
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
        problem_set_name="PROBLEM_SET",
    )

    analysis_inputs = [convert(experiment) for experiment in different_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        different_problem_experiments,
        "cdf",
        ci_options,
        crn_options,
        solve_tolerance=solve_tolerance,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, different_problem_experiments, "cdf")

    assert len(actual) == len(desired)
    for a1, d1 in zip(actual, desired, strict=True):
        for a2, d2 in zip(a1, d1, strict=True):
            np.testing.assert_allclose(a2, d2, rtol=1e-5, atol=1e-5)
