import numpy as np

from simopt.analysis.diff import analyze_many, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from simopt.plot_type import PlotType
from simopt.plots.solvability_profile import plot_solvability_profiles
from test.utils import capture_log_data


def test_diff_solvability_cdf(same_problem_experiments):
    reference = same_problem_experiments[0]
    ci_options = ConfidenceIntervalOptions(n_bootstraps=10, confidence_level=0.95)
    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )

    desired = capture_log_data(
        plot_solvability_profiles,
        experiments=[[experiment] for experiment in same_problem_experiments],
        plot_type=PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        solve_tol=0.1,
        beta=0.5,
        ref_solver="ADAM",
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
        "ADAM",
        ci_options,
        crn_options,
        normalize=True,
        solve_tolerance=0.1,
        beta=0.5,
    )
    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert len(actual) == len(desired)
    for a1, d1 in zip(actual, desired, strict=True):
        for a2, d2 in zip(a1, d1, strict=True):
            np.testing.assert_allclose(a2, d2, rtol=1e-5, atol=1e-5)


def test_diff_solvability_quantile(same_problem_experiments):
    reference = same_problem_experiments[0]
    ci_options = ConfidenceIntervalOptions(n_bootstraps=10, confidence_level=0.95)
    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )

    desired = capture_log_data(
        plot_solvability_profiles,
        experiments=[[experiment] for experiment in same_problem_experiments],
        plot_type=PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=False,
        solve_tol=0.1,
        beta=0.5,
        ref_solver="ADAM",
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
        "quantile",
        "ADAM",
        ci_options,
        crn_options,
        normalize=True,
        solve_tolerance=0.1,
        beta=0.5,
    )
    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert len(actual) == len(desired)
    for a1, d1 in zip(actual, desired, strict=True):
        for a2, d2 in zip(a1, d1, strict=True):
            np.testing.assert_allclose(a2, d2, rtol=1e-5, atol=1e-5)
