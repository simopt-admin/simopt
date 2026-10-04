import matplotlib.pyplot as plt

from simopt.analysis.auc import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from simopt.plots.area_scatterplot import plot_area_scatterplots
from tests.utils import assert_logs_match, capture_log_data


def test_auc(experiment):
    crn_options = CrnOptions(
        across_budget=experiment.crn_across_budget,
        across_macroreps=experiment.crn_across_macroreps,
        across_x0_xstar=experiment.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.9,
        bias_correction=True,
    )
    desired = capture_log_data(
        plot_area_scatterplots,
        experiments=[[experiment]],
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=True,
        plot_title=None,
        legend_loc="best",
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
        problem_set_name="PROBLEM_SET",
    )

    analysis_input = convert(experiment)
    analysis_result = analyze(
        analysis_input,
        ci_options,
        crn_options,
        normalize=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(plot, ax, analysis_result, "C0", "o")

    assert_logs_match(actual, desired)


def test_auc_same_problem_experiments(plot_experiments):
    same_problem_experiments = [experiment for row in plot_experiments for experiment in row]
    reference = same_problem_experiments[0]

    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.9,
        bias_correction=True,
    )
    desired = capture_log_data(
        plot_area_scatterplots,
        experiments=plot_experiments,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=True,
        plot_title=None,
        legend_loc="best",
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
        problem_set_name="PROBLEM_SET",
    )

    analysis_inputs = [convert(experiment) for experiment in same_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        ci_options,
        crn_options,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, same_problem_experiments)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired) == len(same_problem_experiments)


def test_auc_different_problem_experiments(different_problem_experiments):
    reference = different_problem_experiments[0]

    crn_options = CrnOptions(
        across_budget=reference.crn_across_budget,
        across_macroreps=reference.crn_across_macroreps,
        across_x0_xstar=reference.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.9,
        bias_correction=True,
    )
    desired = capture_log_data(
        plot_area_scatterplots,
        experiments=[different_problem_experiments],
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        print_max_hw=True,
        plot_title=None,
        legend_loc="best",
        ext=".png",
        save_as_pickle=False,
        solver_set_name="SOLVER_SET",
        problem_set_name="PROBLEM_SET",
    )

    analysis_inputs = [convert(experiment) for experiment in different_problem_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        ci_options,
        crn_options,
        normalize=True,
    )

    actual = capture_log_data(plot_many, analysis_results, different_problem_experiments)

    assert_logs_match(actual, desired)
    assert len(actual) == len(desired)
