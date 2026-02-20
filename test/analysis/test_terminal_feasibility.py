import matplotlib.pyplot as plt
import numpy as np
import pytest

from simopt.analysis.terminal_feasibility import analyze, analyze_many, plot, plot_many
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from simopt.plot_type import PlotType
from simopt.plots.terminal_feasibility import plot_terminal_feasibility
from test.utils import capture_log_data


@pytest.mark.parametrize(
    "experiment", ["test/expected_results/SAN2_FCSA.pickle.zst"], indirect=True
)
def test_terminal_feasibility(experiment):
    crn_options = CrnOptions(
        across_budget=experiment.crn_across_budget,
        across_macroreps=experiment.crn_across_macroreps,
        across_x0_xstar=experiment.crn_across_init_opt,
    )
    ci_options = ConfidenceIntervalOptions(
        n_bootstraps=10,
        confidence_level=0.95,
        bias_correction=True,
    )

    desired = capture_log_data(
        plot_terminal_feasibility,
        experiments=[[experiment]],
        plot_type=PlotType.FEASIBILITY_SCATTER,
        score_type="norm",
        two_sided=True,
        plot_zero=False,
        plot_optimal=False,
        norm_degree=2,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        bias_correction=True,
        solver_set_name="SOLVER_SET",
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
    )

    analysis_input = convert(experiment)
    analysis_result = analyze(
        analysis_input,
        ci_options,
        crn_options,
        score_type="norm",
        norm_degree=2,
        two_sided=True,
    )

    _, ax = plt.subplots()
    actual = capture_log_data(plot, ax, analysis_result, "C0", "o")

    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)


def test_terminal_feasibility_same_problem_experiments(san2_experiments):
    reference = san2_experiments[0]
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
        plot_terminal_feasibility,
        experiments=[[experiment] for experiment in san2_experiments],
        plot_type=PlotType.FEASIBILITY_SCATTER,
        score_type="norm",
        two_sided=True,
        plot_zero=False,
        plot_optimal=False,
        norm_degree=2,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        bias_correction=True,
        solver_set_name="SOLVER_SET",
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
    )

    analysis_inputs = [convert(experiment) for experiment in san2_experiments]
    analysis_results = analyze_many(
        analysis_inputs,
        ci_options,
        crn_options,
        score_type="norm",
        norm_degree=2,
        two_sided=True,
    )

    actual = capture_log_data(plot_many, analysis_results, san2_experiments)

    assert len(actual) == len(desired)
    for a, d in zip(actual, desired, strict=True):
        np.testing.assert_allclose(a, d, rtol=1e-5, atol=1e-5)


def test_terminal_feasibility_different_problem_experiments(
    different_problem_experiments_stochastic_constraints,
):
    reference = different_problem_experiments_stochastic_constraints[0]
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
        plot_terminal_feasibility,
        experiments=[
            [experiment] for experiment in different_problem_experiments_stochastic_constraints
        ],
        plot_type=PlotType.FEASIBILITY_SCATTER,
        score_type="norm",
        two_sided=True,
        plot_zero=False,
        plot_optimal=False,
        norm_degree=2,
        all_in_one=True,
        n_bootstraps=ci_options.n_bootstraps,
        conf_level=ci_options.confidence_level,
        plot_conf_ints=True,
        bias_correction=True,
        solver_set_name="SOLVER_SET",
        plot_title=None,
        legend_loc=None,
        ext=".png",
        save_as_pickle=False,
    )

    analysis_inputs = [
        convert(experiment) for experiment in different_problem_experiments_stochastic_constraints
    ]
    analysis_results = analyze_many(
        analysis_inputs,
        ci_options,
        crn_options,
        score_type="norm",
        norm_degree=2,
        two_sided=True,
    )

    actual = capture_log_data(
        plot_many,
        analysis_results,
        different_problem_experiments_stochastic_constraints,
    )

    assert len(actual) == len(desired)
    for a, d in zip(actual, desired, strict=True):
        np.testing.assert_allclose(a, d, rtol=1e-5, atol=1e-5)
