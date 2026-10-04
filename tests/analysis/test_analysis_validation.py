from statistics import quantiles

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from simopt.analysis import (
    auc,
    common,
    diff,
    feasibility_progress,
    progress_curve,
    solvability_cdf,
    solvability_profile,
    terminal_feasibility,
    terminal_progress,
    terminal_scatterplot,
)
from simopt.compat import convert
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from tests.analysis.options_helpers import (
    CI_CASES,
    MODULES,
    analyze_case,
    forbid_bootstrap,
    interval_reference,
    plot_module_many,
)


@pytest.mark.parametrize("module", MODULES)
def test_empty_plot_results(module):
    kwargs = {"plot_zero": False} if module == feasibility_progress else {}
    _, ax = plot_module_many(module, [], [], **kwargs)
    assert not ax.lines and not ax.collections and not ax.patches
    if module in (auc, terminal_scatterplot, diff):
        assert ax.get_legend() is None


@pytest.mark.parametrize(
    "module", [module for module in MODULES if module not in (diff, solvability_profile)]
)
def test_mismatched_plot_inputs(module, option_experiment):
    with pytest.raises(ValueError, match="same length"):
        plot_module_many(module, [], [option_experiment])


@pytest.mark.parametrize("module", [solvability_profile, diff])
def test_mismatched_analysis_inputs(module, option_experiment):
    with pytest.raises(ValueError, match=r"shorter|longer|same length"):
        if module == diff:
            module.analyze_many(
                [], [option_experiment], "cdf", option_experiment.solver.name, None, CrnOptions()
            )
        else:
            module.analyze_many([], [option_experiment], "cdf", None, CrnOptions())


@pytest.mark.parametrize("module", [progress_curve, feasibility_progress])
def test_invalid_aggregation(module, option_experiment):
    kwargs = {"two_sided": False} if module == feasibility_progress else {}
    with pytest.raises(ValueError, match=r"unsupported|agg"):
        module.analyze(convert(option_experiment), "unknown", None, CrnOptions(), **kwargs)


@pytest.mark.parametrize("agg", ["mean", "quantile"])
def test_invalid_two_sided_aggregation(option_experiment, agg):
    with pytest.raises(ValueError, match="two sided"):
        feasibility_progress.analyze(
            convert(option_experiment), agg, None, CrnOptions(), two_sided=True
        )


@pytest.mark.parametrize("module", [solvability_profile, diff])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"plot_type": "unknown"}, "plot.type"),
        ({"beta": 0}, "Beta"),
        ({"beta": 1}, "Beta"),
        ({"solve_tolerance": 0}, "tolerance"),
        ({"solve_tolerance": 1.1}, "tolerance"),
    ],
)
def test_invalid_solvability_options(module, option_pair, kwargs, message):
    kwargs = dict(kwargs)
    mode = kwargs.pop("plot_type", "quantile")
    with pytest.raises(ValueError, match=message):
        if module == diff:
            module.analyze_many(
                [convert(experiment) for experiment in option_pair],
                option_pair,
                mode,
                option_pair[0].solver.name,
                None,
                CrnOptions(),
                **kwargs,
            )
        else:
            module.analyze(convert(option_pair[0]), mode, None, CrnOptions(), **kwargs)


def test_missing_reference_solver(option_pair):
    with pytest.raises(ValueError, match=r"reference solver.*not found"):
        diff.analyze_many(
            [convert(experiment) for experiment in option_pair],
            option_pair,
            "cdf",
            "missing",
            None,
            CrnOptions(),
        )


@pytest.mark.parametrize(("name", "mode", "normalize"), CI_CASES)
@pytest.mark.parametrize(
    ("options", "message"),
    [
        (ConfidenceIntervalOptions(n_bootstraps=0), "bootstraps"),
        (ConfidenceIntervalOptions(n_bootstraps=-1), "bootstraps"),
        (ConfidenceIntervalOptions(confidence_level=0), "Confidence level"),
        (ConfidenceIntervalOptions(confidence_level=1), "Confidence level"),
    ],
)
def test_invalid_ci_in_each_analysis(option_pair, name, mode, normalize, options, message):
    with pytest.raises(ValueError, match=message):
        analyze_case(name, mode, option_pair, options, CrnOptions(), normalize)


@pytest.mark.parametrize("module", [terminal_progress, common])
def test_invalid_distribution_mode(module, option_experiment):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"[Pp]lot type"):
        if module == common:
            module.plot_distribution(
                ax, [{"solver": "A", "value": 1}], "solver", "value", "unknown"
            )
        else:
            module.plot(ax, module.analyze(convert(option_experiment)), plot_type="unknown")


@pytest.mark.parametrize("module", [progress_curve, feasibility_progress])
def test_empty_all_curves(module):
    result_type = (
        module.ProgressCurveResult if module == progress_curve else module.FeasibilityProgressResult
    )
    result = result_type(pd.DataFrame(columns=["mrep", "budget", "value"]), "all")
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"no .*curve|no feasibility"):
        module.plot(ax, result)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    ("crossing", "expected_values"), [(None, [0, 0]), (0, [0, 1, 1]), (0.5, [0, 1, 1]), (1, [0, 1])]
)
def test_quantile_crossing_boundaries(option_experiment, normalize, crossing, expected_values):
    data = convert(option_experiment)
    data.mean_df["budget"] = np.tile([0, 5, 10], 4)
    data.mean_df["normalized_budget"] = data.mean_df["budget"] / 10
    for prefix in ["", "normalized_"]:
        if crossing is None:
            data.mean_df[f"{prefix}objective"] = 1.0
        else:
            data.mean_df[f"{prefix}objective"] = np.where(
                data.mean_df["normalized_budget"] >= crossing, 0.0, 1.0
            )
    result = solvability_profile.analyze(
        data, "quantile", None, CrnOptions(), normalize=normalize, solve_tolerance=0.5
    )
    endpoint = 1 if normalize else 10
    budgets = [0, endpoint] if crossing in (None, 1) else [0, crossing * endpoint, endpoint]
    np.testing.assert_allclose(result.df["budget"], budgets)
    np.testing.assert_allclose(result.df["value"], expected_values)
    # Infinite crossing times must never produce NaNs in an unsolved profile.
    assert np.isfinite(result.df.to_numpy()).all()


@pytest.mark.parametrize("module", [solvability_cdf, solvability_profile])
@pytest.mark.parametrize("normalize", [False, True])
def test_strict_crossing_threshold_and_end_budget(option_experiment, module, normalize):
    data = convert(option_experiment)
    data.mean_df["budget"] = np.tile([0, 5, 10], 4)
    data.mean_df["normalized_budget"] = data.mean_df["budget"] / 10
    values = np.tile([1.0, 0.5, 0.0], 4)
    data.mean_df["objective"] = data.mean_df["normalized_objective"] = values
    args = [data]
    if module == solvability_profile:
        args.append("cdf")
    result = module.analyze(*args, None, CrnOptions(), normalize=normalize, solve_tolerance=0.5)
    endpoint = 1 if normalize else 10
    column = "cdf" if module == solvability_cdf else "value"
    np.testing.assert_allclose(result.df["budget"], [0, endpoint, endpoint])
    np.testing.assert_allclose(result.df[column], [0, 1, 1])


def test_ci_estimator_with_duplicate_budgets():
    estimator = pd.DataFrame({"budget": [0, 0.5, 0.5, 1], "value": [0, 0.5, 0.5, 1]})
    samples = [
        pd.DataFrame({"budget": [0, 0.75, 1], "value": [0, value, 1]})
        for value in [0.25, 0.5, 0.75]
    ]
    options = ConfidenceIntervalOptions(n_bootstraps=3, confidence_level=0.8)
    result = common.compute_ci(samples, estimator, "value", options)
    expected = interval_reference(
        [[0, value, 1] for value in [0.25, 0.5, 0.75]], [0, 0.5, 1], 0.8, True
    )
    np.testing.assert_allclose(result[["lb", "ub"]].to_numpy().T, expected)


@pytest.mark.parametrize("method", ["python", "pandas"])
def test_quantile_aggregation_methods(method):
    df = pd.DataFrame({"budget": [0] * 4, "mrep": range(4), "value": [0, 2, 4, 8]})
    result = common.aggregate_curve(df, "quantile", "value", beta=0.73, quantile_method=method)
    expected = (
        quantiles([0, 2, 4, 8], n=100)[int(0.73 * 99)]
        if method == "python"
        else np.quantile([0, 2, 4, 8], 0.73)
    )
    assert result["value"].iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("agg", "beta", "method", "message"),
    [
        ("unknown", 0.5, "python", "unsupported curve"),
        ("quantile", None, "python", "beta is required"),
        ("quantile", 0.5, "unknown", "unsupported quantile method"),
    ],
)
def test_invalid_aggregation_options(agg, beta, method, message):
    df = pd.DataFrame({"budget": [0, 0], "mrep": [0, 1], "value": [1, 2]})
    with pytest.raises(ValueError, match=message):
        common.aggregate_curve(df, agg, "value", beta=beta, quantile_method=method)


def test_missing_bias_estimator():
    df = pd.DataFrame({"budget": [0], "value": [1]})
    with pytest.raises(ValueError, match="bias correction estimator is required"):
        common.compute_ci([df], None, "value", ConfidenceIntervalOptions(n_bootstraps=1))


@pytest.mark.parametrize("module", [progress_curve, feasibility_progress])
@pytest.mark.parametrize("beta", [-0.1, 0, 1, 1.1])
def test_invalid_progress_quantiles(module, option_experiment, beta):
    kwargs = {"two_sided": False} if module == feasibility_progress else {}
    with pytest.raises(ValueError, match="Beta"):
        module.analyze(
            convert(option_experiment), "quantile", None, CrnOptions(), beta=beta, **kwargs
        )


def test_invalid_terminal_feasibility_mode(option_experiment):
    result = terminal_feasibility.analyze(convert(option_experiment), None, CrnOptions())
    with pytest.raises(ValueError, match="plot_type"):
        terminal_feasibility.plot_many([result], [option_experiment], plot_type="unknown")


@pytest.mark.parametrize("module", [feasibility_progress, terminal_feasibility])
def test_invalid_feasibility_score_type(module, option_experiment):
    # The history includes positive violations, which exercise score validation.
    args = [convert(option_experiment)]
    if module == feasibility_progress:
        args.append("all")
    with pytest.raises(ValueError, match="invalid method"):
        module.analyze(*args, None, CrnOptions(), score_type="unknown")


@pytest.mark.parametrize("module", [progress_curve, feasibility_progress])
def test_all_mode_ignores_ci_options(option_experiment, module, monkeypatch):
    monkeypatch.setattr(common, "bootstrap", forbid_bootstrap)
    args = [convert(option_experiment), "all", ConfidenceIntervalOptions(), CrnOptions()]
    result = module.analyze(*args)
    assert "lb" not in result.df and "ub" not in result.df


@pytest.mark.parametrize("module", [solvability_profile, diff])
def test_no_comparison_solver(option_experiment, module):
    args = [[convert(option_experiment)], [option_experiment], "cdf"]
    if module == diff:
        args.append(option_experiment.solver.name)
    results = module.analyze_many(*args, None, CrnOptions())
    assert len(results) == (0 if module == diff else 1)
    _, ax = plot_module_many(module, results, [option_experiment])
    assert len(ax.lines) == len(results)


def test_feasibility_without_constraints(option_experiment):
    data = convert(option_experiment)
    data.mean_df["stochastic_constraints"] = [np.array([])] * len(data.mean_df)
    progress = feasibility_progress.analyze(data, "mean", None, CrnOptions(), two_sided=False)
    terminal = terminal_feasibility.analyze(data, None, CrnOptions())
    np.testing.assert_array_equal(progress.df["value"], np.zeros(len(progress.df)))
    np.testing.assert_array_equal(terminal.df["feasibility"], np.zeros(len(terminal.df)))


def test_inconsistent_postreplication_counts(option_experiment):
    data = convert(option_experiment)
    data.full_df = data.full_df.iloc[:-1]
    with pytest.raises(ValueError, match="not consistent"):
        progress_curve.analyze(
            data, "mean", ConfidenceIntervalOptions(n_bootstraps=2), CrnOptions()
        )
