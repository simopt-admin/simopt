from collections.abc import Callable
from copy import deepcopy
from itertools import product
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from simopt.analysis import (
    auc,
    common,
    diff,
    feasibility_progress,
    progress_curve,
    solvability,
    solvability_cdf,
    solvability_profile,
    terminal_feasibility,
    terminal_progress,
    terminal_scatterplot,
)
from simopt.compat import convert
from simopt.curve import Curve
from simopt.curve_utils import (
    difference_of_curves,
    mean_of_curves,
    quantile_of_curves,
)
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from tests.analysis.options_helpers import (
    MODULES,
    analyze_case,
    curve_reference,
    distribution_records,
    feasibility_reference,
    forbid_bootstrap,
    module_results,
    plot_module_many,
)

SCORE_CASES: list[tuple[Literal["inf_norm", "norm"], int]] = [
    ("inf_norm", 1),
    ("norm", 1),
    ("norm", 2),
    ("norm", 3),
]

CONFIG_CASES = (
    [
        progress_curve.ProgressCurve(agg=agg, beta=beta, normalize=normalize, ci_options=None)
        for agg in ["all", "mean", "quantile"]
        for beta in ([0.1, 0.5, 0.9] if agg == "quantile" else [0.5])
        for normalize in [False, True]
    ]
    + [
        feasibility_progress.FeasibilityProgress(
            agg=agg,
            beta=beta,
            score_type=score_type,
            norm_degree=degree,
            two_sided=two_sided,
            ci_options=None,
        )
        for agg in ["all", "mean", "quantile"]
        for beta in ([0.1, 0.5, 0.9] if agg == "quantile" else [0.5])
        for score_type, degree in SCORE_CASES
        for two_sided in ([False, True] if agg == "all" else [False])
    ]
    + [
        solvability_cdf.SolvabilityCdf(
            normalize=normalize, solve_tolerance=tolerance, ci_options=None
        )
        for normalize in [False, True]
        for tolerance in [0.1, 0.6, 1.0]
    ]
    + [
        solvability_profile.SolvabilityProfile(
            plot_type=mode,
            normalize=normalize,
            beta=beta,
            solve_tolerance=tolerance,
            ci_options=None,
        )
        for mode in ["cdf", "quantile"]
        for normalize in [False, True]
        for beta in ([0.1, 0.5, 0.9] if mode == "quantile" else [0.5])
        for tolerance in [0.1, 0.6, 1.0]
    ]
    + [
        diff.DiffSolvability(
            plot_type=mode,
            reference_solver=reference,
            normalize=normalize,
            beta=beta,
            solve_tolerance=tolerance,
            ci_options=None,
        )
        for mode in ["cdf", "quantile"]
        for reference in ["RNDSRCH", "comparison"]
        for normalize in [False, True]
        for beta in ([0.1, 0.5, 0.9] if mode == "quantile" else [0.5])
        for tolerance in [0.1, 0.6, 1.0]
    ]
    + [auc.Auc(ci_options=None)]
    + [
        terminal_feasibility.TerminalFeasibility(
            plot_type=mode,
            score_type=score_type,
            norm_degree=degree,
            two_sided=two_sided,
            ci_options=None,
        )
        for mode in ["scatter", "violin"]
        for score_type, degree in SCORE_CASES
        for two_sided in [False, True]
    ]
    + [
        terminal_progress.TerminalProgress(normalize=normalize, plot_type=mode)
        for normalize in [False, True]
        for mode in ["box", "violin"]
    ]
    + [terminal_scatterplot.TerminalScatter(normalize=normalize) for normalize in [False, True]]
)


@pytest.mark.parametrize("config", CONFIG_CASES, ids=lambda config: repr(config))
def test_configuration_plot_options(option_pair, config, monkeypatch):
    monkeypatch.setattr(common, "bootstrap", forbid_bootstrap)
    monkeypatch.setattr(solvability, "bootstrap", forbid_bootstrap)
    monkeypatch.setattr(terminal_feasibility, "_bootstrap", forbid_bootstrap)
    records = distribution_records(monkeypatch)
    inputs = [convert(experiment) for experiment in option_pair]
    fig, ax = config.plot(inputs, option_pair)
    assert fig is ax.figure
    if isinstance(config, (progress_curve.ProgressCurve, feasibility_progress.FeasibilityProgress)):
        expected = []
        for experiment in option_pair:
            if isinstance(config, progress_curve.ProgressCurve):
                curves = (
                    experiment.progress_curves if config.normalize else experiment.objective_curves
                )
            else:
                curves = [
                    Curve(
                        budgets,
                        [
                            feasibility_reference(
                                lhs, config.score_type, config.norm_degree, config.two_sided
                            )
                            for lhs in values
                        ],
                    )
                    for budgets, values in zip(
                        experiment.all_intermediate_budgets, experiment.all_est_lhs, strict=True
                    )
                ]
            selected = (
                curves
                if config.agg == "all"
                else [
                    mean_of_curves(curves)
                    if config.agg == "mean"
                    else quantile_of_curves(curves, config.beta)
                ]
            )
            expected.extend(selected)
        lines = [line for line in ax.lines if line.get_linewidth() == 2]
        assert len(lines) == len(expected)
        for line, curve in zip(lines, expected, strict=True):
            np.testing.assert_allclose(np.asarray(line.get_xdata()), curve.x_vals)
            np.testing.assert_allclose(np.asarray(line.get_ydata()), curve.y_vals)
            assert line.get_drawstyle() == "steps-post"
        if isinstance(config, feasibility_progress.FeasibilityProgress):
            zeros = [line for line in ax.lines if line.get_linewidth() != 2]
            assert len(zeros) == len(option_pair)
            for line in zeros:
                np.testing.assert_array_equal(line.get_ydata(), [0, 0])
            assert ax.get_ylabel() == "Feasibility Score"
        else:
            assert ax.get_ylabel() == "Objective"
        assert ax.get_xlabel() == "Budget"
    elif isinstance(
        config,
        (
            solvability_cdf.SolvabilityCdf,
            solvability_profile.SolvabilityProfile,
            diff.DiffSolvability,
        ),
    ):
        mode = "cdf" if isinstance(config, solvability_cdf.SolvabilityCdf) else config.plot_type
        beta = getattr(config, "beta", 0.5)
        curves = [
            curve_reference(
                "profile",
                mode,
                experiment.progress_curves if config.normalize else experiment.objective_curves,
                [],
                beta,
                config.solve_tolerance,
            )
            for experiment in option_pair
        ]
        if isinstance(config, diff.DiffSolvability):
            reference = 0 if config.reference_solver == option_pair[0].solver.name else 1
            curves = [difference_of_curves(curves[1 - reference], curves[reference])]
            labels = [f"{option_pair[1 - reference].solver.name} - {config.reference_solver}"]
            assert ax.get_ylabel() == "Difference"
        else:
            labels = [experiment.solver.name for experiment in option_pair]
        assert len(ax.lines) == len(curves)
        for line, curve in zip(ax.lines, curves, strict=True):
            # Aggregated profiles remove duplicated budgets.
            np.testing.assert_allclose(
                line.get_ydata(), [curve.lookup(budget) for budget in line.get_xdata()]
            )
            assert max(line.get_xdata()) == (1 if config.normalize else 10)
            assert np.all(np.diff(line.get_xdata()) >= 0)
        if not isinstance(config, solvability_cdf.SolvabilityCdf):
            assert [text.get_text() for text in ax.get_legend().get_texts()] == labels
            assert ax.get_xlabel() == "Budget"
    elif isinstance(
        config, (terminal_progress.TerminalProgress, terminal_feasibility.TerminalFeasibility)
    ):
        expected = []
        for experiment in option_pair:
            if isinstance(config, terminal_progress.TerminalProgress):
                curves = (
                    experiment.progress_curves if config.normalize else experiment.objective_curves
                )
                expected.extend(curve.y_vals[-1] for curve in curves)
            else:
                expected.extend(
                    feasibility_reference(
                        values[-1], config.score_type, config.norm_degree, config.two_sided
                    )
                    for values in experiment.all_est_lhs
                )
        if config.plot_type == "scatter":
            assert len(ax.collections) == 2
            for i, collection in enumerate(ax.collections):
                objectives = [curve.y_vals[-1] for curve in option_pair[i].objective_curves]
                np.testing.assert_allclose(
                    collection.get_offsets(),
                    np.array([objectives, expected[i * 4 : (i + 1) * 4]]).T,
                )
            assert ax.get_xlabel() == "Objective"
            assert ax.get_ylabel() == "Feasibility Score"
        else:
            assert len(records) == 1
            np.testing.assert_allclose(records[0], expected)
            assert ax.patches or ax.collections or ax.lines
            assert ax.get_xlabel() == "Solver"
    else:
        expected = []
        for experiment in option_pair:
            if isinstance(config, auc.Auc):
                values = [curve.compute_area_under_curve() for curve in experiment.progress_curves]
            else:
                curves = (
                    experiment.progress_curves if config.normalize else experiment.objective_curves
                )
                values = [curve.y_vals[-1] for curve in curves]
            expected.append([np.mean(values), np.std(values, ddof=1)])
        assert len(ax.collections) == len(expected)
        for collection, value in zip(ax.collections, expected, strict=True):
            np.testing.assert_allclose(collection.get_offsets(), [value])
        assert ax.get_xlabel() == "Mean"
        assert ax.get_ylabel() == "Standard Deviation"
    assert not any(line.get_linestyle() == "--" for line in ax.lines)


@pytest.mark.parametrize("plot_zero", [False, True])
@pytest.mark.parametrize("agg", ["all", "mean", "quantile"])
def test_feasibility_zero_line(option_experiment, plot_zero, agg):
    result = feasibility_progress.analyze(
        convert(option_experiment), agg, None, CrnOptions(), two_sided=False
    )
    _, ax = plt.subplots()
    feasibility_progress.plot(ax, result, plot_zero=plot_zero, color="magenta")
    n_curves = 4 if agg == "all" else 1
    assert len(ax.lines) == n_curves + plot_zero
    for line in ax.lines[:n_curves]:
        assert to_rgba(line.get_color()) == to_rgba("magenta")
    if plot_zero:
        np.testing.assert_array_equal(ax.lines[-1].get_ydata(), [0, 0])


@pytest.mark.parametrize("name", ["progress", "feasibility", "cdf", "profile", "difference"])
def test_curve_ci_artists(option_pair, name):
    result = analyze_case(
        name,
        "cdf" if name in ("cdf", "profile", "difference") else "mean",
        option_pair,
        ConfidenceIntervalOptions(n_bootstraps=8),
        CrnOptions(),
        True,
    )
    module = {
        "progress": progress_curve,
        "feasibility": feasibility_progress,
        "cdf": solvability_cdf,
        "profile": solvability_profile,
        "difference": diff,
    }[name]
    _, ax = plt.subplots()
    if name == "feasibility":
        feasibility_progress.plot(ax, result, plot_zero=False, color="magenta")
    else:
        module.plot(ax, result, color="magenta")
    ci = result.ci if name in ("cdf", "profile", "difference") else result.df
    assert len(ax.lines) == 3
    assert len(ax.collections) == 1
    for line, column in zip(ax.lines[1:], ["lb", "ub"], strict=True):
        np.testing.assert_allclose(np.asarray(line.get_xdata()), ci["budget"])
        np.testing.assert_allclose(np.asarray(line.get_ydata()), ci[column])
        assert line.get_linestyle() == "--"
        assert line.get_linewidth() == 1
        assert line.get_drawstyle() == "steps-post"
        assert to_rgba(line.get_color()) == to_rgba("magenta")
    collection = ax.collections[0]
    assert collection.get_alpha() == 0.2
    np.testing.assert_allclose(np.asarray(collection.get_facecolor())[0], to_rgba("magenta", 0.2))
    vertices = np.concatenate([path.vertices for path in collection.get_paths()])
    assert min(vertices[:, 0]) == min(ci["budget"])
    assert max(vertices[:, 0]) == max(ci["budget"])
    assert min(vertices[:, 1]) == pytest.approx(min(ci["lb"]))
    assert max(vertices[:, 1]) == pytest.approx(max(ci["ub"]))


@pytest.mark.parametrize("module", [module for module in MODULES if module != terminal_progress])
@pytest.mark.parametrize("custom_style", [False, True])
def test_plot_many_styles_and_legends(option_pair, module, custom_style):
    experiments = [*option_pair, deepcopy(option_pair[0])]
    inputs = [convert(experiment) for experiment in experiments]
    results = module_results(module, inputs, experiments)
    kwargs = {}
    if custom_style:
        kwargs["colors"] = ["magenta", "cyan"]
        if module in (auc, terminal_feasibility, terminal_scatterplot):
            kwargs["markers"] = ["D"]
    if module == feasibility_progress:
        kwargs["plot_zero"] = False
    _, ax = plot_module_many(module, results, experiments, **kwargs)
    palette = ["magenta", "cyan"] if custom_style else [f"C{i}" for i in range(len(results))]
    indices = [0, 1, 0] if module in (auc, terminal_scatterplot) else list(range(len(results)))
    if module in (auc, terminal_feasibility, terminal_scatterplot):
        from matplotlib.markers import MarkerStyle

        assert len(ax.collections) == len(results)
        for collection, index in zip(ax.collections, indices, strict=True):
            np.testing.assert_allclose(
                collection.get_facecolor()[0], to_rgba(palette[index % len(palette)])
            )
            if custom_style:
                marker = MarkerStyle("D")
                np.testing.assert_allclose(
                    np.asarray(collection.get_paths()[0].vertices),
                    np.asarray(marker.get_path().transformed(marker.get_transform()).vertices),
                )
    else:
        assert len(ax.lines) == len(results)
        for line, index in zip(ax.lines, indices, strict=True):
            assert to_rgba(line.get_color()) == to_rgba(palette[index % len(palette)])
    if module == solvability_cdf:
        assert ax.get_legend() is None
    else:
        legend = ax.get_legend()
        assert legend is not None
        if module == diff:
            expected = [f"comparison - {experiments[0].solver.name}"]
        elif module == terminal_feasibility:
            expected = [
                f"{experiment.solver.name} | {experiment.problem.name}"
                for experiment in experiments
            ]
        elif module in (auc, terminal_scatterplot, solvability_profile):
            expected = [experiment.solver.name for experiment in experiments[:2]]
        else:
            expected = [experiment.solver.name for experiment in experiments]
        assert [text.get_text() for text in legend.get_texts()] == expected


@pytest.mark.parametrize("module", [auc, terminal_feasibility])
@pytest.mark.parametrize("with_ci", [False, True])
def test_scatter_artists_and_labels(option_experiment, module, with_ci):
    options = ConfidenceIntervalOptions(n_bootstraps=8, bias_correction=False) if with_ci else None
    result = module.analyze(convert(option_experiment), options, CrnOptions())
    _, ax = plt.subplots()
    handle = module.plot(ax, result, "magenta", "D", label="chosen label")
    assert handle.get_label() == "chosen label"
    if not with_ci:
        assert len(ax.collections) == 1
        assert not ax.lines
        return
    assert len(handle.lines[2]) == 2
    if module == auc:
        centers = np.array([[result.mean, result.std]])
        errors = [
            np.array(
                [[max(0, result.mean - result.lb_mean), max(0, result.ub_mean - result.mean)]]
            ),
            np.array([[max(0, result.std - result.lb_std), max(0, result.ub_std - result.std)]]),
        ]
    else:
        centers = result.df[["objective", "feasibility"]].to_numpy()
        errors = [
            np.abs(result.df[["lb_objective", "ub_objective"]].to_numpy() - centers[:, 0, None]),
            np.abs(
                result.df[["lb_feasibility", "ub_feasibility"]].to_numpy() - centers[:, 1, None]
            ),
        ]
    np.testing.assert_allclose(np.asarray(handle.lines[0].get_xdata(), dtype=float), centers[:, 0])
    np.testing.assert_allclose(np.asarray(handle.lines[0].get_ydata(), dtype=float), centers[:, 1])
    assert handle.lines[0].get_marker() == "D"
    assert handle.lines[0].get_color() == "magenta"
    for dimension, collection in enumerate(handle.lines[2]):
        segments = np.array(collection.get_segments())
        expected = np.repeat(centers[:, None, :], 2, axis=1)
        expected[:, 0, dimension] -= errors[dimension][:, 0]
        expected[:, 1, dimension] += errors[dimension][:, 1]
        np.testing.assert_allclose(segments, expected)


CI_CONFIGS = [
    progress_curve.ProgressCurve(agg="mean", normalize=False),
    progress_curve.ProgressCurve(agg="quantile", beta=0.9),
    feasibility_progress.FeasibilityProgress(agg="mean", two_sided=False),
    feasibility_progress.FeasibilityProgress(
        agg="quantile", beta=0.1, score_type="norm", norm_degree=3, two_sided=False
    ),
    solvability_cdf.SolvabilityCdf(normalize=False, solve_tolerance=0.8),
    solvability_profile.SolvabilityProfile(plot_type="cdf", normalize=False, solve_tolerance=0.8),
    solvability_profile.SolvabilityProfile(plot_type="quantile", beta=0.1),
    diff.DiffSolvability(plot_type="cdf", reference_solver="comparison", normalize=False),
    diff.DiffSolvability(plot_type="quantile", reference_solver="RNDSRCH", beta=0.9),
    auc.Auc(),
    terminal_feasibility.TerminalFeasibility(score_type="norm", norm_degree=3, two_sided=False),
    terminal_feasibility.TerminalFeasibility(plot_type="violin"),
]


@pytest.mark.parametrize(
    "config",
    CI_CONFIGS,
    ids=lambda config: (
        type(config).__name__ + getattr(config, "plot_type", getattr(config, "agg", ""))
    ),
)
@pytest.mark.parametrize("bias_correction", [False, True])
@pytest.mark.parametrize("crn_flags", list(product([False, True], repeat=3)))
def test_configuration_ci_forwarding(option_pair, config, bias_correction, crn_flags, monkeypatch):
    from dataclasses import replace

    config = replace(
        config,
        ci_options=ConfidenceIntervalOptions(
            n_bootstraps=3, confidence_level=0.8, bias_correction=bias_correction
        ),
        crn_options=CrnOptions(*crn_flags),
    )
    module = next(module for module in MODULES if config.__class__.__module__ == module.__name__)
    inputs = [convert(experiment) for experiment in option_pair]
    captured = []
    original = cast(Callable[..., list[Any]], module.analyze_many)

    def record_results(*args, **kwargs):
        results = original(*args, **kwargs)
        captured.extend(results)
        return results

    monkeypatch.setattr(module, "analyze_many", record_results)
    _, ax = config.plot(inputs, option_pair)
    # The CI matrix checks the calculations independently; here verify that the
    # configuration wrapper forwards every option to its analysis entry point.
    kwargs = {
        key: value
        for key, value in vars(config).items()
        if key not in ("ci_options", "crn_options", "plot_type", "agg", "reference_solver")
    }
    if module in (progress_curve, feasibility_progress):
        expected = original(inputs, config.agg, config.ci_options, config.crn_options, **kwargs)
    elif module in (solvability_profile, diff):
        args = [inputs, option_pair, config.plot_type]
        if module == diff:
            args.append(config.reference_solver)
        expected = original(*args, config.ci_options, config.crn_options, **kwargs)
    else:
        expected = original(inputs, config.ci_options, config.crn_options, **kwargs)
    assert len(captured) == len(expected)
    for actual, desired in zip(captured, expected, strict=True):
        if module == auc:
            np.testing.assert_allclose(actual.to_numpy(), desired.to_numpy())
        else:
            pd.testing.assert_frame_equal(actual.df, desired.df)
            if hasattr(actual, "ci"):
                pd.testing.assert_frame_equal(actual.ci, desired.ci)
    if module in (progress_curve, feasibility_progress, solvability_cdf, solvability_profile, diff):
        assert len(ax.collections) == len(expected)
        assert len([line for line in ax.lines if line.get_linestyle() == "--"]) == 2 * len(expected)
    elif module == auc or config.plot_type == "scatter":
        assert len(ax.containers) == len(expected)


def test_plot_helper_styles():
    _, ax = plt.subplots()
    line = common.plot_step(
        ax, np.array([0, 1]), np.array([2, 3]), color="cyan", linestyle=":", linewidth=4
    )
    assert line.get_drawstyle() == "steps-post"
    assert line.get_linestyle() == ":"
    assert line.get_linewidth() == 4
    assert line.get_color() == "cyan"
    df = pd.DataFrame({"budget": [0, 1], "lb": [1, 2], "ub": [3, 4]})
    common.plot_ci(ax, df, color="cyan", alpha=0.4)
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_facecolor())[0], to_rgba("cyan", 0.4)
    )
