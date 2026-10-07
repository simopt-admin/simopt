import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

from simopt.analysis import auc, feasibility_progress, progress_curve, solvability_cdf
from simopt.experiment.api import AnalysisInput
from simopt.options import ConfidenceIntervalOptions, CrnOptions


@pytest.fixture
def analysis_input():
    df = pd.DataFrame(
        {
            "mrep": [0, 0, 0, 1, 1, 1],
            "step": [0, 1, 2, 0, 1, 2],
            "budget": [0, 5, 10, 0, 5, 10],
            "solution": [(0,), (1,), (2,), (0,), (1,), (2,)],
            "objective": [10.0, 6.0, 2.0, 10.0, 4.0, 2.0],
            "stochastic_constraints": [
                np.array([value]) for value in [-2.0, 1.0, 0.0, -1.0, 3.0, 0.0]
            ],
        }
    )
    return AnalysisInput(
        df,
        np.array([0]),
        np.array([10.0]),
        np.array([2]),
        np.array([2.0]),
        skip_aggregation=True,
    )


@pytest.mark.parametrize("agg", ["all", "mean", "quantile"])
@pytest.mark.parametrize("normalize", [False, True])
def test_progress_without_ci(analysis_input, agg, normalize):
    original = analysis_input.mean_df.copy(deep=True)
    result = progress_curve.analyze(analysis_input, agg, None, CrnOptions(), normalize=normalize)
    budget = [0.0, 0.5, 1.0] if normalize else [0.0, 5.0, 10.0]
    values = (
        [[1.0, 0.5, 0.0], [1.0, 0.25, 0.0]] if normalize else [[10.0, 6.0, 2.0], [10.0, 4.0, 2.0]]
    )
    fig, ax = plt.subplots()
    try:
        progress_curve.plot(ax, result)
        expected = values if agg == "all" else [np.mean(values, axis=0)]
        assert len(ax.lines) == len(expected)
        for line, value in zip(ax.lines, expected, strict=True):
            np.testing.assert_allclose(np.asarray(line.get_xdata()), budget)
            np.testing.assert_allclose(np.asarray(line.get_ydata()), value)
        assert "lb" not in result.df and "ub" not in result.df
        pd.testing.assert_frame_equal(analysis_input.mean_df, original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("agg", ["all", "mean", "quantile"])
def test_feasibility_without_ci(analysis_input, agg):
    original = analysis_input.mean_df.copy(deep=True)
    result = feasibility_progress.analyze(analysis_input, agg, None, CrnOptions(), two_sided=False)
    fig, ax = plt.subplots()
    try:
        feasibility_progress.plot(ax, result, plot_zero=False)
        expected = [[0, -1, 0], [0, -3, 0]] if agg == "all" else [[0, -2, 0]]
        assert len(ax.lines) == len(expected)
        for line, value in zip(ax.lines, expected, strict=True):
            np.testing.assert_allclose(np.asarray(line.get_xdata()), [0, 5, 10])
            np.testing.assert_allclose(np.asarray(line.get_ydata()), value)
        assert "lb" not in result.df and "ub" not in result.df
        pd.testing.assert_frame_equal(analysis_input.mean_df, original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("normalize", [False, True])
def test_auc_without_ci(analysis_input, normalize):
    result = auc.analyze(analysis_input, None, CrnOptions(), normalize=normalize)
    areas = [0.75, 0.625] if normalize else [80.0, 70.0]
    assert result.mean == pytest.approx(np.mean(areas))
    assert result.std == pytest.approx(np.std(areas, ddof=1))
    assert result.lb_mean is None and result.ub_mean is None
    assert result.lb_std is None and result.ub_std is None
    fig, ax = plt.subplots()
    try:
        handle = auc.plot(ax, result, "C0", "o")
        assert isinstance(handle, PathCollection)
        np.testing.assert_allclose(np.asarray(handle.get_offsets()), [[result.mean, result.std]])
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("solve_tolerance", "budget", "cdf"),
    [(0.0, [0, 1], [0, 0]), (0.5, [0, 0.5, 1, 1], [0, 0.5, 1, 1])],
)
def test_solvability_without_ci(analysis_input, solve_tolerance, budget, cdf):
    result = solvability_cdf.analyze(
        analysis_input, None, CrnOptions(), solve_tolerance=solve_tolerance
    )
    assert result.ci is None
    np.testing.assert_allclose(result.df["budget"], budget)
    np.testing.assert_allclose(result.df["cdf"], cdf)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (ConfidenceIntervalOptions(n_bootstraps=0), "bootstraps"),
        (ConfidenceIntervalOptions(n_bootstraps=-1), "bootstraps"),
        (ConfidenceIntervalOptions(confidence_level=0), "Confidence level"),
        (ConfidenceIntervalOptions(confidence_level=1), "Confidence level"),
    ],
)
def test_invalid_ci_options(analysis_input, options, message):
    with pytest.raises(ValueError, match=message):
        progress_curve.analyze(analysis_input, "mean", options, CrnOptions())
