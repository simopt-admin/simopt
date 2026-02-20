"""Analysis and plotting of area under the curve (AUC)."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.container import ErrorbarContainer

from simopt.bootstrap import compute_bootstrap_conf_int
from simopt.experiment.api import AnalysisInput, PlotConfig
from simopt.experiment.single import ProblemSolver
from simopt.logging import Logger, null_logger
from simopt.options import (
    DEFAULT_CONFIDENCE_INTERVAL_OPTIONS,
    DEFAULT_CRN_OPTIONS,
    ConfidenceIntervalOptions,
    CrnOptions,
)

from .common import compute_estimator_and_ci, styled_group_entries


@dataclass(frozen=True)
class AucResult:
    """Result of an AUC analysis."""

    mean: float
    std: float
    lb_mean: float | None = None
    ub_mean: float | None = None
    lb_std: float | None = None
    ub_std: float | None = None

    def to_numpy(self) -> np.ndarray:
        """Convert the AUC result to a NumPy array."""
        return np.array([self.mean, self.std, self.lb_mean, self.ub_mean, self.lb_std, self.ub_std])


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    dx = np.diff(x)
    return float(np.sum(dx * y[:-1]))


def _estimator(analysis_input: AnalysisInput) -> tuple[float, float]:
    aucs = analysis_input.mean_df.groupby("mrep")[["budget", "objective"]].apply(
        lambda x: _auc(x["budget"], x["objective"])
    )
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def _ci(
    bootstraps: list[tuple[float, float]],
    estimator: tuple[float, float] | None,
    options: ConfidenceIntervalOptions,
) -> tuple[float, float, float, float]:
    mean_estimator, std_estimator = estimator or (None, None)

    means = [bootstrap[0] for bootstrap in bootstraps]
    stds = [bootstrap[1] for bootstrap in bootstraps]

    lb_mean, ub_mean = compute_bootstrap_conf_int(
        means,
        options.confidence_level,
        options.bias_correction,
        mean_estimator,
    )
    lb_std, ub_std = compute_bootstrap_conf_int(
        stds,
        options.confidence_level,
        options.bias_correction,
        std_estimator,
    )

    return lb_mean, ub_mean, lb_std, ub_std


def analyze(
    analysis_input: AnalysisInput,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    normalize: bool = True,
) -> AucResult:
    """Analyze the AUC of progress curves.

    Args:
        analysis_input: Analysis input containing the experiment results and references.
        ci_options: Options for computing confidence intervals.
        crn_options: Options for handling common random numbers.
        normalize: Whether to normalize the objectives and budget.

    Returns:
        AucResult: The results of the AUC analysis.
    """
    (mean, std), ci = compute_estimator_and_ci(
        analysis_input, _estimator, _ci, ci_options, crn_options, normalize=normalize
    )

    if ci is None:
        return AucResult(mean=mean, std=std)

    lb_mean, ub_mean, lb_std, ub_std = ci
    return AucResult(
        mean=mean,
        std=std,
        lb_mean=lb_mean,
        ub_mean=ub_mean,
        lb_std=lb_std,
        ub_std=ub_std,
    )


def plot(
    ax: plt.Axes,
    result: AucResult,
    color: str,
    marker: str,
    label: str | None = None,
    logger: Logger = null_logger,
) -> ErrorbarContainer | PathCollection:
    """Plot the AUC result on the given axes.

    Args:
        ax: Matplotlib axes to plot on.
        result: The AUC result to plot.
        color: Color of the plot element.
        marker: Marker style for the plot.
        label: Legend label for the plot element.
        logger: Logger for debugging.

    Returns:
        The created plot container or collection.
    """
    if result.lb_mean is None:
        return ax.scatter(
            x=result.mean,
            y=result.std,
            color=color,
            marker=marker,
            label=label,
        )

    logger.debug("data", data=result.to_numpy())

    # If any error values are negative, set them to zero.
    assert result.lb_mean is not None
    assert result.ub_mean is not None
    assert result.lb_std is not None
    assert result.ub_std is not None
    x_err_low = max(0, result.mean - result.lb_mean)
    x_err_high = max(0, result.ub_mean - result.mean)
    y_err_low = max(0, result.std - result.lb_std)
    y_err_high = max(0, result.ub_std - result.std)

    x_err = [[x_err_low], [x_err_high]]
    y_err = [[y_err_low], [y_err_high]]

    return ax.errorbar(
        x=result.mean,
        y=result.std,
        xerr=x_err,
        yerr=y_err,
        color=color,
        marker=marker,
        elinewidth=1,
        label=label,
    )


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    normalize: bool = True,
) -> list[AucResult]:
    """Analyze AUC for multiple experiments on the same problem."""
    return [
        analyze(analysis_input, ci_options, crn_options, normalize=normalize)
        for analysis_input in analysis_inputs
    ]


def plot_many(
    results: list[AucResult],
    experiments: list[ProblemSolver],
    /,
    *,
    colors: list[str] | None = None,
    markers: list[str] | None = None,
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple AUC results."""
    if len(experiments) != len(results):
        raise ValueError("experiments must be the same length as results.")

    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]
    if markers is None:
        markers = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]

    group = list(zip(experiments, results, strict=True))
    fig, ax = plt.subplots()
    styled_entries, used_labels = styled_group_entries(group, colors, markers)
    for _, result, color, marker, label in styled_entries:
        plot(ax, result, color, marker, label=label, logger=logger)

    ax.set_xlabel("Mean")
    ax.set_ylabel("Standard Deviation")
    if used_labels:
        ax.legend()

    return fig, ax


@dataclass(frozen=True)
class Auc(PlotConfig):
    """Options for AUC analysis."""

    ci_options: ConfidenceIntervalOptions | None = DEFAULT_CONFIDENCE_INTERVAL_OPTIONS
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS

    def plot(
        self,
        analysis_inputs: list[AnalysisInput],
        experiments: list[ProblemSolver],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Analyze and plot AUC results."""
        results = analyze_many(analysis_inputs, self.ci_options, self.crn_options)
        return plot_many(results, experiments)
