"""Analysis and plotting of progress curves."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simopt.experiment.api import AnalysisInput, PlotConfig
from simopt.experiment.single import ProblemSolver
from simopt.logging import Logger, null_logger
from simopt.options import (
    DEFAULT_CONFIDENCE_INTERVAL_OPTIONS,
    DEFAULT_CRN_OPTIONS,
    ConfidenceIntervalOptions,
    CrnOptions,
)

from .common import (
    aggregate_curve,
    compute_ci,
    compute_estimator_and_ci,
    plot_ci,
    plot_step,
)


@dataclass(frozen=True)
class ProgressCurveResult:
    """Container for progress curve results.

    Attributes:
        df: DataFrame containing the progress curve data and confidence intervals.
        agg: Aggregation type ('all', 'mean', or 'quantile').
    """

    df: pd.DataFrame
    agg: str


def _estimator(
    analysis_input: AnalysisInput,
    agg: str,
    beta: float,
) -> pd.DataFrame:
    df = analysis_input.mean_df
    if agg == "mean":
        return aggregate_curve(df, "mean", "objective")
    if agg == "quantile":
        return aggregate_curve(df, "quantile", "objective", beta=beta)
    raise ValueError("unsupported progress curve type.")


def _ci(
    bootstraps: list[pd.DataFrame],
    estimator: pd.DataFrame | None,
    options: ConfidenceIntervalOptions,
) -> pd.DataFrame:
    return compute_ci(bootstraps, estimator, "objective", options)


def analyze(
    analysis_input: AnalysisInput,
    agg: str,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    normalize: bool = True,
    beta: float = 0.5,
) -> ProgressCurveResult:
    """Analyze the progress curves.

    Args:
        analysis_input: Analysis input containing the experiment results and references.
        agg: Aggregation type ('all', 'mean', or 'quantile').
        ci_options: Options for computing confidence intervals, or None to skip.
        crn_options: Options for handling common random numbers in bootstraps.
        normalize: Whether to normalize the objectives and budget.
        beta: Quantile level for quantile plots.

    Returns:
        ProgressCurveResult containing the progress curve analysis results.
    """
    df = analysis_input.mean_df
    if agg == "all":
        df = df.copy()
        column = "normalized_objective" if normalize else "objective"
        df["value"] = df[column]
        return ProgressCurveResult(df=df, agg="all")

    estimator = _estimator(analysis_input, agg, beta)

    estimator, ci = compute_estimator_and_ci(
        analysis_input,
        lambda data: _estimator(data, agg, beta),
        _ci,
        ci_options,
        crn_options,
        normalize=normalize,
    )

    if ci is None:
        return ProgressCurveResult(df=estimator, agg=agg)

    result = pd.merge(estimator, ci, on="budget", how="outer")
    result[["objective", "lb", "ub"]] = result[["objective", "lb", "ub"]].ffill()
    result["value"] = result["objective"]
    return ProgressCurveResult(df=result, agg=agg)


def plot(
    ax: plt.Axes,
    result: ProgressCurveResult,
    color: str = "C0",
    logger: Logger = null_logger,
) -> plt.Line2D:
    """Plot the progress curve result.

    Args:
        ax: Matplotlib axes to plot on.
        result: Progress curve data and aggregation metadata.
        color: Line and confidence interval color.
        logger: Logger for debugging.

    Returns:
        The created plot artist, or axes when multiple artists are drawn.
    """
    df = result.df

    if result.agg == "all":
        handle = None
        for _, group in df.groupby("mrep"):
            handle = plot_step(ax, group["budget"], group["value"], linewidth=2, color=color)
        if handle is None:
            raise ValueError("no progress curve created.")
        return handle

    handle = plot_step(ax, df["budget"], df["value"], linewidth=2, color=color)

    if "lb" in df.columns and "ub" in df.columns:
        logger.debug("data", data=np.array([df["budget"], df["value"], df["lb"], df["ub"]]))

        plot_ci(ax, df, color=color)

    return handle


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    agg: str,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    normalize: bool = True,
    beta: float = 0.5,
) -> list[ProgressCurveResult]:
    """Analyze progress curves for multiple experiments on the same problem."""
    return [
        analyze(
            analysis_input,
            agg,
            ci_options,
            crn_options,
            normalize=normalize,
            beta=beta,
        )
        for analysis_input in analysis_inputs
    ]


def plot_many(
    results: list[ProgressCurveResult],
    experiments: list[ProblemSolver],
    /,
    *,
    colors: list[str] | None = None,
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple progress curve results."""
    if len(experiments) != len(results):
        raise ValueError("experiments must be the same length as results.")

    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]

    fig, ax = plt.subplots()
    for i, (experiment, result) in enumerate(zip(experiments, results, strict=True)):
        color = colors[i % len(colors)]
        handle = plot(ax, result, color=color, logger=logger)
        handle.set_label(experiment.solver.name)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Objective")
    ax.legend()
    return fig, ax


@dataclass(frozen=True)
class ProgressCurve(PlotConfig):
    """Options for progress curve analysis."""

    agg: str = "mean"
    ci_options: ConfidenceIntervalOptions | None = DEFAULT_CONFIDENCE_INTERVAL_OPTIONS
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS
    normalize: bool = True
    beta: float = 0.5

    def plot(
        self,
        analysis_inputs: list[AnalysisInput],
        experiments: list[ProblemSolver],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Analyze and plot progress curves."""
        results = analyze_many(
            analysis_inputs,
            self.agg,
            self.ci_options,
            self.crn_options,
            normalize=self.normalize,
            beta=self.beta,
        )
        return plot_many(results, experiments)
