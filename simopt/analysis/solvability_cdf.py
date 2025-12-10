"""Analysis and plotting of solvability CDFs."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
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
    compute_ci,
    compute_estimator_and_ci,
    plot_ci,
    plot_step,
)


@dataclass(frozen=True)
class SolvabilityCdfResult:
    """Container for solvability CDF results.

    Attributes:
        df: DataFrame containing the solvability CDF data and confidence intervals.
        normalize: Whether the data is normalized.
    """

    df: pd.DataFrame
    normalize: bool


def _estimator(analysis_input: AnalysisInput, solve_tolerance: float) -> pd.DataFrame:
    df = analysis_input.mean_df
    group_column = "mrep"

    solved = df.loc[df["objective"] < solve_tolerance, [group_column, "budget"]]
    n = df[group_column].nunique()

    first_hit = solved.groupby(group_column)["budget"].min().reset_index()
    counts = (
        first_hit.groupby("budget")[group_column]
        .size()
        .rename("cdf")
        .to_frame()
        .sort_index()
    )
    counts["cdf"] = counts["cdf"].cumsum() / n

    budget_points = [0.0, *counts.index, 1.0]
    counts = counts.reindex(budget_points).ffill().fillna(0.0)
    return counts.reset_index()


def _ci(
    bootstraps: list[pd.DataFrame],
    estimator: pd.DataFrame | None,
    options: ConfidenceIntervalOptions,
) -> pd.DataFrame:
    return compute_ci(bootstraps, estimator, "cdf", options)


def analyze(
    analysis_input: AnalysisInput,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    normalize: bool = True,
    solve_tolerance: float = 0.1,
) -> SolvabilityCdfResult:
    """Analyze the solvability CDF.

    Args:
        analysis_input: Analysis input containing the experiment results and references.
        ci_options: Options for computing confidence intervals.
        crn_options: Options for handling common random numbers.
        solve_tolerance: Tolerance for considering a problem solved.
        normalize: Whether to normalize the objectives and budget.

    Returns:
        SolvabilityCdfResult containing the analysis results.
    """
    estimator, ci = compute_estimator_and_ci(
        analysis_input,
        lambda data: _estimator(data, solve_tolerance),
        _ci,
        ci_options,
        crn_options,
        normalize=normalize,
    )

    if ci is None:
        return SolvabilityCdfResult(df=estimator, normalize=normalize)

    result = pd.merge(estimator, ci, on="budget", how="outer")
    result[["cdf", "lb", "ub"]] = result[["cdf", "lb", "ub"]].ffill()
    result["value"] = result["cdf"]
    return SolvabilityCdfResult(df=result, normalize=normalize)


def plot(
    ax: plt.Axes,
    result: SolvabilityCdfResult,
    color: str = "C0",
    logger: Logger = null_logger,
) -> plt.Line2D:
    """Plot the solvability CDF result.

    Args:
        ax: Matplotlib axes to plot on.
        result: Solvability CDF analysis result.
        color: Line and confidence interval color.
        logger: Logger for debugging.

    Returns:
        The created plot artist.
    """
    df = result.df

    handle = plot_step(ax, df["budget"], df["value"], linewidth=2, color=color)

    if "lb" in df.columns and "ub" in df.columns:
        logger.debug("data", data=[df["budget"], df["lb"], df["ub"]])
        plot_ci(ax, df, color=color)

    return handle


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    normalize: bool = True,
    solve_tolerance: float = 0.1,
) -> list[SolvabilityCdfResult]:
    """Analyze solvability CDFs for multiple experiments on the same problem."""
    return [
        analyze(
            analysis_input,
            ci_options,
            crn_options,
            normalize=normalize,
            solve_tolerance=solve_tolerance,
        )
        for analysis_input in analysis_inputs
    ]


def plot_many(
    results: list[SolvabilityCdfResult],
    experiments: list[ProblemSolver],
    /,
    *,
    colors: list[str] | None = None,
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple solvability CDF results."""
    if len(experiments) != len(results):
        raise ValueError("experiments must be the same length as results.")

    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]

    fig, ax = plt.subplots()
    for i, (_, result) in enumerate(zip(experiments, results, strict=True)):
        color = colors[i % len(colors)]
        plot(ax, result, color=color, logger=logger)
    return fig, ax


@dataclass(frozen=True)
class SolvabilityCdf(PlotConfig):
    """Options for solvability CDF analysis."""

    ci_options: ConfidenceIntervalOptions | None = DEFAULT_CONFIDENCE_INTERVAL_OPTIONS
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS
    normalize: bool = True
    solve_tolerance: float = 0.1

    def plot(
        self,
        analysis_inputs: list[AnalysisInput],
        experiments: list[ProblemSolver],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Analyze and plot solvability CDF curves."""
        results = analyze_many(
            analysis_inputs,
            self.ci_options,
            self.crn_options,
            normalize=self.normalize,
            solve_tolerance=self.solve_tolerance,
        )
        return plot_many(results, experiments)
