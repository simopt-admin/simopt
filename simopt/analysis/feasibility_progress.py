"""Analysis and plotting of feasibility progress."""

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simopt.experiment.api import AnalysisInput, PlotConfig
from simopt.experiment.single import ProblemSolver
from simopt.feasibility import feasibility_score
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
class FeasibilityProgressResult:
    """Container for feasibility progress results.

    Attributes:
        df: DataFrame containing the feasibility scores and confidence intervals.
        agg: Aggregation type ("all", "mean", or "quantile").
    """

    df: pd.DataFrame
    agg: str


def _append_feasibility_score(
    df: pd.DataFrame, score_type: str, norm_degree: int, two_sided: bool
) -> pd.DataFrame:
    df["score"] = df["stochastic_constraints"].apply(
        lambda x: feasibility_score(x, score_type, norm_degree, two_sided)
    )
    return df


def _estimator(
    analysis_input: AnalysisInput,
    agg: str,
    score_type: str,
    norm_degree: int,
    two_sided: bool,
    beta: float,
) -> pd.DataFrame:
    df = analysis_input.mean_df
    df = _append_feasibility_score(df, score_type, norm_degree, two_sided)
    if agg == "mean":
        return aggregate_curve(df, "mean", "score")
    if agg == "quantile":
        return aggregate_curve(df, "quantile", "score", beta=beta)
    raise ValueError("unsupported feasibility progress type.")


def _ci(
    bootstraps: list[pd.DataFrame],
    estimator: pd.DataFrame | None,
    options: ConfidenceIntervalOptions,
) -> pd.DataFrame:
    return compute_ci(bootstraps, estimator, "score", options)


def analyze(
    analysis_input: AnalysisInput,
    agg: str,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    score_type: Literal["inf_norm", "norm"] = "inf_norm",
    norm_degree: int = 1,
    two_sided: bool = True,
    beta: float = 0.5,
) -> FeasibilityProgressResult:
    """Analyze the feasibility progress.

    Args:
        analysis_input: Analysis input containing the experiment results and references.
        agg: Aggregation type ("all", "mean", or "quantile").
        ci_options: Options for computing confidence intervals, or None to skip.
        crn_options: Options for handling common random numbers in bootstraps.
        score_type: Type of feasibility score to use.
        norm_degree: Degree of the norm if score_type is "norm".
        two_sided: Whether the constraints are two-sided.
        beta: Quantile level for quantile plots.

    Returns:
        FeasibilityProgressResult containing the feasibility progress analysis results.
    """
    if agg not in ("all", "mean", "quantile"):
        raise ValueError("agg must be 'all', 'mean', or 'quantile'.")

    if agg != "all" and two_sided:
        raise ValueError("mean and quantile plots not supported for two sided feasibility .")

    df_mean = analysis_input.mean_df
    df_mean = _append_feasibility_score(df_mean, score_type, norm_degree, two_sided)

    if agg == "all":
        df_mean = df_mean.copy()
        df_mean["value"] = df_mean["score"]
        return FeasibilityProgressResult(df=df_mean, agg="all")

    estimator, ci = compute_estimator_and_ci(
        analysis_input,
        lambda data: _estimator(data, agg, score_type, norm_degree, two_sided, beta),
        _ci,
        ci_options,
        crn_options,
        normalize=False,
    )

    if ci is None:
        return FeasibilityProgressResult(df=estimator, agg=agg)

    result = pd.merge(estimator, ci, on="budget", how="outer")
    result[["score", "lb", "ub"]] = result[["score", "lb", "ub"]].ffill()
    result["value"] = result["score"]
    return FeasibilityProgressResult(df=result, agg=agg)


def plot(
    ax: plt.Axes,
    result: FeasibilityProgressResult,
    plot_zero: bool = True,
    color: str = "C0",
    logger: Logger = null_logger,
) -> plt.Line2D:
    """Plot the feasibility progress result.

    Args:
        ax: Matplotlib axes to plot on.
        result: Feasibility progress data and aggregation metadata.
        plot_zero: Whether to plot a horizontal line at zero.
        color: Line and confidence interval color.
        logger: Logger for debugging.

    Returns:
        The created plot artist (Line2D).
    """
    df = result.df

    if result.agg == "all":
        handle = None
        for _, group in df.groupby("mrep"):
            handle = plot_step(ax, group["budget"], group["value"], linewidth=2, color=color)
        if handle is None:
            raise ValueError("no feasibility progress groups found for plotting.")
    else:
        handle = plot_step(ax, df["budget"], df["value"], linewidth=2, color=color)

        if "lb" in df.columns and "ub" in df.columns:
            logger.debug("data", data=np.array([df["budget"], df["value"], df["lb"], df["ub"]]))
            plot_ci(ax, df, color=color)

    if plot_zero:
        ax.axhline(y=0)

    return handle


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    agg: str,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    score_type: Literal["inf_norm", "norm"] = "inf_norm",
    norm_degree: int = 1,
    two_sided: bool = True,
    beta: float = 0.5,
) -> list[FeasibilityProgressResult]:
    """Analyze feasibility progress for multiple experiments on the same problem."""
    return [
        analyze(
            analysis_input,
            agg,
            ci_options,
            crn_options,
            score_type=score_type,
            norm_degree=norm_degree,
            two_sided=two_sided,
            beta=beta,
        )
        for analysis_input in analysis_inputs
    ]


def plot_many(
    results: list[FeasibilityProgressResult],
    experiments: list[ProblemSolver],
    /,
    *,
    plot_zero: bool = True,
    colors: list[str] | None = None,
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple feasibility progress results."""
    if len(experiments) != len(results):
        raise ValueError("experiments must be the same length as results.")

    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]

    fig, ax = plt.subplots()
    for i, (experiment, result) in enumerate(zip(experiments, results, strict=True)):
        color = colors[i % len(colors)]
        handle = plot(ax, result, plot_zero=plot_zero, color=color, logger=logger)
        handle.set_label(experiment.solver.name)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Feasibility Score")
    ax.legend()
    return fig, ax


@dataclass(frozen=True)
class FeasibilityProgress(PlotConfig):
    """Options for feasibility progress analysis."""

    agg: str = "mean"
    ci_options: ConfidenceIntervalOptions | None = DEFAULT_CONFIDENCE_INTERVAL_OPTIONS
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS
    score_type: Literal["inf_norm", "norm"] = "inf_norm"
    norm_degree: int = 1
    two_sided: bool = True
    beta: float = 0.5

    def plot(
        self,
        analysis_inputs: list[AnalysisInput],
        experiments: list[ProblemSolver],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Analyze and plot feasibility progress curves."""
        results = analyze_many(
            analysis_inputs,
            self.agg,
            self.ci_options,
            self.crn_options,
            score_type=self.score_type,
            norm_degree=self.norm_degree,
            two_sided=self.two_sided,
            beta=self.beta,
        )
        return plot_many(results, experiments)
