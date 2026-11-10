"""Analysis and plotting of deterministic feasibility progress."""

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simopt.experiment.api import AnalysisInput, PlotConfig
from simopt.experiment.experiment import Experiment
from simopt.logging import Logger, null_logger

from .common import plot_step
from .merit import deterministic_feasibility_scores


@dataclass(frozen=True)
class DeterministicFeasibilityProgressResult:
    """Container for deterministic feasibility progress results.

    Attributes:
        df: DataFrame with one row per (mrep, step) and the feasibility `score` in `value`.
        score_type: "norm" or "objective".
    """

    df: pd.DataFrame
    score_type: str


def analyze(
    analysis_input: AnalysisInput,
    /,
    *,
    score_type: Literal["norm", "objective"] = "norm",
    obj_const: float = 100.0,
    feas_tol_upper: float = 1e-5,
    feas_tol_lower: float = 1e-8,
    floor_feas: bool = False,
) -> DeterministicFeasibilityProgressResult:
    """Analyze the deterministic feasibility progress of every macroreplication.

    Args:
        analysis_input: Analysis input with recorded deterministic violations.
        score_type: "norm" for the aggregate constraint violation, or "objective" for the
            objective plus `obj_const` times the violation.
        obj_const: Penalty multiplier on the violation for "objective".
        feas_tol_upper: Violation above this is infinitely penalized for "objective".
        feas_tol_lower: Violation at or below this counts as feasible.
        floor_feas: For "norm", raise violations at or below `feas_tol_lower` to it.

    Returns:
        DeterministicFeasibilityProgressResult with the score of each macroreplication and step.
    """
    df = deterministic_feasibility_scores(
        analysis_input, score_type, obj_const, feas_tol_upper, feas_tol_lower, floor_feas
    )
    df["value"] = df["score"]
    return DeterministicFeasibilityProgressResult(df=df, score_type=score_type)


def plot(
    ax: plt.Axes,
    result: DeterministicFeasibilityProgressResult,
    plot_zero: bool = False,
    color: str = "C0",
    logger: Logger = null_logger,
) -> plt.Line2D:
    """Plot the feasibility progress of all macroreplications.

    Args:
        ax: Matplotlib axes to plot on.
        result: Deterministic feasibility progress data.
        plot_zero: Whether to plot a horizontal line at zero.
        color: Line color.
        logger: Logger for debugging.

    Returns:
        The artist of the first macroreplication.
    """
    handle = None
    for _, group in result.df.groupby("mrep"):
        logger.debug("data", data=np.array([group["budget"], group["value"]]))
        line = plot_step(ax, group["budget"], group["value"], linewidth=2, color=color)
        if handle is None:
            handle = line
    if handle is None:
        raise ValueError("no feasibility progress groups found for plotting.")

    if plot_zero:
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.75)

    return handle


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    /,
    *,
    score_type: Literal["norm", "objective"] = "norm",
    obj_const: float = 100.0,
    feas_tol_upper: float = 1e-5,
    feas_tol_lower: float = 1e-8,
    floor_feas: bool = False,
) -> list[DeterministicFeasibilityProgressResult]:
    """Analyze deterministic feasibility progress for multiple experiments."""
    return [
        analyze(
            analysis_input,
            score_type=score_type,
            obj_const=obj_const,
            feas_tol_upper=feas_tol_upper,
            feas_tol_lower=feas_tol_lower,
            floor_feas=floor_feas,
        )
        for analysis_input in analysis_inputs
    ]


def plot_many(
    results: list[DeterministicFeasibilityProgressResult],
    experiments: list[Experiment],
    /,
    *,
    plot_zero: bool = False,
    log_scale: bool = False,
    log_base: int = 10,
    sym_log: bool = False,
    colors: list[str] | None = None,
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple deterministic feasibility progress results."""
    if len(experiments) != len(results):
        raise ValueError("experiments must be the same length as results.")

    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]

    fig, ax = plt.subplots()
    for i, (experiment, result) in enumerate(zip(experiments, results, strict=True)):
        handle = plot(ax, result, plot_zero=plot_zero, color=colors[i % len(colors)], logger=logger)
        handle.set_label(experiment.solver.name)
    if log_scale:
        ax.set_yscale("log", base=log_base)
    if sym_log:
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Budget")
    score_types = {result.score_type for result in results}
    ax.set_ylabel("Merit" if score_types == {"objective"} else "Feasibility Norm")
    ax.legend()
    return fig, ax


@dataclass(frozen=True)
class DeterministicFeasibilityProgress(PlotConfig):
    """Options for deterministic feasibility progress analysis."""

    score_type: Literal["norm", "objective"] = "norm"
    obj_const: float = 100.0
    feas_tol_upper: float = 1e-5
    feas_tol_lower: float = 1e-8
    floor_feas: bool = False
    log_scale: bool = False
    log_base: int = 10
    sym_log: bool = False
    plot_zero: bool = False

    def plot(
        self,
        analysis_inputs: list[AnalysisInput],
        experiments: list[Experiment],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Analyze and plot deterministic feasibility progress curves."""
        results = analyze_many(
            analysis_inputs,
            score_type=self.score_type,
            obj_const=self.obj_const,
            feas_tol_upper=self.feas_tol_upper,
            feas_tol_lower=self.feas_tol_lower,
            floor_feas=self.floor_feas,
        )
        return plot_many(
            results,
            experiments,
            plot_zero=self.plot_zero,
            log_scale=self.log_scale,
            log_base=self.log_base,
            sym_log=self.sym_log,
        )
