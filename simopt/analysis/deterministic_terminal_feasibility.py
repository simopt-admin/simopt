"""Analysis and plotting of deterministic terminal feasibility."""

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.container import ErrorbarContainer

from simopt.experiment.api import AnalysisInput, PlotConfig
from simopt.experiment.experiment import Experiment
from simopt.logging import Logger, null_logger
from simopt.options import (
    DEFAULT_CONFIDENCE_INTERVAL_OPTIONS,
    DEFAULT_CRN_OPTIONS,
    ConfidenceIntervalOptions,
    CrnOptions,
)

from .bootstrap import compute_bootstrap_conf_int
from .common import compute_estimator_and_ci
from .merit import deterministic_feasibility_scores
from .terminal_feasibility import _bootstrap


@dataclass(frozen=True)
class DeterministicTerminalFeasibilityResult:
    """Container for deterministic terminal feasibility results.

    Attributes:
        df: DataFrame with the terminal `objective` and `feasibility` of each macroreplication,
            and bootstrap bounds `lb_objective` and `ub_objective` if requested.
    """

    df: pd.DataFrame


def _estimator(
    analysis_input: AnalysisInput, score_type: str, feas_tol: float, floor_feas: bool
) -> tuple[np.ndarray, np.ndarray]:
    # "value" is the raw violation; "norm" can floor it at the feasibility tolerance.
    df = deterministic_feasibility_scores(
        analysis_input,
        "norm",
        0.0,
        np.inf,
        feas_tol,
        floor_feas=floor_feas and score_type == "norm",
    )
    terminal_values = df.groupby("mrep").last()
    return terminal_values["objective"].to_numpy(), terminal_values["score"].to_numpy()


def _ci(
    bootstraps: list[tuple[np.ndarray, np.ndarray]],
    estimator: tuple[np.ndarray, np.ndarray] | None,
    options: ConfidenceIntervalOptions,
) -> pd.DataFrame:
    # Feasibility is deterministic given the solution, so only the objective has a CI.
    n_mreps = len(bootstraps[0][0])
    lower, upper = [], []
    for i in range(n_mreps):
        lb, ub = compute_bootstrap_conf_int(
            [r[0][i] for r in bootstraps],
            options.confidence_level,
            options.bias_correction,
            estimator[0][i] if estimator is not None else None,
        )
        lower.append(lb)
        upper.append(ub)
    return pd.DataFrame({"lb_objective": lower, "ub_objective": upper})


def analyze(
    analysis_input: AnalysisInput,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    score_type: Literal["norm", "value"] = "norm",
    feas_tol: float | None = None,
    feas_tol_lower: float = 1e-8,
    floor_feas: bool = False,
) -> DeterministicTerminalFeasibilityResult:
    """Analyze the deterministic terminal feasibility.

    Args:
        analysis_input: Analysis input with recorded deterministic violations.
        ci_options: Options for computing confidence intervals on the objective, or None.
        crn_options: Options for handling common random numbers.
        score_type: "norm" for the aggregate violation, optionally floored at the tolerance
            by `floor_feas`; "value" for the raw aggregate violation.
        feas_tol: Tolerance for flooring; defaults to `feas_tol_lower`.
        feas_tol_lower: Violation at or below this counts as feasible.
        floor_feas: Whether to raise "norm" scores at or below the tolerance to it.

    Returns:
        DeterministicTerminalFeasibilityResult with the terminal objective and score per macrorep.
    """
    if score_type not in ("norm", "value"):
        raise ValueError("score_type must be 'norm' or 'value'.")
    tol = feas_tol_lower if feas_tol is None else feas_tol
    (objectives, scores), ci = compute_estimator_and_ci(
        analysis_input,
        lambda data: _estimator(data, score_type, tol, floor_feas),
        _ci,
        ci_options,
        crn_options,
        normalize=False,
        bootstrap_fn=_bootstrap,
    )
    result_df = pd.DataFrame({"objective": objectives, "feasibility": scores})
    if ci is not None:
        result_df = pd.concat([result_df, ci], axis=1)
    return DeterministicTerminalFeasibilityResult(df=result_df)


def plot(
    ax: plt.Axes,
    result: DeterministicTerminalFeasibilityResult,
    color: str,
    marker: str,
    label: str | None = None,
    logger: Logger = null_logger,
) -> ErrorbarContainer | PathCollection:
    """Plot the deterministic terminal feasibility result."""
    df = result.df
    objectives = df["objective"]
    feasibility_scores = df["feasibility"]

    if "lb_objective" in df.columns:
        x_err = [
            np.abs(objectives - df["lb_objective"]),
            np.abs(df["ub_objective"] - objectives),
        ]
        logger.debug("data", data=np.array([objectives, feasibility_scores, x_err[0], x_err[1]]))
        return ax.errorbar(
            x=objectives,
            y=feasibility_scores,
            xerr=x_err,
            color=color,
            marker=marker,
            elinewidth=1,
            linestyle="none",
            label=label,
        )
    logger.debug("data", data=np.array([objectives, feasibility_scores]))
    return ax.scatter(x=objectives, y=feasibility_scores, color=color, marker=marker, label=label)


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    score_type: Literal["norm", "value"] = "norm",
    feas_tol: float | None = None,
    feas_tol_lower: float = 1e-8,
    floor_feas: bool = False,
) -> list[DeterministicTerminalFeasibilityResult]:
    """Analyze deterministic terminal feasibility for multiple experiments."""
    return [
        analyze(
            analysis_input,
            ci_options,
            crn_options,
            score_type=score_type,
            feas_tol=feas_tol,
            feas_tol_lower=feas_tol_lower,
            floor_feas=floor_feas,
        )
        for analysis_input in analysis_inputs
    ]


def plot_many(
    results: list[DeterministicTerminalFeasibilityResult],
    experiments: list[Experiment],
    /,
    *,
    plot_zero: bool = True,
    log_scale: bool = False,
    sym_log: bool = False,
    colors: list[str] | None = None,
    markers: list[str] | None = None,
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple deterministic terminal feasibility results."""
    if len(experiments) != len(results):
        raise ValueError("experiments must be the same length as results.")

    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]
    if markers is None:
        markers = ["o", "v", "s", "*", "P", "X", "D", "^", ">", "<"]

    fig, ax = plt.subplots()
    for i, (experiment, result) in enumerate(zip(experiments, results, strict=True)):
        plot(
            ax,
            result,
            colors[i % len(colors)],
            markers[i % len(markers)],
            label=experiment.solver.name,
            logger=logger,
        )
    if plot_zero:
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.75)
    if log_scale:
        ax.set_yscale("log")
    if sym_log:
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Terminal Objective")
    ax.set_ylabel("Feasibility Norm")
    ax.legend()
    return fig, ax


@dataclass(frozen=True)
class DeterministicTerminalFeasibility(PlotConfig):
    """Options for deterministic terminal feasibility analysis."""

    ci_options: ConfidenceIntervalOptions | None = DEFAULT_CONFIDENCE_INTERVAL_OPTIONS
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS
    score_type: Literal["norm", "value"] = "norm"
    feas_tol: float | None = None
    feas_tol_lower: float = 1e-8
    floor_feas: bool = False
    plot_zero: bool = True
    log_scale: bool = False
    sym_log: bool = False

    def plot(
        self,
        analysis_inputs: list[AnalysisInput],
        experiments: list[Experiment],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Analyze and plot deterministic terminal feasibility summaries."""
        results = analyze_many(
            analysis_inputs,
            self.ci_options,
            self.crn_options,
            score_type=self.score_type,
            feas_tol=self.feas_tol,
            feas_tol_lower=self.feas_tol_lower,
            floor_feas=self.floor_feas,
        )
        return plot_many(
            results,
            experiments,
            plot_zero=self.plot_zero,
            log_scale=self.log_scale,
            sym_log=self.sym_log,
        )
