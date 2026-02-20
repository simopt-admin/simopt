"""Analysis and plotting of terminal feasibility."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.container import ErrorbarContainer

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.bootstrap import compute_bootstrap_conf_int
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

from .bootstrap import _bootstrap_sample, _get_n_preps, _transform_bootstrap_data
from .common import compute_estimator_and_ci, plot_distribution


@dataclass(frozen=True)
class TerminalFeasibilityResult:
    """Container for terminal feasibility results.

    Attributes:
        df: DataFrame containing the terminal feasibility data and confidence intervals.
    """

    df: pd.DataFrame


def _bootstrap(
    analysis_input: AnalysisInput,
    n_bootstraps: int,
    f: Callable[[AnalysisInput], tuple[np.ndarray, np.ndarray]],
    crn_options: CrnOptions,
) -> list[tuple[np.ndarray, np.ndarray]]:
    df = analysis_input.full_df
    n_preps = _get_n_preps(df)
    data = _transform_bootstrap_data(df, n_preps)
    rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
    results = []
    for _ in range(n_bootstraps):
        bootstrap_input = _bootstrap_sample(
            analysis_input,
            data,
            rng,
            crn_options.across_budget,
            crn_options.across_macroreps,
            crn_options.across_x0_xstar,
            True,
        )
        results.append(f(bootstrap_input))
    return results


def _estimator(
    analysis_input: AnalysisInput,
    score_type: Literal["inf_norm", "norm"],
    norm_degree: int,
    two_sided: bool,
) -> tuple[np.ndarray, np.ndarray]:
    df = analysis_input.mean_df
    df["score"] = df["stochastic_constraints"].apply(
        lambda x: feasibility_score(x, score_type, norm_degree, two_sided)
    )
    terminal_values = df.groupby("mrep").last()
    objectives = terminal_values["objective"]
    feasibility_scores = terminal_values["score"]
    return objectives, feasibility_scores


def _ci(
    bootstraps: list[tuple[np.ndarray, np.ndarray]],
    estimator: tuple[np.ndarray, np.ndarray] | None,
    options: ConfidenceIntervalOptions,
) -> pd.DataFrame:
    n_mreps = len(bootstraps[0][0])
    if estimator is None:
        objectives, feasibility_scores = None, None
    else:
        objectives, feasibility_scores = estimator

    lb_objective_values = []
    ub_objective_values = []
    lb_feasibility_values = []
    ub_feasibility_values = []
    for i in range(n_mreps):
        bootstrap_objectives = [r[0][i] for r in bootstraps]
        bootstrap_feasibility_scores = [r[1][i] for r in bootstraps]
        lb_objective, ub_objective = compute_bootstrap_conf_int(
            bootstrap_objectives,
            options.confidence_level,
            options.bias_correction,
            objectives[i] if objectives is not None else None,
        )
        lb_feasibility, ub_feasibility = compute_bootstrap_conf_int(
            bootstrap_feasibility_scores,
            options.confidence_level,
            options.bias_correction,
            feasibility_scores[i] if feasibility_scores is not None else None,
        )
        lb_objective_values.append(lb_objective)
        ub_objective_values.append(ub_objective)
        lb_feasibility_values.append(lb_feasibility)
        ub_feasibility_values.append(ub_feasibility)

    return pd.DataFrame(
        {
            "lb_objective": lb_objective_values,
            "ub_objective": ub_objective_values,
            "lb_feasibility": lb_feasibility_values,
            "ub_feasibility": ub_feasibility_values,
        }
    )


def analyze(
    analysis_input: AnalysisInput,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    score_type: Literal["inf_norm", "norm"] = "inf_norm",
    norm_degree: int = 1,
    two_sided: bool = True,
) -> TerminalFeasibilityResult:
    """Analyze the terminal feasibility.

    Args:
        analysis_input: Analysis input containing the experiment results and references.
        ci_options: Options for computing confidence intervals.
        crn_options: Options for handling common random numbers.
        score_type: Type of feasibility score to use.
        norm_degree: Degree of the norm if score_type is "norm".
        two_sided: Whether the constraints are two-sided.

    Returns:
        TerminalFeasibilityResult containing terminal feasibility analysis results.
    """
    (objectives, feasibility_scores), ci = compute_estimator_and_ci(
        analysis_input,
        lambda data: _estimator(data, score_type, norm_degree, two_sided),
        _ci,
        ci_options,
        crn_options,
        normalize=False,
        bootstrap_fn=_bootstrap,
    )

    result_df = pd.DataFrame({"objective": objectives, "feasibility": feasibility_scores})
    if ci is not None:
        result_df = pd.concat([result_df.reset_index(drop=True), ci], axis=1)

    return TerminalFeasibilityResult(df=result_df)


def plot(
    ax: plt.Axes,
    result: TerminalFeasibilityResult,
    color: str,
    marker: str,
    label: str | None = None,
    logger: Logger = null_logger,
) -> ErrorbarContainer | PathCollection:
    """Plot the terminal feasibility result.

    Args:
        ax: Matplotlib axes to plot on.
        result: Terminal feasibility data and normalization metadata.
        color: Color of the plot element.
        marker: Marker style for the plot.
        label: Legend label for the plotted series.
        logger: Logger for debugging.

    Returns:
        The created plot container or collection.
    """
    df = result.df
    objectives = df["objective"]
    feasibility_scores = df["feasibility"]

    if "lb_objective" in df.columns:
        x_err = [
            np.abs(objectives - df["lb_objective"]),
            np.abs(df["ub_objective"] - objectives),
        ]
        y_err = [
            np.abs(feasibility_scores - df["lb_feasibility"]),
            np.abs(df["ub_feasibility"] - feasibility_scores),
        ]
        logger.debug(
            "data",
            data=np.array([objectives, feasibility_scores, x_err[0], x_err[1], y_err[0], y_err[1]]),
        )
        return ax.errorbar(
            x=objectives,
            y=feasibility_scores,
            xerr=x_err,
            yerr=y_err,
            color=color,
            marker=marker,
            linestyle="none",
            label=label,
        )
    logger.debug("data", data=np.array([objectives, feasibility_scores]))
    return ax.scatter(
        x=objectives,
        y=feasibility_scores,
        color=color,
        marker=marker,
        label=label,
    )


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    score_type: Literal["inf_norm", "norm"] = "inf_norm",
    norm_degree: int = 1,
    two_sided: bool = True,
) -> list[TerminalFeasibilityResult]:
    """Analyze terminal feasibility for multiple experiments on the same problem."""
    return [
        analyze(
            analysis_input,
            ci_options,
            crn_options,
            score_type=score_type,
            norm_degree=norm_degree,
            two_sided=two_sided,
        )
        for analysis_input in analysis_inputs
    ]


def plot_many(
    results: list[TerminalFeasibilityResult],
    experiments: list[ProblemSolver],
    /,
    *,
    plot_type: Literal["scatter", "violin"] = "scatter",
    colors: list[str] | None = None,
    markers: list[str] | None = None,
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple terminal feasibility results."""
    if len(experiments) != len(results):
        raise ValueError("experiments must be the same length as results.")

    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]
    if markers is None:
        markers = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]

    fig, ax = plt.subplots()

    if plot_type == "violin":
        records = []
        for experiment, result in zip(experiments, results, strict=True):
            for value in result.df["feasibility"]:
                records.append(
                    {
                        "Solver": experiment.solver.name,
                        "Feasibility Score": float(value),
                    }
                )
        plot_distribution(ax, records, "Solver", "Feasibility Score", "violin")
    else:
        for i, (experiment, result) in enumerate(zip(experiments, results, strict=True)):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            label = f"{experiment.solver.name} | {experiment.problem.name}"
            plot(
                ax,
                result,
                color,
                marker,
                label=label,
                logger=logger,
            )
        ax.set_xlabel("Objective")
        ax.set_ylabel("Feasibility Score")
        ax.legend()

    return fig, ax


@dataclass(frozen=True)
class TerminalFeasibility(PlotConfig):
    """Options for terminal feasibility analysis."""

    plot_type: Literal["scatter", "violin"] = "scatter"
    ci_options: ConfidenceIntervalOptions | None = DEFAULT_CONFIDENCE_INTERVAL_OPTIONS
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS
    score_type: Literal["inf_norm", "norm"] = "inf_norm"
    norm_degree: int = 1
    two_sided: bool = True

    def plot(
        self,
        analysis_inputs: list[AnalysisInput],
        experiments: list[ProblemSolver],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Analyze and plot terminal feasibility summaries."""
        results = analyze_many(
            analysis_inputs,
            self.ci_options,
            self.crn_options,
            score_type=self.score_type,
            norm_degree=self.norm_degree,
            two_sided=self.two_sided,
        )
        return plot_many(results, experiments, plot_type=self.plot_type)
