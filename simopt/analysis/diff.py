"""Analysis and plotting of difference solvability profiles."""

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

from .common import compute_ci, plot_ci, plot_step
from .solvability import (
    cdf,
    group_by_solver,
    mean,
    problem_bootstraps,
    quantile,
    select_df,
    solver_bootstraps,
)


@dataclass(frozen=True)
class DiffSolvabilityResult:
    """Container for difference solvability profile results."""

    df: pd.DataFrame
    ci: pd.DataFrame | None
    label: str
    normalize: bool


def _solvability(
    data: AnalysisInput,
    plot_type: str,
    solve_tolerance: float,
    beta: float,
    normalize: bool,
) -> pd.DataFrame:
    df = select_df(data, normalize)
    if plot_type == "cdf":
        return cdf(df, solve_tolerance)
    if plot_type == "quantile":
        return quantile(df, solve_tolerance, beta, 1.0 if normalize else data.budget)
    raise ValueError(f"invalid plot_type: {plot_type}")


def _aggregate(
    inputs: list[AnalysisInput],
    plot_type: str,
    solve_tolerance: float,
    beta: float,
    normalize: bool,
) -> pd.DataFrame:
    data = [
        _solvability(data, plot_type, solve_tolerance, beta, normalize)
        for data in inputs
    ]
    return mean(data)


def _difference(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    lhs = lhs.rename(columns={"value": "lhs"})
    rhs = rhs.rename(columns={"value": "rhs"})
    df = (
        pd.merge(lhs, rhs, on="budget", how="outer")
        .sort_values("budget")
        .ffill()
        .fillna(0.0)
    )
    return df.assign(value=df["lhs"] - df["rhs"])[["budget", "value"]]


def _analyze_solver(
    inputs: list[AnalysisInput],
    reference_inputs: list[AnalysisInput],
    plot_type: str,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    normalize: bool = True,
    solve_tolerance: float = 0.1,
    beta: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Analyze one solver against a reference solver."""
    estimator = _difference(
        _aggregate(inputs, plot_type, solve_tolerance, beta, normalize),
        _aggregate(reference_inputs, plot_type, solve_tolerance, beta, normalize),
    )

    def _bootstrap(
        solver_inputs: list[AnalysisInput],
        n_bootstraps: int,
    ) -> list[pd.DataFrame]:
        problem_level_bootstraps = problem_bootstraps(
            solver_inputs,
            n_bootstraps,
            lambda data: _solvability(
                data, plot_type, solve_tolerance, beta, normalize
            ),
            crn_options,
        )
        return solver_bootstraps(problem_level_bootstraps, n_bootstraps)

    ci = None
    if ci_options is not None:
        n_bootstraps = ci_options.n_bootstraps
        solver_level_bootstraps = _bootstrap(inputs, n_bootstraps)
        reference_solver_bootstraps = _bootstrap(reference_inputs, n_bootstraps)
        diff_bootstraps = [
            _difference(solver_level_bootstraps[i], reference_solver_bootstraps[i])
            for i in range(n_bootstraps)
        ]
        ci = compute_ci(diff_bootstraps, estimator, "value", ci_options)

    return estimator, ci


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    experiments: list[ProblemSolver],
    plot_type: str,
    reference_solver: str,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    normalize: bool = True,
    solve_tolerance: float = 0.1,
    beta: float = 0.5,
) -> list[DiffSolvabilityResult]:
    """Analyze difference solvability profiles against a reference solver."""
    solvers, solver_to_inputs = group_by_solver(analysis_inputs, experiments)

    if reference_solver not in solver_to_inputs:
        raise ValueError(f"reference solver '{reference_solver}' not found.")

    reference_inputs = solver_to_inputs[reference_solver]
    results: list[DiffSolvabilityResult] = []
    for solver in solvers:
        if solver == reference_solver:
            continue

        estimator, ci = _analyze_solver(
            solver_to_inputs[solver],
            reference_inputs,
            plot_type,
            ci_options,
            crn_options,
            normalize,
            solve_tolerance,
            beta,
        )

        results.append(
            DiffSolvabilityResult(
                df=estimator,
                ci=ci,
                label=f"{solver} - {reference_solver}",
                normalize=normalize,
            )
        )

    return results


def plot(
    ax: plt.Axes,
    result: DiffSolvabilityResult,
    color: str = "C0",
    logger: Logger = null_logger,
) -> plt.Line2D:
    """Plot one difference solvability profile."""
    df = result.df
    handle = plot_step(
        ax, df["budget"].to_numpy(), df["value"].to_numpy(), linewidth=2, color=color
    )
    if result.ci is not None:
        ci = result.ci
        logger.debug(
            "data", data=[df["budget"], df["value"], ci["budget"], ci["lb"], ci["ub"]]
        )
        plot_ci(ax, ci, color=color)
    else:
        logger.debug("data", data=np.array([df["budget"], df["value"]]))
    return handle


def plot_many(
    results: list[DiffSolvabilityResult],
    _experiments: list[ProblemSolver],
    /,
    *,
    colors: list[str] | None = None,
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple difference solvability profiles."""
    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]

    fig, ax = plt.subplots()
    handles = []
    labels = []
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        handles.append(plot(ax, result, color=color, logger=logger))
        labels.append(result.label)
    if handles:
        ax.legend(handles=handles, labels=labels)

    ax.set_xlabel("Budget")
    ax.set_ylabel("Difference")

    return fig, ax


@dataclass(frozen=True)
class DiffSolvability(PlotConfig):
    """Options for difference solvability profile analysis."""

    plot_type: str = "cdf"
    reference_solver: str = ""
    ci_options: ConfidenceIntervalOptions | None = DEFAULT_CONFIDENCE_INTERVAL_OPTIONS
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS
    normalize: bool = True
    solve_tolerance: float = 0.1
    beta: float = 0.5

    def plot(
        self,
        analysis_inputs: list[AnalysisInput],
        experiments: list[ProblemSolver],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Analyze and plot difference solvability profiles."""
        results = analyze_many(
            analysis_inputs,
            experiments,
            self.plot_type,
            self.reference_solver,
            self.ci_options,
            self.crn_options,
            normalize=self.normalize,
            solve_tolerance=self.solve_tolerance,
            beta=self.beta,
        )
        return plot_many(results, experiments)
