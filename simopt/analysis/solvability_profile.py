"""Analysis and plotting of solvability profiles."""

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
class SolvabilityProfileResult:
    """Container for solvability profile results.

    Attributes:
        df: DataFrame containing the solvability profile data and confidence intervals.
        normalize: Whether the data is normalized.
    """

    df: pd.DataFrame
    normalize: bool


def _estimator(
    analysis_input: AnalysisInput,
    plot_type: str,
    solve_tolerance: float,
    beta: float,
    budget: float,
    # TODO: check better way to unify _select_df and _select_input in
    # simopt.analysis.common
    use_normalized: bool = False,
) -> pd.DataFrame:
    df = select_df(analysis_input, use_normalized)
    if plot_type == "cdf":
        return cdf(df, solve_tolerance)
    if plot_type == "quantile":
        return quantile(df, solve_tolerance, beta, budget)
    raise ValueError("unsupported plot type.")


def _ci(
    bootstraps: list[pd.DataFrame],
    estimator: pd.DataFrame | None,
    options: ConfidenceIntervalOptions,
) -> pd.DataFrame:
    return compute_ci(bootstraps, estimator, "value", options)


def analyze(
    analysis_input: AnalysisInput,
    plot_type: str,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    normalize: bool = True,
    solve_tolerance: float = 0.1,
    beta: float = 0.5,
) -> SolvabilityProfileResult:
    """Analyze the solvability profile.

    Args:
        analysis_input: Analysis input containing the experiment results and references.
        plot_type: "cdf" or "quantile".
        solve_tolerance: Tolerance for considering a problem solved.
        beta: Quantile level for quantile plots.
        normalize: Whether to normalize the objectives and budget.
        ci_options: Options for computing confidence intervals, or None to skip.
        crn_options: Options for handling common random numbers in bootstraps.

    Returns:
        SolvabilityProfileResult containing the solvability profile results.
    """
    if plot_type not in ("cdf", "quantile"):
        raise ValueError("Unsupported solvability profile plot type.")
    if plot_type == "quantile" and not 0 < beta < 1:
        raise ValueError("Beta quantile must be in (0, 1).")
    if not 0 < solve_tolerance <= 1:
        raise ValueError("Solve tolerance must be in (0, 1].")

    budget = analysis_input.budget if not normalize else 1.0

    estimator, ci = compute_estimator_and_ci(
        analysis_input,
        lambda data: _estimator(data, plot_type, solve_tolerance, beta, budget),
        _ci,
        ci_options,
        crn_options,
        normalize=normalize,
    )

    if ci is None:
        return SolvabilityProfileResult(df=estimator, normalize=normalize)

    df = pd.merge(estimator, ci, on="budget", how="outer")
    df[["value", "lb", "ub"]] = df[["value", "lb", "ub"]].ffill()
    return SolvabilityProfileResult(df=df, normalize=normalize)


def plot(
    ax: plt.Axes,
    result: SolvabilityProfileResult,
    color: str = "C0",
    logger: Logger = null_logger,
) -> plt.Line2D:
    """Plot the solvability profile result.

    Args:
        ax: Matplotlib axes to plot on.
        result: Solvability profile data and metadata.
        color: Line and confidence interval color.
        logger: Logger for debugging.

    Returns:
        The created plot artist.
    """
    df = result.df
    handle = plot_step(
        ax, df["budget"].to_numpy(), df["value"].to_numpy(), linewidth=2, color=color
    )

    if "lb" in df.columns and "ub" in df.columns:
        logger.debug("data", data=[df["budget"], df["lb"], df["ub"]])
        plot_ci(ax, df, color=color)

    return handle


def _analyze_solver(
    inputs: list[AnalysisInput],
    plot_type: str,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    normalize: bool,
    solve_tolerance: float,
    beta: float,
) -> SolvabilityProfileResult:
    """Analyze one solver over multiple problem inputs."""
    problem_data = [
        _estimator(
            analysis_input,
            plot_type,
            solve_tolerance,
            beta,
            analysis_input.budget if not normalize else 1.0,
            use_normalized=normalize,
        )
        for analysis_input in inputs
    ]
    estimator = mean(problem_data)

    ci = None
    if ci_options is not None:
        problem_level_bootstraps = problem_bootstraps(
            inputs,
            ci_options.n_bootstraps,
            lambda data: _estimator(
                data,
                plot_type,
                solve_tolerance,
                beta,
                data.budget if not normalize else 1.0,
                use_normalized=normalize,
            ),
            crn_options,
        )
        solver_level_bootstraps = solver_bootstraps(
            problem_level_bootstraps, ci_options.n_bootstraps
        )
        ci = compute_ci(solver_level_bootstraps, estimator, "value", ci_options)

    return SolvabilityProfileResult(df=estimator, ci=ci, normalize=normalize)


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    experiments: list[ProblemSolver],
    plot_type: str,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    /,
    *,
    normalize: bool = True,
    solve_tolerance: float = 0.1,
    beta: float = 0.5,
) -> list[SolvabilityProfileResult]:
    """Analyze solvability profiles for multiple problems."""
    solvers, solver_to_inputs = group_by_solver(analysis_inputs, experiments)

    results: list[SolvabilityProfileResult] = []
    for solver in solvers:
        results.append(
            _analyze_solver(
                solver_to_inputs[solver],
                plot_type,
                ci_options,
                crn_options,
                normalize,
                solve_tolerance,
                beta,
            )
        )

    return results


def plot_many(
    results: list[SolvabilityProfileResult],
    experiments: list[ProblemSolver],
    plot_type: str,
    /,
    *,
    colors: list[str] | None = None,
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple solvability profile results."""
    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]

    solvers: list[str] = []
    seen: set[str] = set()
    for experiment in experiments:
        name = experiment.solver.name
        if name not in seen:
            solvers.append(name)
            seen.add(name)

    fig, ax = plt.subplots()
    handles = []
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        handles.append(plot(ax, result, color=color, logger=logger))
    ax.legend(handles=handles, labels=solvers)

    ax.set_xlabel("Budget")
    label = "Solvability CDF" if plot_type == "cdf" else "Solvability Quantile"
    ax.set_ylabel(label)

    return fig, ax


@dataclass(frozen=True)
class SolvabilityProfile(PlotConfig):
    """Options for solvability profile analysis."""

    plot_type: str = "cdf"
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
        """Analyze and plot aggregated solvability profiles."""
        results = analyze_many(
            analysis_inputs,
            experiments,
            self.plot_type,
            self.ci_options,
            self.crn_options,
            normalize=self.normalize,
            solve_tolerance=self.solve_tolerance,
            beta=self.beta,
        )
        return plot_many(results, experiments, self.plot_type)
