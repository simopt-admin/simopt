"""Common helpers for plots."""

import copy
from collections.abc import Callable
from statistics import quantiles
from typing import Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from simopt.analysis.bootstrap import bootstrap
from simopt.bootstrap import compute_bootstrap_conf_int
from simopt.experiment.api import AnalysisInput
from simopt.experiment.single import ProblemSolver
from simopt.options import ConfidenceIntervalOptions, CrnOptions

R = TypeVar("R")


def styled_group_entries(
    group: list[tuple[ProblemSolver, R]],
    colors: list[str],
    markers: list[str],
) -> tuple[list[tuple[ProblemSolver, R, str, str, str]], set[str]]:
    """Attach color, marker, and legend label metadata to grouped entries."""
    style_map: dict[str, tuple[str, str]] = {}
    style_index = 0
    used_labels: set[str] = set()
    styled_entries: list[tuple[ProblemSolver, R, str, str, str]] = []

    for experiment, result in group:
        style_key = experiment.solver.name
        if style_key not in style_map:
            color = colors[style_index % len(colors)]
            marker = markers[style_index % len(markers)]
            style_map[style_key] = (color, marker)
            style_index += 1
        color, marker = style_map[style_key]
        label = style_key if style_key not in used_labels else "_nolegend_"
        used_labels.add(style_key)
        styled_entries.append((experiment, result, color, marker, label))

    return styled_entries, used_labels


def aggregate_curve(
    df: pd.DataFrame,
    agg: str,
    value_column: str,
    beta: float | None = None,
    quantile_method: Literal["python", "pandas"] = "python",
) -> pd.DataFrame:
    """Aggregate a curve across macroreplications.

    Aggregates objective values (or other specified values) across different
    macroreplications at each budget level. Supports mean and quantile aggregation.

    Args:
        df: DataFrame containing the data with 'budget' and value columns.
        agg: Aggregation type, either 'mean' or 'quantile'.
        value_column: Name of the column containing values to aggregate.
        beta: Quantile level (0-1) required when agg is 'quantile'.
        quantile_method: Quantile implementation method. Supported values:
            "python" (statistics.quantiles) or "pandas" (DataFrame.quantile).

    Returns:
        DataFrame with 'budget' and aggregated value columns.

    Raises:
        ValueError: If agg is 'quantile' but beta is None, if quantile method is
            unsupported, or if agg is unsupported.
    """
    x = np.sort(df["budget"].unique())
    pivot = df.pivot_table(index="budget", columns="mrep", values=value_column)
    pivot = pivot.reindex(x).ffill()
    if agg == "mean":
        y = pivot.mean(axis=1)
    elif agg == "quantile":
        if beta is None:
            raise ValueError("beta is required for quantile aggregation.")
        if quantile_method == "python":
            i = int(beta * 99)
            y = [quantiles(row, n=100)[i] for _, row in pivot.iterrows()]
        elif quantile_method == "pandas":
            y = pivot.quantile(q=beta, axis=1).to_numpy()
        else:
            raise ValueError("unsupported quantile method.")
    else:
        raise ValueError("unsupported curve type.")
    return pd.DataFrame({"budget": x, value_column: np.asarray(y)})


def compute_ci(
    bootstraps: list[pd.DataFrame],
    estimator: pd.DataFrame | None,
    value_column: str,
    ci_options: ConfidenceIntervalOptions,
) -> pd.DataFrame:
    """Compute confidence intervals from bootstrap samples.

    Computes confidence intervals for each budget level using bootstrap samples.
    Supports bias correction if a bias correction estimator is provided.

    Args:
        bootstraps: List of DataFrames, each containing bootstrap sample results.
        ci_options: Options for confidence interval computation including
            confidence level and bias correction settings.
        estimator: DataFrame with bias correction estimates, or None
            if bias correction is not used.
        value_column: Name of the column containing values to compute CIs for.

    Returns:
        DataFrame with 'budget', 'lb' (lower bound), and 'ub' (upper bound) columns.

    Raises:
        ValueError: If bias correction is requested but ci_bias_correction is None.
    """
    dfs = []
    for i, df in enumerate(bootstraps):
        df["bootstrap"] = i
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    budget = np.sort(df["budget"].unique())

    pivot = df.pivot_table(index="budget", columns="bootstrap", values=value_column)
    pivot = pivot.reindex(budget).ffill()

    if ci_options.bias_correction and estimator is None:
        raise ValueError("bias correction estimator is required.")

    if estimator is not None:
        estimator = (
            estimator.set_index("budget")[value_column]
            .reindex(np.sort(np.unique(np.concatenate([budget, estimator["budget"]]))))
            .ffill()
            .reindex(budget)
            .to_numpy()
        )

    lb_values: list[float] = []
    ub_values: list[float] = []
    for i, (_, row) in enumerate(pivot.iterrows()):
        lb, ub = compute_bootstrap_conf_int(
            row.to_list(),
            ci_options.confidence_level,
            ci_options.bias_correction,
            estimator[i] if estimator is not None else None,
        )
        lb_values.append(lb)
        ub_values.append(ub)

    return pd.DataFrame({"budget": budget, "lb": lb_values, "ub": ub_values})


E = TypeVar("E")
B = TypeVar("B")
Ci = TypeVar("Ci")
BootstrapFn = Callable[
    [AnalysisInput, int, Callable[[AnalysisInput], B], CrnOptions], list[B]
]
CiFn = Callable[[list[B], E | None, ConfidenceIntervalOptions], Ci]


def _default_bootstrap_fn(
    analysis_input: AnalysisInput,
    n_bootstraps: int,
    estimator_fn: Callable[[AnalysisInput], B],
    crn_options: CrnOptions,
) -> list[B]:
    return bootstrap(
        analysis_input,
        n_bootstraps,
        estimator_fn,
        crn_options.across_budget,
        crn_options.across_macroreps,
        crn_options.across_x0_xstar,
    )


def compute_estimator_and_ci(
    analysis_input: AnalysisInput,
    estimator_fn: Callable[[AnalysisInput], E],
    ci_fn: CiFn,
    ci_options: ConfidenceIntervalOptions | None,
    crn_options: CrnOptions,
    *,
    normalize: bool = True,
    bootstrap_fn: BootstrapFn | None = None,
) -> tuple[E, Ci | None]:
    """Compute an estimator and optional bootstrap confidence intervals."""

    def select_input(analysis_input: AnalysisInput) -> AnalysisInput:
        analysis_input = copy.copy(analysis_input)

        if not normalize:
            return analysis_input

        df_mean = analysis_input.mean_df
        df_mean = df_mean[
            [
                "mrep",
                "step",
                "normalized_budget",
                "normalized_objective",
                "stochastic_constraints",
            ]
        ]
        df_mean = df_mean.rename(
            columns={"normalized_budget": "budget", "normalized_objective": "objective"}
        )
        analysis_input.mean_df = df_mean
        return analysis_input

    def wrapped_estimator(analysis_input: AnalysisInput) -> E:
        return estimator_fn(select_input(analysis_input))

    estimator = wrapped_estimator(analysis_input)

    if ci_options is None:
        return estimator, None

    if ci_options.n_bootstraps < 1:
        raise ValueError("Number of bootstraps must be a positive integer.")
    if not 0 < ci_options.confidence_level < 1:
        raise ValueError("Confidence level must be in (0, 1).")

    bootstrap_fn = bootstrap_fn or _default_bootstrap_fn
    bootstraps = bootstrap_fn(
        analysis_input,
        ci_options.n_bootstraps,
        wrapped_estimator,
        crn_options,
    )
    ci = ci_fn(
        bootstraps, estimator if ci_options.bias_correction else None, ci_options
    )
    return estimator, ci


def plot_step(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str = "C0",
    linestyle: str = "-",
    linewidth: float = 2,
) -> plt.Line2D:
    """Plot a step function on a matplotlib axes.

    Creates a step plot where the value is constant between x points (post-step style).

    Args:
        ax: Matplotlib axes to plot on.
        x: Array of x-coordinates (budget values).
        y: Array of y-coordinates (values to plot).
        color: Line color (default: 'C0').
        linestyle: Line style (default: '-').
        linewidth: Line width (default: 2).

    Returns:
        The Line2D object representing the plotted step function.
    """
    return ax.step(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        where="post",
    )[0]


def step_to_continuous(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert step function data to continuous format for filled plots.

    Transforms step function data by duplicating points to create a continuous
    representation suitable for fill_between plots.

    Args:
        x: Array of x-coordinates from step function.
        y: Array of y-coordinates from step function.

    Returns:
        Tuple of (x_continuous, y_continuous) arrays suitable for continuous plotting.
    """
    x = np.repeat(x, 2)
    y = np.repeat(y, 2)
    return x[1:], y[:-1]


def plot_ci(
    ax: plt.Axes,
    df: pd.DataFrame,
    color: str = "C0",
    alpha: float = 0.2,
) -> None:
    """Plot confidence intervals as shaded regions with bounds.

    Plots confidence intervals by drawing dashed lines for upper and lower bounds
    and filling the region between them with a semi-transparent color.

    Args:
        ax: Matplotlib axes to plot on.
        df: DataFrame containing 'budget', 'lb' (lower bound), and 'ub'
            (upper bound) columns.
        color: Color for the confidence interval lines and fill (default: 'C0').
        alpha: Transparency level for the filled region (default: 0.2).
    """
    plot_step(
        ax,
        df["budget"].to_numpy(),
        df["lb"].to_numpy(),
        color=color,
        linestyle="--",
        linewidth=1,
    )
    plot_step(
        ax,
        df["budget"].to_numpy(),
        df["ub"].to_numpy(),
        color=color,
        linestyle="--",
        linewidth=1,
    )
    budget, lb = step_to_continuous(df["budget"].to_numpy(), df["lb"].to_numpy())
    _, ub = step_to_continuous(df["budget"].to_numpy(), df["ub"].to_numpy())
    ax.fill_between(
        x=budget,
        y1=lb,
        y2=ub,
        color=color,
        alpha=alpha,
    )


def plot_distribution(
    ax: plt.Axes,
    data: list[dict[str, str | float]],
    x: str,
    y: str,
    plot_type: str,
) -> None:
    """Plot distribution as boxplot or violin plot."""
    if not data:
        return

    df = pd.DataFrame.from_records(data)

    if plot_type == "box":
        sns.boxplot(data=df, x=x, y=y, ax=ax)
        return

    if plot_type == "violin":
        sns.violinplot(
            data=df,
            x=x,
            y=y,
            hue=x,
            inner="stick",
            density_norm="width",
            cut=0.1,
            dodge=False,
            legend=False,
            ax=ax,
        )
        return

    raise ValueError("plot type must be either 'box' or 'violin'.")
