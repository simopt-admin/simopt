"""Analysis and plotting of terminal progress."""

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from simopt.experiment.api import AnalysisInput, PlotConfig
from simopt.experiment.single import ProblemSolver
from simopt.logging import Logger, null_logger

from .common import plot_distribution


@dataclass(frozen=True)
class TerminalProgressResult:
    """Container for terminal progress results.

    Attributes:
        data: Array of terminal progress values.
        normalize: Whether the data is normalized.
    """

    data: np.ndarray
    normalize: bool


def analyze(
    analysis_input: AnalysisInput, /, *, normalize: bool = True
) -> TerminalProgressResult:
    """Analyze the terminal progress.

    Args:
        analysis_input: Analysis input containing the experiment results and references.
        normalize: Whether to normalize the objectives.

    Returns:
        TerminalProgressResult containing terminal progress analysis results.
    """
    df = analysis_input.mean_df
    df = df.groupby("mrep", as_index=False).last()
    column = "normalized_objective" if normalize else "objective"
    return TerminalProgressResult(data=df[column], normalize=normalize)


def plot(
    ax: plt.Axes,
    result: TerminalProgressResult,
    plot_type: str = "violin",
    logger: Logger = null_logger,
) -> None:
    """Plot the terminal progress result.

    Args:
        ax: Matplotlib axes to plot on.
        result: Terminal progress data and normalization metadata.
        plot_type: Type of plot to generate (BOX or VIOLIN).
        logger: Logger for debugging.

    Returns:
        The created plot artist dictionary, or None when no plot is produced.
    """
    data = result.data
    logger.debug("data", data=data)
    if plot_type == "box":
        sns.boxplot(y=data, ax=ax)
        return
    if plot_type == "violin":
        # TODO: When setting cut=0.1, it matches the old plot.
        sns.violinplot(y=data, ax=ax)
        return
    raise ValueError("Plot type must be either 'box' or 'violin'.")


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    /,
    *,
    normalize: bool = True,
) -> list[TerminalProgressResult]:
    """Analyze terminal progress for multiple experiments on the same problem."""
    return [
        analyze(analysis_input, normalize=normalize)
        for analysis_input in analysis_inputs
    ]


def plot_many(
    results: list[TerminalProgressResult],
    experiments: list[ProblemSolver],
    /,
    *,
    plot_type: str = "violin",
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple terminal progress results."""
    if len(experiments) != len(results):
        raise ValueError("experiments must be the same length as results.")

    fig, ax = plt.subplots()

    records = []
    for experiment, result in zip(experiments, results, strict=True):
        data = np.asarray(result.data, dtype=float)
        logger.debug("data", data=data)
        for value in data:
            records.append(
                {"Solver": experiment.solver.name, "Objective": float(value)}
            )
    plot_distribution(ax, records, "Solver", "Objective", plot_type)

    return fig, ax


@dataclass(frozen=True)
class TerminalProgress(PlotConfig):
    """Options for terminal progress analysis."""

    normalize: bool = True
    plot_type: Literal["box", "violin"] = "violin"

    def plot(
        self,
        analysis_inputs: list[AnalysisInput],
        experiments: list[ProblemSolver],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Analyze and plot terminal progress distributions."""
        results = analyze_many(analysis_inputs, normalize=self.normalize)
        return plot_many(results, experiments, plot_type=self.plot_type)
