"""Analysis and plotting of terminal scatterplots."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

from simopt.experiment.api import AnalysisInput, PlotConfig
from simopt.experiment.single import ProblemSolver
from simopt.logging import Logger, null_logger

from .common import styled_group_entries


@dataclass(frozen=True)
class TerminalScatterResult:
    """Container for terminal scatterplot results.

    Attributes:
        mean: Mean of the terminal values.
        std: Standard deviation of the terminal values.
        normalize: Whether the data is normalized.
    """

    mean: float
    std: float
    normalize: bool

    def to_numpy(self) -> np.ndarray:
        """Convert the terminal scatter result to a NumPy array."""
        return np.array([self.mean, self.std])


def analyze(
    analysis_input: AnalysisInput, /, *, normalize: bool = True
) -> TerminalScatterResult:
    """Analyze the terminal results for a scatterplot.

    Args:
        analysis_input: Analysis input containing the experiment results and references.
        normalize: Whether to normalize the objectives.

    Returns:
        TerminalScatterResult containing the terminal scatterplot summary.
    """
    df = analysis_input.mean_df
    values = df.groupby("mrep").last()
    column = "normalized_objective" if normalize else "objective"
    mean = float(np.mean(values[column]))
    std = float(np.std(values[column], ddof=1))
    return TerminalScatterResult(mean=mean, std=std, normalize=normalize)


def plot(
    ax: plt.Axes,
    result: TerminalScatterResult,
    color: str,
    marker: str,
    label: str | None = None,
    logger: Logger = null_logger,
) -> PathCollection:
    """Plot the terminal scatter result.

    Args:
        ax: Matplotlib axes to plot on.
        result: Terminal scatter data and normalization metadata.
        color: Color of the plot element.
        marker: Marker style for the plot.
        label: Legend label for the plot element.
        logger: Logger for debugging.

    Returns:
        The created plot collection.
    """
    logger.debug("data", data=result.to_numpy())
    return ax.scatter(
        x=result.mean, y=result.std, color=color, marker=marker, label=label
    )


def analyze_many(
    analysis_inputs: list[AnalysisInput],
    /,
    *,
    normalize: bool = True,
) -> list[TerminalScatterResult]:
    """Analyze terminal scatterplots for multiple experiments on the same problem."""
    return [
        analyze(analysis_input, normalize=normalize)
        for analysis_input in analysis_inputs
    ]


def plot_many(
    results: list[TerminalScatterResult],
    experiments: list[ProblemSolver],
    /,
    *,
    colors: list[str] | None = None,
    markers: list[str] | None = None,
    logger: Logger = null_logger,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple terminal scatter results."""
    if len(experiments) != len(results):
        raise ValueError("experiments must be the same length as results.")

    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]
    if markers is None:
        markers = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]

    group = list(zip(experiments, results, strict=True))
    fig, ax = plt.subplots()
    styled_entries, used_labels = styled_group_entries(group, colors, markers)
    for _, result, color, marker, label in styled_entries:
        plot(ax, result, color, marker, label=label, logger=logger)

    ax.set_xlabel("Mean")
    ax.set_ylabel("Standard Deviation")
    if used_labels:
        ax.legend()

    return fig, ax


@dataclass(frozen=True)
class TerminalScatter(PlotConfig):
    """Options for terminal scatterplot analysis."""

    normalize: bool = True

    def plot(
        self,
        analysis_inputs: list[AnalysisInput],
        experiments: list[ProblemSolver],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Analyze and plot terminal scatter summaries."""
        results = analyze_many(analysis_inputs, normalize=self.normalize)
        return plot_many(results, experiments)
