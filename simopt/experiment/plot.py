"""Save plots produced by analysis configurations for experiment collections."""

import pickle
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.container import ErrorbarContainer

from simopt.analysis.auc import Auc
from simopt.analysis.diff import DiffSolvability
from simopt.analysis.feasibility_progress import FeasibilityProgress
from simopt.analysis.progress_curve import ProgressCurve
from simopt.analysis.solvability_cdf import SolvabilityCdf
from simopt.analysis.solvability_profile import SolvabilityProfile
from simopt.analysis.terminal_feasibility import TerminalFeasibility
from simopt.analysis.terminal_progress import TerminalProgress
from simopt.analysis.terminal_scatterplot import TerminalScatter
from simopt.compat import convert
from simopt.experiment import EXPERIMENT_DIR, Experiment, ProblemSolver
from simopt.options import ConfidenceIntervalOptions, CrnOptions

if TYPE_CHECKING:
    from matplotlib.typing import LegendLocType

PlotConfiguration = (
    Auc
    | DiffSolvability
    | FeasibilityProgress
    | ProgressCurve
    | SolvabilityCdf
    | SolvabilityProfile
    | TerminalFeasibility
    | TerminalProgress
    | TerminalScatter
)


def _flatten(experiments: list[ProblemSolver] | list[list[ProblemSolver]]) -> list[ProblemSolver]:
    result = []
    for group in experiments:
        if isinstance(group, ProblemSolver):
            result.append(group)
        else:
            result.extend(group)
    return result


def _groups(
    experiments: list[ProblemSolver], config: PlotConfiguration, all_in_one: bool
) -> list[list[ProblemSolver]]:
    if isinstance(config, (Auc, TerminalScatter, SolvabilityProfile, DiffSolvability)):
        if all_in_one:
            return [experiments]
        groups = {}
        for experiment in experiments:
            groups.setdefault(experiment.solver.name, []).append(experiment)
        if isinstance(config, DiffSolvability):
            reference = groups.get(config.reference_solver, [])
            return [
                reference + group
                for name, group in groups.items()
                if name != config.reference_solver
            ]
        return list(groups.values())
    if not all_in_one:
        return [[experiment] for experiment in experiments]
    groups = {}
    for experiment in experiments:
        groups.setdefault(experiment.problem.name, []).append(experiment)
    return list(groups.values())


def _plot_name(config: PlotConfiguration) -> str:
    parts = [type(config).__name__]
    for key in ("agg", "plot_type", "score_type", "norm_degree", "two_sided"):
        value = getattr(config, key, None)
        if value is not None:
            parts.append(str(value))
    if hasattr(config, "normalize"):
        parts.append("normalized" if config.normalize else "raw")
    if (
        getattr(config, "agg", None) == "quantile"
        or getattr(config, "plot_type", None) == "quantile"
    ):
        beta = getattr(config, "beta", None)
        parts.append(f"beta_{beta:g}")
    if hasattr(config, "solve_tolerance"):
        parts.append(f"tol_{config.solve_tolerance:g}")
    return "_".join(parts)


def _confidence_display(
    fig: plt.Figure,
    ax: plt.Axes,
    ci_options: ConfidenceIntervalOptions,
    plot_conf_ints: bool,
    print_max_hw: bool,
) -> None:
    bounds = [
        line for line in ax.lines if line.get_linestyle() == "--" and line.get_linewidth() == 1
    ]
    halfwidths = []
    for lower, upper in zip(bounds[::2], bounds[1::2], strict=True):
        halfwidths.extend((np.asarray(upper.get_ydata()) - np.asarray(lower.get_ydata())) / 2)
    for collection in ax.collections:
        if isinstance(collection, LineCollection):
            for segment in collection.get_segments():
                halfwidths.append(float(np.max(np.abs(segment[-1] - segment[0]))) / 2)
    if print_max_hw and halfwidths:
        finite = np.asarray(halfwidths)[np.isfinite(halfwidths)]
        if len(finite):
            confidence_level = ci_options.confidence_level
            fig.text(
                0.5,
                0.01,
                f"Max halfwidth ({confidence_level:.0%} CI): {np.max(finite):.3g}",
                ha="center",
            )
    if plot_conf_ints:
        return
    for line in bounds:
        line.remove()
    for container in list(ax.containers):
        if isinstance(container, ErrorbarContainer):
            _, caps, bars = container.lines
            for artist in (*caps, *bars):
                artist.remove()
    for collection in list(ax.collections):
        if isinstance(collection, PolyCollection) and collection.get_alpha() == 0.2:
            collection.remove()


def save_plots(
    experiments: list[ProblemSolver] | list[list[ProblemSolver]],
    config: PlotConfiguration,
    *,
    output_dir: Path | str = EXPERIMENT_DIR / "plots",
    filename: str | None = None,
    all_in_one: bool = True,
    use_experiment_crn: bool = False,
    plot_title: str | None = None,
    legend_loc: "LegendLocType | None" = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    plot_conf_ints: bool = True,
    print_max_hw: bool = False,
    plot_zero: bool = True,
    plot_optimal: bool = False,
    solver_set_name: str = "SOLVER_SET",
    problem_set_name: str = "PROBLEM_SET",
) -> list[Path]:
    """Analyze and save experiment plots, grouping by problem or solver as appropriate.

    Pass an analysis configuration to choose the statistic, normalization, confidence
    intervals and CRN options. Flat collections and either orientation of an experiment
    grid are accepted. Separate plots receive numbered filename suffixes. Returned paths
    refer to image files; optional figure pickles are saved alongside them.
    Set use_experiment_crn to use the stored experiment CRN flags instead of the
    configuration's CRN options when migrating experiment plots.
    """
    experiments = _flatten(experiments)
    if not experiments:
        return []
    ci_config = isinstance(
        config,
        (
            Auc,
            DiffSolvability,
            FeasibilityProgress,
            ProgressCurve,
            SolvabilityCdf,
            SolvabilityProfile,
            TerminalFeasibility,
        ),
    )
    if ci_config and not plot_conf_ints and not print_max_hw:
        config = replace(config, ci_options=None)
    groups = _groups(experiments, config, all_in_one)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, group in enumerate(groups):
        if use_experiment_crn and isinstance(
            config,
            (
                Auc,
                DiffSolvability,
                FeasibilityProgress,
                ProgressCurve,
                SolvabilityCdf,
                SolvabilityProfile,
                TerminalFeasibility,
            ),
        ):
            reference = group[0]
            config = replace(
                config,
                crn_options=CrnOptions(
                    across_budget=reference.crn_across_budget,
                    across_macroreps=reference.crn_across_macroreps,
                    across_x0_xstar=reference.crn_across_init_opt,
                ),
            )
        # Conversion preserves experiment data while the analysis selects raw or normalized values.
        analysis_inputs = [convert(experiment) for experiment in group]
        before = set(plt.get_fignums())
        try:
            pairs = [Experiment(solver=e.solver, problem=e.problem) for e in group]
            fig, ax = config.plot(analysis_inputs, pairs)
            if (
                isinstance(
                    config,
                    (
                        Auc,
                        DiffSolvability,
                        FeasibilityProgress,
                        ProgressCurve,
                        SolvabilityCdf,
                        SolvabilityProfile,
                        TerminalFeasibility,
                    ),
                )
                and config.ci_options is not None
            ):
                _confidence_display(fig, ax, config.ci_options, plot_conf_ints, print_max_hw)
            if isinstance(config, FeasibilityProgress) and not plot_zero:
                for line in list(ax.lines):
                    if line.get_transform() == ax.get_yaxis_transform() and np.all(
                        np.asarray(line.get_ydata()) == 0
                    ):
                        line.remove()
            if isinstance(config, TerminalFeasibility):
                if plot_zero:
                    ax.axhline(0, color="red", linestyle="--", linewidth=0.75)
                if plot_optimal and config.plot_type == "scatter":
                    ax.axvline(
                        float(np.mean(group[0].xstar_postreps)),
                        color="red",
                        linestyle="--",
                        linewidth=0.75,
                    )
            if isinstance(config, (ProgressCurve, TerminalProgress)):
                ax.set_ylabel("Normalized Objective" if config.normalize else "Objective")
            if isinstance(config, ProgressCurve):
                ax.set_xlabel("Fraction of Budget" if config.normalize else "Budget")
            problems = ", ".join(dict.fromkeys(experiment.problem.name for experiment in group))
            solvers = ", ".join(dict.fromkeys(experiment.solver.name for experiment in group))
            solver_label = (
                solver_set_name
                if len({experiment.solver.name for experiment in group}) > 1
                else solvers
            )
            problem_label = (
                problem_set_name
                if len({experiment.problem.name for experiment in group}) > 1
                else problems
            )
            ax.set_title(plot_title or f"{_plot_name(config)}: {solver_label} on {problem_label}")
            legend = ax.get_legend()
            if legend_loc is not None and legend is not None:
                legend.set_loc(legend_loc)
            name = Path(filename or f"{_plot_name(config)}_{solver_label}_{problem_label}{ext}")
            if not name.suffix:
                name = name.with_suffix(ext)
            if len(groups) > 1:
                name = name.with_name(f"{name.stem}_{index + 1}{name.suffix}")
            path = output_dir / name
            fig.savefig(path, dpi=150, bbox_inches="tight")
            if save_as_pickle:
                with path.with_suffix(".pickle").open("wb") as file:
                    pickle.dump(fig, file, protocol=pickle.HIGHEST_PROTOCOL)
            paths.append(path)
        finally:
            for number in set(plt.get_fignums()) - before:
                plt.close(number)
    return paths
