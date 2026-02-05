from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simopt.analysis.auc import Auc
from simopt.analysis.diff import DiffSolvability
from simopt.analysis.progress_curve import ProgressCurve
from simopt.analysis.solvability_profile import SolvabilityProfile
from simopt.analysis.terminal_progress import TerminalProgress
from simopt.analysis.terminal_scatterplot import TerminalScatter
from simopt.experiment.api import AnalysisInput
from simopt.experiment.single import ProblemSolver


def _save_figure(fig_ax: tuple[plt.Figure, plt.Axes], output: Path) -> None:
    fig, _ = fig_ax
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _first_problem_slice(
    results: list[AnalysisInput], experiments: list[ProblemSolver], n_solvers: int = 5
) -> tuple[list[AnalysisInput], list[ProblemSolver]]:
    return results[:n_solvers], experiments[:n_solvers]


def test_auc(
    sscont_results: list[AnalysisInput],
    sscont_experiments: list[ProblemSolver],
    sscont_plot_dir: Path,
) -> None:
    output = sscont_plot_dir / "auc.png"
    figs_axes = Auc().plot(sscont_results, sscont_experiments)
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_solvability_profile_cdf(
    sscont_results: list[AnalysisInput],
    sscont_experiments: list[ProblemSolver],
    sscont_plot_dir: Path,
) -> None:
    output = sscont_plot_dir / "solvability_profile_cdf.png"
    figs_axes = SolvabilityProfile(
        plot_type="cdf",
        solve_tolerance=0.1,
    ).plot(sscont_results, sscont_experiments)
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_solvability_profile_quantile(
    sscont_results: list[AnalysisInput],
    sscont_experiments: list[ProblemSolver],
    sscont_plot_dir: Path,
) -> None:
    output = sscont_plot_dir / "solvability_profile_quantile.png"
    figs_axes = SolvabilityProfile(
        plot_type="quantile",
        solve_tolerance=0.1,
        beta=0.5,
    ).plot(sscont_results, sscont_experiments)
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_diff_solvability_cdf(
    sscont_results: list[AnalysisInput],
    sscont_experiments: list[ProblemSolver],
    sscont_plot_dir: Path,
) -> None:
    output = sscont_plot_dir / "diff_solvability_cdf.png"
    figs_axes = DiffSolvability(
        plot_type="cdf",
        solve_tolerance=0.1,
        reference_solver="ASTRODF",
    ).plot(sscont_results, sscont_experiments)
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_diff_solvability_quantile(
    sscont_results: list[AnalysisInput],
    sscont_experiments: list[ProblemSolver],
    sscont_plot_dir: Path,
) -> None:
    output = sscont_plot_dir / "diff_solvability_quantile.png"
    figs_axes = DiffSolvability(
        plot_type="quantile",
        solve_tolerance=0.1,
        beta=0.5,
        reference_solver="ASTRODF",
    ).plot(sscont_results, sscont_experiments)
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_terminal_scatter(
    sscont_results: list[AnalysisInput],
    sscont_experiments: list[ProblemSolver],
    sscont_plot_dir: Path,
) -> None:
    output = sscont_plot_dir / "terminal_scatter.png"
    figs_axes = TerminalScatter().plot(sscont_results, sscont_experiments)
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_progress_curve(
    sscont_results: list[AnalysisInput],
    sscont_experiments: list[ProblemSolver],
    sscont_plot_dir: Path,
) -> None:
    output = sscont_plot_dir / "progress_curve.png"
    first_results, first_experiments = _first_problem_slice(
        sscont_results, sscont_experiments
    )
    figs_axes = ProgressCurve(agg="mean", normalize=False).plot(
        first_results, first_experiments
    )
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_terminal_progress(
    sscont_results: list[AnalysisInput],
    sscont_experiments: list[ProblemSolver],
    sscont_plot_dir: Path,
) -> None:
    output = sscont_plot_dir / "terminal_progress.png"
    first_results, first_experiments = _first_problem_slice(
        sscont_results, sscont_experiments
    )
    figs_axes = TerminalProgress(normalize=True).plot(first_results, first_experiments)
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0
