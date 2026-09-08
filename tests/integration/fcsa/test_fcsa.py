from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simopt.analysis.feasibility_progress import FeasibilityProgress
from simopt.analysis.progress_curve import ProgressCurve
from simopt.analysis.terminal_feasibility import TerminalFeasibility
from simopt.analysis.terminal_progress import TerminalProgress
from simopt.experiment.api import AnalysisInput
from simopt.experiment.single import ProblemSolver


def _save_figure(fig_ax: tuple[plt.Figure, plt.Axes], output: Path) -> None:
    fig, _ = fig_ax
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def test_terminal_progress_csa_saved(
    fcsa_all_solutions_results: list[AnalysisInput],
    fcsa_all_solutions_experiments: list[ProblemSolver],
    fcsa_plot_dir: Path,
) -> None:
    output = fcsa_plot_dir / "terminal_progress_csa.png"
    figs_axes = TerminalProgress(normalize=False).plot(
        [fcsa_all_solutions_results[0]],
        [fcsa_all_solutions_experiments[0]],
    )
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_terminal_feasibility_csa_saved(
    fcsa_all_solutions_results: list[AnalysisInput],
    fcsa_all_solutions_experiments: list[ProblemSolver],
    fcsa_plot_dir: Path,
) -> None:
    output = fcsa_plot_dir / "terminal_feasibility_csa.png"
    figs_axes = TerminalFeasibility(two_sided=True, plot_type="violin").plot(
        [fcsa_all_solutions_results[0]],
        [fcsa_all_solutions_experiments[0]],
    )
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_progress_curve_csa_saved(
    fcsa_all_solutions_results: list[AnalysisInput],
    fcsa_all_solutions_experiments: list[ProblemSolver],
    fcsa_plot_dir: Path,
) -> None:
    output = fcsa_plot_dir / "progress_curve_csa.png"
    figs_axes = ProgressCurve(agg="all", normalize=False).plot(
        [fcsa_all_solutions_results[0]],
        [fcsa_all_solutions_experiments[0]],
    )
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_feasibility_progress_csa_saved(
    fcsa_all_solutions_results: list[AnalysisInput],
    fcsa_all_solutions_experiments: list[ProblemSolver],
    fcsa_plot_dir: Path,
) -> None:
    output = fcsa_plot_dir / "feasibility_progress_csa.png"
    figs_axes = FeasibilityProgress(agg="all", two_sided=True).plot(
        [fcsa_all_solutions_results[0]],
        [fcsa_all_solutions_experiments[0]],
    )
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_progress_curve_csa_n_vs_fcsa_saved(
    fcsa_all_solutions_results: list[AnalysisInput],
    fcsa_all_solutions_experiments: list[ProblemSolver],
    fcsa_plot_dir: Path,
) -> None:
    output = fcsa_plot_dir / "progress_curve_csa_n_vs_fcsa.png"
    figs_axes = ProgressCurve(agg="all", normalize=False).plot(
        [fcsa_all_solutions_results[1], fcsa_all_solutions_results[2]],
        [fcsa_all_solutions_experiments[1], fcsa_all_solutions_experiments[2]],
    )
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_feasibility_progress_csa_n_vs_fcsa_saved(
    fcsa_all_solutions_results: list[AnalysisInput],
    fcsa_all_solutions_experiments: list[ProblemSolver],
    fcsa_plot_dir: Path,
) -> None:
    output = fcsa_plot_dir / "feasibility_progress_csa_n_vs_fcsa.png"
    figs_axes = FeasibilityProgress(agg="all", two_sided=True).plot(
        [fcsa_all_solutions_results[1], fcsa_all_solutions_results[2]],
        [fcsa_all_solutions_experiments[1], fcsa_all_solutions_experiments[2]],
    )
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_terminal_feasibility_recommended_saved(
    fcsa_recommended_results: list[AnalysisInput],
    fcsa_recommended_experiments: list[ProblemSolver],
    fcsa_plot_dir: Path,
) -> None:
    output = fcsa_plot_dir / "terminal_feasibility_recommended.png"
    figs_axes = TerminalFeasibility().plot(
        fcsa_recommended_results,
        fcsa_recommended_experiments,
    )
    _save_figure(figs_axes, output)
    assert output.exists()
    assert output.stat().st_size > 0
