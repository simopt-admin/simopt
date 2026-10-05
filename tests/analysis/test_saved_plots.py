# ruff: noqa: F811
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import pytest

from simopt.analysis import common
from simopt.analysis.auc import Auc
from simopt.analysis.diff import DiffSolvability
from simopt.analysis.feasibility_progress import FeasibilityProgress
from simopt.analysis.progress_curve import ProgressCurve
from simopt.analysis.solvability_cdf import SolvabilityCdf
from simopt.analysis.solvability_profile import SolvabilityProfile
from simopt.analysis.terminal_feasibility import TerminalFeasibility
from simopt.analysis.terminal_progress import TerminalProgress
from simopt.analysis.terminal_scatterplot import TerminalScatter
from simopt.experiment.plot import save_plots
from simopt.options import ConfidenceIntervalOptions
from tests.analysis.legacy_options_helpers import option_experiment, option_pair  # noqa: F401


@pytest.mark.parametrize("all_in_one", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize(
    "config",
    [
        ProgressCurve(ci_options=None),
        FeasibilityProgress(ci_options=None, two_sided=False),
        SolvabilityCdf(ci_options=None),
        TerminalProgress(),
        TerminalFeasibility(ci_options=None),
        SolvabilityProfile(ci_options=None),
        DiffSolvability(reference_solver="comparison", ci_options=None),
        Auc(ci_options=None),
        TerminalScatter(),
    ],
)
def test_saved_plot_grouping(option_pair, config, all_in_one, transpose, tmp_path):
    second_problem = deepcopy(option_pair)
    for experiment in second_problem:
        experiment.problem.name = "second problem"
    grid = [option_pair, second_problem]
    if transpose:
        grid = [list(group) for group in zip(*grid, strict=True)]
    before = set(plt.get_fignums())
    paths = save_plots(
        grid,
        config,
        output_dir=tmp_path,
        filename="result.png",
        all_in_one=all_in_one,
        save_as_pickle=True,
        plot_title="Custom title",
    )
    per_problem = isinstance(
        config,
        (ProgressCurve, FeasibilityProgress, SolvabilityCdf, TerminalProgress, TerminalFeasibility),
    )
    expected = (
        (2 if all_in_one else 4)
        if per_problem
        else (1 if all_in_one or isinstance(config, DiffSolvability) else 2)
    )
    assert len(paths) == expected
    assert len(set(paths)) == expected
    assert set(plt.get_fignums()) == before
    for path in paths:
        assert path.parent == tmp_path and path.stat().st_size > 0
        with path.with_suffix(".pickle").open("rb") as file:
            fig = pickle.load(file)
        assert fig.axes[0].get_title() == "Custom title"
        plt.close(fig)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("plot_conf_ints", [False, True])
@pytest.mark.parametrize("print_max_hw", [False, True])
def test_saved_ci_display(
    option_experiment, normalize, plot_conf_ints, print_max_hw, tmp_path, monkeypatch
):
    if not plot_conf_ints and not print_max_hw:

        def fail_bootstrap(*args, **kwargs):
            raise AssertionError("CI-disabled saving must not bootstrap")

        monkeypatch.setattr(common, "bootstrap", fail_bootstrap)
    paths = save_plots(
        [option_experiment],
        ProgressCurve(
            normalize=normalize,
            ci_options=ConfidenceIntervalOptions(n_bootstraps=8, bias_correction=False),
        ),
        output_dir=tmp_path,
        filename="result",
        save_as_pickle=True,
        plot_conf_ints=plot_conf_ints,
        print_max_hw=print_max_hw,
    )
    with paths[0].with_suffix(".pickle").open("rb") as file:
        fig = pickle.load(file)
    ax = fig.axes[0]
    assert len(ax.lines) == (3 if plot_conf_ints else 1)
    assert len(ax.collections) == int(plot_conf_ints)
    assert bool(fig.texts) == print_max_hw
    assert max(ax.lines[0].get_xdata()) == (1 if normalize else 10)
    assert ax.get_ylabel() == ("Normalized Objective" if normalize else "Objective")


@pytest.mark.parametrize("plot_zero", [False, True])
@pytest.mark.parametrize("plot_optimal", [False, True])
def test_saved_feasibility_references(option_experiment, plot_zero, plot_optimal, tmp_path):
    paths = save_plots(
        [option_experiment],
        TerminalFeasibility(ci_options=None),
        output_dir=tmp_path,
        save_as_pickle=True,
        plot_zero=plot_zero,
        plot_optimal=plot_optimal,
    )
    with paths[0].with_suffix(".pickle").open("rb") as file:
        fig = pickle.load(file)
    assert len(fig.axes[0].lines) == int(plot_zero) + int(plot_optimal)


def test_saved_plot_cleanup_on_error(option_experiment, tmp_path, monkeypatch):
    before = set(plt.get_fignums())

    def fail_save(*args, **kwargs):
        raise OSError("save failed")

    monkeypatch.setattr(plt.Figure, "savefig", fail_save)
    with pytest.raises(OSError, match="save failed"):
        save_plots([option_experiment], ProgressCurve(ci_options=None), output_dir=tmp_path)
    assert set(plt.get_fignums()) == before


def test_saved_empty_collection(tmp_path):
    assert save_plots([], ProgressCurve(), output_dir=tmp_path) == []
    assert not list(tmp_path.iterdir())


def test_saved_default_names_distinguish_options(option_experiment, tmp_path):
    configs = [
        ProgressCurve(agg="all", ci_options=None),
        ProgressCurve(agg="mean", ci_options=None),
        ProgressCurve(agg="mean", normalize=False, ci_options=None),
        ProgressCurve(agg="quantile", beta=0.1, ci_options=None),
        ProgressCurve(agg="quantile", beta=0.9, ci_options=None),
    ]
    paths = [save_plots([option_experiment], config, output_dir=tmp_path)[0] for config in configs]
    assert len(set(paths)) == len(configs)
    assert len(list(tmp_path.glob("*.png"))) == len(configs)


@pytest.mark.parametrize("plot_zero", [False, True])
def test_saved_feasibility_zero_line(option_experiment, plot_zero, tmp_path):
    paths = save_plots(
        [option_experiment],
        FeasibilityProgress(agg="all", ci_options=None),
        output_dir=tmp_path,
        save_as_pickle=True,
        plot_zero=plot_zero,
    )
    with paths[0].with_suffix(".pickle").open("rb") as file:
        fig = pickle.load(file)
    assert len(fig.axes[0].lines) == option_experiment.n_macroreps + int(plot_zero)


@pytest.mark.parametrize("across_budget", [False, True])
@pytest.mark.parametrize("across_macroreps", [False, True])
@pytest.mark.parametrize("across_x0_xstar", [False, True])
@pytest.mark.parametrize("use_experiment_crn", [False, True])
def test_saved_experiment_crn_flags(
    option_experiment,
    across_budget,
    across_macroreps,
    across_x0_xstar,
    use_experiment_crn,
    tmp_path,
    monkeypatch,
):
    from simopt.options import CrnOptions

    option_experiment.crn_across_budget = across_budget
    option_experiment.crn_across_macroreps = across_macroreps
    option_experiment.crn_across_init_opt = across_x0_xstar
    configured = CrnOptions(
        across_budget=not across_budget,
        across_macroreps=not across_macroreps,
        across_x0_xstar=not across_x0_xstar,
    )
    records = []

    def record_plot(self, analysis_inputs, experiments):
        records.append(self.crn_options)
        return plt.subplots()

    monkeypatch.setattr(ProgressCurve, "plot", record_plot)
    config = ProgressCurve(ci_options=None, crn_options=configured)
    save_plots(
        [option_experiment], config, output_dir=tmp_path, use_experiment_crn=use_experiment_crn
    )
    desired = (
        CrnOptions(
            across_budget=across_budget,
            across_macroreps=across_macroreps,
            across_x0_xstar=across_x0_xstar,
        )
        if use_experiment_crn
        else configured
    )
    assert records == [desired]
    assert config.crn_options == configured
