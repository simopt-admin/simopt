from copy import deepcopy
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from simopt.analysis.progress_curve import ProgressCurve
from simopt.experiment.plot import save_plots
from simopt.options import CrnOptions
from simopt.web.server import generate_plots

PLOT_TYPES = [
    "ALL",
    "MEAN",
    "QUANTILE",
    "BOX",
    "VIOLIN",
    "AREA",
    "AREA_MEAN",
    "AREA_STD_DEV",
    "SOLVE_TIME_CDF",
    "SOLVE_TIME_QUANTILE",
    "CDF_SOLVABILITY",
    "QUANTILE_SOLVABILITY",
    "DIFFERENCE_OF_CDF_SOLVABILITY",
    "DIFFERENCE_OF_QUANTILE_SOLVABILITY",
    "TERMINAL_SCATTER",
    "ALL_FEASIBILITY_PROGRESS",
    "MEAN_FEASIBILITY_PROGRESS",
    "QUANTILE_FEASIBILITY_PROGRESS",
    "FEASIBILITY_SCATTER",
    "FEASIBILITY_VIOLIN",
]


def web_entries(pairs, monkeypatch, crn=(True, False, True)):
    """Stand in for the ProblemSolvers the web server still holds until it moves to Experiment."""
    entries = [
        SimpleNamespace(
            solver=experiment.solver,
            problem=experiment.problem,
            crn_across_budget=crn[0],
            crn_across_macroreps=crn[1],
            crn_across_init_opt=crn[2],
        )
        for experiment, _ in pairs
    ]
    data = {
        id(entry): analysis_input for entry, (_, analysis_input) in zip(entries, pairs, strict=True)
    }
    monkeypatch.setattr("simopt.web.server.convert", lambda entry: data[id(entry)])
    return entries


@pytest.mark.parametrize("plot_type", PLOT_TYPES)
@pytest.mark.parametrize("all_in_one", [False, True])
def test_web_plot_generation(option_pair, plot_type, all_in_one, tmp_path, monkeypatch):
    second_problem = deepcopy(option_pair)
    for experiment, _ in second_problem:
        experiment.problem.name = "second problem"
    grid = web_entries(option_pair + second_problem, monkeypatch)
    params = {
        "all_in_one": all_in_one,
        "plot_conf_ints": False,
        "print_max_hw": False,
        "ref_solver": "comparison",
        "normalize": False,
        "beta": 0.73,
        "score_type": "norm",
        "norm_degree": 2,
        "two_sided": False,
    }
    before = set(plt.get_fignums())
    files = generate_plots(
        [{"plot_type": plot_type, "params": params}],
        [grid[:2], grid[2:]],
        [0, 1],
        [0, 1],
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        [{"name": "RNDSRCH"}, {"name": "comparison"}],
        [{"name": "SAN-2"}, {"name": "second problem"}],
        tmp_path,
    )
    assert files and len(files) == len(set(files))
    assert set(files) == {path.name for path in tmp_path.glob("*.png")}
    assert set(plt.get_fignums()) == before


def test_web_selection_and_option_forwarding(option_pair, tmp_path, monkeypatch):
    entries = web_entries(option_pair, monkeypatch)
    records = []

    def record_save(experiments, config, **kwargs):
        records.append((experiments, config, kwargs))
        return save_plots(experiments, config, **kwargs)

    monkeypatch.setattr("simopt.experiment.plot.save_plots", record_save)
    params = {
        "normalize": False,
        "beta": 0.73,
        "plot_conf_ints": True,
        "print_max_hw": False,
        "n_bootstraps": 8,
        "conf_level": 0.8,
        "bias_correction": False,
        "crn_across_budget": False,
        "crn_across_macroreps": True,
        "crn_across_init_opt": False,
    }
    files = generate_plots(
        [
            {
                "plot_type": "QUANTILE",
                "params": params,
                "solvers": ["comparison"],
                "problems": ["SAN-2"],
            }
        ],
        [entries],
        [2, 5],
        [4],
        {2: 0, 5: 1},
        {4: 0},
        [{"name": "unused"}] * 2
        + [{"name": "RNDSRCH"}]
        + [{"name": "unused"}] * 2
        + [{"name": "comparison"}],
        [{"name": "unused"}] * 4 + [{"name": "SAN-2"}],
        tmp_path,
    )
    assert len(files) == len(records) == 1
    experiments, config, _kwargs = records[0]
    ((selected, analysis_input),) = experiments
    expected, expected_input = option_pair[1]
    assert selected.solver is expected.solver and selected.problem is expected.problem
    assert analysis_input is expected_input
    assert isinstance(config, ProgressCurve)
    assert config.agg == "quantile" and config.beta == 0.73 and not config.normalize
    assert config.ci_options.n_bootstraps == 8
    assert config.ci_options.confidence_level == 0.8 and not config.ci_options.bias_correction
    assert not config.crn_options.across_budget and config.crn_options.across_macroreps
    assert not config.crn_options.across_x0_xstar
    assert files == ["quantile_progress_curves_problem_5.png"]


def test_web_crn_falls_back_to_reference_flags(option_pair, tmp_path, monkeypatch):
    entries = web_entries(option_pair, monkeypatch, crn=(False, True, False))
    records = []

    def record_save(experiments, config, **kwargs):
        records.append(config)
        return save_plots(experiments, config, **kwargs)

    monkeypatch.setattr("simopt.experiment.plot.save_plots", record_save)
    files = generate_plots(
        [{"plot_type": "MEAN", "params": {"plot_conf_ints": False, "print_max_hw": False}}],
        [entries],
        [0, 1],
        [0],
        {0: 0, 1: 1},
        {0: 0},
        [{"name": "RNDSRCH"}, {"name": "comparison"}],
        [{"name": "SAN-2"}],
        tmp_path,
    )
    assert files and len(records) == 1
    assert records[0].crn_options == CrnOptions(False, True, False)
