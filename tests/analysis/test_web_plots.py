# ruff: noqa: F811
from copy import deepcopy

import matplotlib.pyplot as plt
import pytest

from simopt.analysis.progress_curve import ProgressCurve
from simopt.experiment.plot import save_plots
from simopt.web.server import generate_plots
from tests.analysis.legacy_options_helpers import option_experiment, option_pair  # noqa: F401

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


@pytest.mark.parametrize("plot_type", PLOT_TYPES)
@pytest.mark.parametrize("all_in_one", [False, True])
def test_web_plot_generation(option_pair, plot_type, all_in_one, tmp_path):
    second_problem = deepcopy(option_pair)
    for experiment in second_problem:
        experiment.problem.name = "second problem"
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
        [option_pair, second_problem],
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
        [option_pair],
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
    assert experiments == [option_pair[1]]
    assert isinstance(config, ProgressCurve)
    assert config.agg == "quantile" and config.beta == 0.73 and not config.normalize
    assert config.ci_options.n_bootstraps == 8
    assert config.ci_options.confidence_level == 0.8 and not config.ci_options.bias_correction
    assert not config.crn_options.across_budget and config.crn_options.across_macroreps
    assert not config.crn_options.across_x0_xstar
    assert files == ["quantile_progress_curves_problem_5.png"]
