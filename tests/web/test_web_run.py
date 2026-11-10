"""End-to-end tests for the web server's run path and plots-only rerun."""

import json
import pickle

import pytest

from simopt.experiment import api
from simopt.experiment.plot import save_plots
from simopt.options import CrnOptions
from simopt.web import server


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    original_run = api.run
    monkeypatch.setattr(
        api, "run", lambda *args, **kwargs: original_run(*args, **{**kwargs, "n_jobs": 1})
    )
    return tmp_path


def payload(plots, rename=None):
    return {
        "experiment_params": {"num_macroreps": 2, "num_postreps": 2, "num_postnorms": 2},
        "problems": [{"name": "SAN-2", "fixed_factors": {"budget": 20}, "model_fixed_factors": {}}],
        "solvers": [
            {"name": "RNDSRCH", "fixed_factors": {"sample_size": 2}},
            {"name": "RNDSRCH", "rename": rename, "fixed_factors": {"sample_size": 3}},
        ],
        "plots": plots,
    }


MEAN_PLOT = {"plot_type": "MEAN", "params": {"plot_conf_ints": False, "print_max_hw": False}}


def test_run_experiment_writes_results_and_plots(results_dir):
    folder = results_dir / "run"
    folder.mkdir()
    server.run_experiment_async("run", payload([MEAN_PLOT], rename="second"))
    assert "Complete!" in (folder / "index.html").read_text()
    assert (folder / "mean_progress_curves_problem_1.png").exists()
    assert json.loads((folder / "experiment_config.json").read_text())["solvers"]
    with (folder / "results.pkl").open("rb") as f:
        (pairs,) = pickle.load(f)
    assert [experiment.solver.name for experiment, _ in pairs] == ["RNDSRCH", "second"]
    # Experiments of one problem share a Problem instance and are normalized together.
    assert pairs[0][0].problem is pairs[1][0].problem
    assert all(analysis_input.full_df is not None for _, analysis_input in pairs)
    with (folder / "index_maps.pkl").open("rb") as f:
        assert pickle.load(f)["crn_options"] == CrnOptions()


def test_plots_only_rerun_reuses_saved_results(results_dir):
    folder = results_dir / "run"
    folder.mkdir()
    server.run_experiment_async("run", payload([MEAN_PLOT]))
    (folder / "mean_progress_curves_problem_1.png").unlink()
    box = {"plot_type": "BOX", "params": {}}
    server.run_plots_only("run", payload([MEAN_PLOT, box]))
    assert "Complete!" in (folder / "index.html").read_text()
    assert (folder / "mean_progress_curves_problem_1.png").exists()
    assert (folder / "box_progress_curves_problem_1.png").exists()
    with (folder / "index_maps.pkl").open("rb") as f:
        assert pickle.load(f)["needed_solver_indices"] == [0, 1]


def test_plots_only_rerun_falls_back_to_saved_crn(results_dir, monkeypatch):
    folder = results_dir / "run"
    folder.mkdir()
    server.run_experiment_async("run", payload([MEAN_PLOT]))
    with (folder / "index_maps.pkl").open("rb") as f:
        maps = pickle.load(f)
    maps["crn_options"] = CrnOptions(False, True, False)
    with (folder / "index_maps.pkl").open("wb") as f:
        pickle.dump(maps, f)
    seen = []

    def spy(pairs, config, **kwargs):
        seen.append(config.crn_options)
        return save_plots(pairs, config, **kwargs)

    monkeypatch.setattr("simopt.experiment.plot.save_plots", spy)
    server.run_plots_only("run", payload([MEAN_PLOT]))
    assert seen and all(options == CrnOptions(False, True, False) for options in seen)


def test_plots_only_rerun_rejects_legacy_results(results_dir):
    folder = results_dir / "run"
    folder.mkdir()
    (folder / "experiments.pkl").write_bytes(pickle.dumps([]))
    (folder / "index_maps.pkl").write_bytes(pickle.dumps({}))
    server.run_plots_only("run", payload([MEAN_PLOT]))
    assert "re-run the experiment" in (folder / "index.html").read_text()
    assert server._check_rerun_logic({"last_run_id": "run"})
