"""Check the server passes its stored (Experiment, AnalysisInput) pairs to save_plots."""

from simopt.experiment import api
from simopt.experiment.experiment import from_names
from simopt.experiment.plot import save_plots
from simopt.web.server import generate_plots


def test_generate_plots_builds_pairs_from_problem_solvers(tmp_path, monkeypatch):
    experiment = from_names("RNDSRCH", "SAN-2", problem_fixed_factors={"budget": 20})
    (analysis_input,) = api.run(
        [experiment], api.SimulationConfig(n_mreps=2, n_preps=2, n_preps_x0_xstar=2), n_jobs=1
    )
    stored = [[(experiment, analysis_input)]]
    records = []

    def spy_save(pairs, config, **kwargs):
        records.append(pairs)
        return save_plots(pairs, config, **kwargs)

    monkeypatch.setattr("simopt.experiment.plot.save_plots", spy_save)
    files = generate_plots(
        [{"plot_type": "MEAN", "params": {"plot_conf_ints": False, "print_max_hw": False}}],
        stored,
        [0],
        [0],
        {0: 0},
        {0: 0},
        [{"name": "RNDSRCH"}],
        [{"name": "SAN-2"}],
        tmp_path,
    )
    assert files and (tmp_path / files[0]).exists()
    (pairs,) = records
    ((used_experiment, used_input),) = pairs
    assert used_experiment is experiment and used_input is analysis_input
