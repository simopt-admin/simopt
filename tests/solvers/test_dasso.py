"""Tests for the DASSO solver."""

from simopt.experiment.api import SimulationConfig, run
from simopt.experiment.experiment import from_names

CONFIG = SimulationConfig(n_mreps=1, n_preps=1, n_preps_x0_xstar=1)


def test_dasso_repeats_same_history_for_same_problem() -> None:
    """Verify DASSO produces repeatable histories for the same problem."""
    (result1,) = run([from_names("DASSO", "EXAMPLE-2")], CONFIG, n_jobs=1)
    (result2,) = run([from_names("DASSO", "EXAMPLE-2")], CONFIG, n_jobs=1)

    for column in ("solution", "budget"):
        assert result1.full_df[column].tolist() == result2.full_df[column].tolist()
