"""Tests for the DASSO solver."""

from simopt.experiment_base import ProblemSolver


def test_dasso_repeats_same_history_for_same_problem() -> None:
    """Verify DASSO produces repeatable histories for the same problem."""
    experiment1 = ProblemSolver("DASSO", "EXAMPLE-2")
    experiment1.run(n_macroreps=1)

    experiment2 = ProblemSolver("DASSO", "EXAMPLE-2")
    experiment2.run(n_macroreps=1)

    assert experiment1.all_recommended_xs == experiment2.all_recommended_xs
    assert experiment1.all_intermediate_budgets == experiment2.all_intermediate_budgets
