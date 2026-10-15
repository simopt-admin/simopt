"""Tests for ASTRO-DF run-state isolation."""

import pytest
from pandas.testing import assert_frame_equal

from simopt.models.example import ExampleProblem
from simopt_solvers import ASTRODF, solve


@pytest.mark.parametrize("crn_across_solns", [True, False])
@pytest.mark.parametrize("use_gradients", [True, False])
@pytest.mark.parametrize("reuse_points", [True, False])
def test_astrodf_reuses_solver_without_retaining_run_state(
    crn_across_solns, use_gradients, reuse_points
):
    solver = ASTRODF(
        fixed_factors={
            "crn_across_solns": crn_across_solns,
            "use_gradients": use_gradients,
            "reuse_points": reuse_points,
        }
    )
    solver_attributes = set(vars(solver))
    problem = ExampleProblem().optimization_problem
    expected = solve(problem, solver, budget=200, seed=0)

    # Interleave a run with a different dimension and independent random streams.
    other_problem = ExampleProblem(
        fixed_factors={"initial_solution": (1.0, 2.0, 3.0)}
    ).optimization_problem
    solve(other_problem, solver, budget=100, seed=1)
    actual = solve(problem, solver, budget=200, seed=0)

    assert_frame_equal(actual.history, expected.history)
    assert actual.budget_used == expected.budget_used
    assert set(vars(solver)) == solver_attributes
