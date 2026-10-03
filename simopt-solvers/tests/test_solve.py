"""Tests for running solvers on DSL models with ``solve``."""

import subprocess
import sys
import textwrap

import pandas as pd
import pytest
import simopt_dsl as dsl

from simopt_solvers import ADAM, RandomSearch, solve


def _noisy_quadratic() -> dsl.Model:
    model = dsl.Model()
    x = model.add_continuous_vector(-10, 10, name="x", shape=2, initial=[5, -5])

    def run(decisions: dict, rngs: list) -> tuple[dict, dict]:
        x1, x2 = decisions["x"]
        cost = (x1 - 3) ** 2 + (x2 + 1) ** 2 + rngs[0].normalvariate(0, 1)
        return {"cost": cost}, {}

    simulation = model.add_simulation(run=run, decisions={"x": x})
    model.minimize(dsl.mean(simulation.metric("cost")))
    return model


def test_solve_dsl_model() -> None:
    model = _noisy_quadratic()
    model.set_random_solution(lambda rng: (rng.uniform(-10, 10), rng.uniform(-10, 10)))

    result = solve(model, RandomSearch(), budget=500, seed=0)

    assert isinstance(result.history, pd.DataFrame)
    assert result.x is not None
    assert result.x == result.history["solution"].iloc[-1]
    assert all(-10 <= value <= 10 for value in result.x)
    assert 0 < result.budget_used <= 500


def test_solve_gradient_free_continuous_solver() -> None:
    result = solve(_noisy_quadratic(), ADAM(), budget=2000, seed=0)

    assert result.x is not None
    x1, x2 = result.x
    assert abs(x1 - 3) < abs(5 - 3)
    assert abs(x2 + 1) < abs(-5 + 1)


def test_solve_is_reproducible_per_seed() -> None:
    first = solve(_noisy_quadratic(), ADAM(), budget=1000, seed=3)
    second = solve(_noisy_quadratic(), ADAM(), budget=1000, seed=3)
    other = solve(_noisy_quadratic(), ADAM(), budget=1000, seed=4)

    assert first.history.equals(second.history)
    assert not first.history.equals(other.history)


@pytest.mark.parametrize(("budget", "seed"), [(0, 0), (100, -1)])
def test_solve_rejects_invalid_arguments(budget: int, seed: int) -> None:
    with pytest.raises(ValueError):
        solve(_noisy_quadratic(), RandomSearch(), budget=budget, seed=seed)


def test_solving_does_not_import_simopt_benchmark() -> None:
    code = textwrap.dedent(
        """
        import sys

        import simopt_dsl as dsl
        from simopt_solvers import ADAM, solve

        model = dsl.Model()
        x = model.add_continuous_variable(0, 10, initial=5)
        simulation = model.add_simulation(
            run=lambda decisions, rngs: ({"y": (decisions["x"] - 2) ** 2}, {}),
            decisions={"x": x},
        )
        model.minimize(dsl.mean(simulation.metric("y")))
        solve(model, ADAM(), budget=200)

        loaded = sorted(m for m in sys.modules if m == "simopt" or m.startswith("simopt."))
        assert not loaded, loaded
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
