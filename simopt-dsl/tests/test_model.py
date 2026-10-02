"""Tests for model declarations and replication evaluation."""

import math
import random
import warnings

import pytest

from simopt_dsl import Model


def test_vector_component_does_not_alias_scalar_with_same_name() -> None:
    model = Model()
    vector = model.add_continuous_vector(0, 10, name="x", shape=1)
    scalar = model.add_continuous_variable(0, 10, name="x[0]")
    model.minimize(vector[0])

    assert vector[0] != scalar

    evaluation = model.run_replication((2, 7), [object()])

    assert evaluation.objective == 2
    assert evaluation.objective_gradient == (1, 0)


def _gradient_model(*, gradients: bool) -> Model:
    model = Model()
    x = model.add_continuous_variable(0, 10, name="x")

    def run(decisions: dict, rngs: list) -> tuple[dict, dict]:
        return {"y": decisions["x"] ** 2}, {"y": {"x": 2 * decisions["x"]}}

    simulation = model.add_simulation(run=run, decisions={"x": x}, gradients=gradients)
    model.minimize(simulation.metric("y"))
    return model


def test_gradient_available_follows_simulation_flag() -> None:
    assert _gradient_model(gradients=True).gradient_available
    assert not _gradient_model(gradients=False).gradient_available


def test_gradient_available_without_simulation_metrics() -> None:
    model = Model()
    x = model.add_continuous_variable(0, 10)
    model.minimize(2 * x)

    assert model.gradient_available


def test_gradient_available_ignores_simulations_outside_objective() -> None:
    model = Model()
    x = model.add_continuous_variable(0, 10, name="x")

    def run(decisions: dict, rngs: list) -> tuple[dict, dict]:
        return {"y": decisions["x"]}, {}

    used = model.add_simulation(
        name="used",
        run=lambda decisions, rngs: ({"y": decisions["x"]}, {"y": {"x": 1.0}}),
        decisions={"x": x},
        gradients=True,
    )
    unused = model.add_simulation(name="unused", run=run, decisions={"x": x})
    model.minimize(used.metric("y"))
    model.add_stochastic_constraint(unused.metric("y") <= 5)

    assert model.gradient_available


def test_gradient_available_without_objective() -> None:
    assert not Model().gradient_available


def test_random_solution_uses_sampler() -> None:
    model = Model()
    model.add_continuous_vector(0, 10, shape=2)
    model.set_random_solution(lambda rng: [rng.uniform(0, 1), 5])

    solution = model.random_solution(random.Random(0))

    assert isinstance(solution, tuple)
    assert solution[1] == 5

    model.set_random_solution(None)
    assert model._random_solution is None


def test_random_solution_default_respects_bounds_and_integers() -> None:
    model = Model()
    model.add_continuous_variable(-1, 1)
    model.add_integer_vector(0, 3, shape=3)
    rng = random.Random(0)

    for _ in range(100):
        x, *ks = model.random_solution(rng)
        assert -1 <= x <= 1
        assert all(k in (0, 1, 2, 3) for k in ks)


def test_random_solution_default_respects_deterministic_constraints() -> None:
    model = Model()
    x = model.add_continuous_variable(0, 10)
    y = model.add_continuous_variable(0, 10)
    model.add_linear_constraint(x + y <= 5)
    rng = random.Random(0)

    for _ in range(50):
        a, b = model.random_solution(rng)
        assert a + b <= 5


def test_random_solution_default_rejects_infinite_bounds() -> None:
    model = Model()
    model.add_continuous_variable(0, math.inf)

    with pytest.raises(ValueError, match="infinite bound"):
        model.random_solution(random.Random(0))


def test_derivatives_without_gradients_flag_warn() -> None:
    model = _gradient_model(gradients=False)

    with pytest.warns(UserWarning, match="gradients=False"):
        evaluation = model.run_replication((3,), [object()])

    assert evaluation.objective_gradient == (6,)


def test_derivatives_with_gradients_flag_do_not_warn() -> None:
    model = _gradient_model(gradients=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.run_replication((3,), [object()])
