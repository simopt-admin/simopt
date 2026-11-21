"""Smoke tests for constrained ASTRO-DF family solvers on a small constrained model."""

import numpy as np
import simopt_dsl as dsl

from simopt_solvers import CASTRODF, solve


def _constrained_model() -> dsl.Model:
    """min E[(x1-2)^2 + (x2-1)^2 + noise] s.t. x1 + x2 - 2 = 0, x1^2 - 4 <= 0."""
    model = dsl.Model()
    x = model.add_continuous_vector(-5, 5, name="x", shape=2, initial=[0.0, 2.0])

    def run(decisions: dict, rngs: list) -> tuple[dict, dict]:
        x1, x2 = decisions["x"]
        cost = (x1 - 2) ** 2 + (x2 - 1) ** 2 + rngs[0].normalvariate(0, 0.1)
        return {"cost": cost}, {}

    simulation = model.add_simulation(run=run, decisions={"x": x})
    model.minimize(dsl.mean(simulation.metric("cost")))
    model.set_random_solution(lambda rng: (rng.uniform(-1, 1), rng.uniform(-1, 1)))
    model.set_deterministic_constraint_functions(
        equality=lambda x: [x[0] + x[1] - 2.0],
        inequality=lambda x: [x[0] ** 2 - 4.0],
        equality_jacobian=lambda x: np.array([[1.0, 1.0]]),
        inequality_jacobian=lambda x: np.array([[2.0 * x[0], 0.0]]),
        hessian=lambda x: np.array([np.zeros((2, 2)), np.array([[2.0, 0.0], [0.0, 0.0]])]),
    )
    return model


def test_constrained_solver_runs() -> None:
    result = solve(_constrained_model(), CASTRODF(), budget=300, seed=0)

    assert result.x is not None
    assert len(result.history) > 0
    assert 0 < result.budget_used <= 300
