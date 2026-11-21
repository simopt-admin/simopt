"""Tests for the constrained SAN problems and constraint hooks."""

import numpy as np
import pytest

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.directory import problem_directory
from simopt.models.amusementpark import AmusementParkMinDepart

CONSTRAINED = ["SAN-3", "SAN-4"]
EQ_INEQ = {
    "SAN-3": (True, False),
    "SAN-4": (False, True),
}


def _fd_jacobian(fn, x, eps=1e-6):
    base = np.atleast_1d(np.asarray(fn(tuple(x)), dtype=float))
    jac = np.zeros((base.size, len(x)))
    for j in range(len(x)):
        xp = np.array(x, dtype=float)
        xp[j] += eps
        xm = np.array(x, dtype=float)
        xm[j] -= eps
        fp = np.atleast_1d(np.asarray(fn(tuple(xp)), dtype=float))
        fm = np.atleast_1d(np.asarray(fn(tuple(xm)), dtype=float))
        jac[:, j] = (fp - fm) / (2 * eps)
    return jac


@pytest.mark.parametrize("abbr", CONSTRAINED)
def test_problem_simulates_and_hooks(abbr):
    problem = problem_directory[abbr]()
    rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
    x = problem.get_random_solution(rng)
    assert len(x) == problem.dim
    # Evaluate a couple of replications.
    rngs = [MRG32k3a(s_ss_sss_index=[2, 0, i]) for i in range(problem.model.n_rngs)]
    for _ in range(2):
        out = problem.optimization_problem.run_replication(x, rngs)
        assert np.isfinite(out.objective)

    has_eq, has_ineq = EQ_INEQ[abbr]
    n_total = 0
    for flag, val_name, jac_name in (
        (
            has_eq,
            "get_deterministic_equality_constraints",
            "get_deterministic_equality_constraints_gradients",
        ),
        (
            has_ineq,
            "get_deterministic_inequality_constraints",
            "get_deterministic_inequality_constraints_gradients",
        ),
    ):
        val = getattr(problem, val_name)
        jac = getattr(problem, jac_name)
        if flag:
            n = np.atleast_1d(val(x)).size
            n_total += n
            assert jac(x).shape == (n, problem.dim)
            np.testing.assert_allclose(jac(x), _fd_jacobian(val, x), rtol=1e-4, atol=1e-4)
        else:
            assert val(x) is None
            assert jac(x) is None
    hess = problem.get_deterministic_constraints_hessian(x)
    assert hess is not None
    assert hess.shape == (n_total, problem.dim, problem.dim)


def test_san_cost_constraint_feasibility():
    # Default arc_costs are 1 and total_cost is 5; x = 13/5 on every arc is on the boundary.
    x = (13 / 5,) * 13
    assert problem_directory["SAN-3"]().check_deterministic_constraints(x)
    assert problem_directory["SAN-4"]().check_deterministic_constraints(x)
    assert problem_directory["SAN-4"]().check_deterministic_constraints((10.0,) * 13)
    assert not problem_directory["SAN-3"]().check_deterministic_constraints((10.0,) * 13)
    assert not problem_directory["SAN-4"]().check_deterministic_constraints((1.0,) * 13)


def test_san4_custom_arcs_costs_total_cost():
    problem = problem_directory["SAN-4"](
        fixed_factors={
            "initial_solution": (4.0,) * 3,
            "arc_costs": (1.0, 2.0, 3.0),
            "total_cost": 4.0,
        },
        model_fixed_factors={
            "num_nodes": 3,
            "arcs": [(1, 2), (2, 3), (1, 3)],
            "arc_means": (1.0, 1.0, 1.0),
        },
    )
    assert problem.dim == 3
    c_ineq = problem.get_deterministic_inequality_constraints((1.0, 2.0, 3.0))
    assert c_ineq is not None
    assert np.atleast_1d(np.asarray(c_ineq))[0] == pytest.approx(1.0 + 1.0 + 1.0 - 4.0)


def test_amusementpark_constraint_hooks():
    problem = AmusementParkMinDepart()
    x = tuple(problem.factors["initial_solution"])
    cap = problem.model.factors["park_capacity"]
    assert problem.get_deterministic_inequality_constraints(x) == (sum(x) - cap,)
    assert problem.get_deterministic_inequality_constraints_gradients(x).shape == (1, problem.dim)
    assert problem.get_deterministic_constraints_hessian(x).shape == (1, problem.dim, problem.dim)
