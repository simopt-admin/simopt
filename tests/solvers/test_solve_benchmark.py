"""Tests for direct solver execution and discovery."""

import simopt_solvers
from simopt.directory import load_solvers, problem_directory
from simopt.experiment.run_solver import run_solver
from simopt_solvers import RandomSearch
from simopt_solvers.solver import solve


def test_solve_matches_benchmark_macroreplication() -> None:
    problem = problem_directory["CNTNEWS-1"](fixed_factors={"budget": 200})
    solver = RandomSearch()

    expected, _ = run_solver(solver, problem, n_macroreps=2, n_jobs=1)
    expected = expected[expected["mrep"] == 1]

    result = solve(problem.optimization_problem, RandomSearch(), budget=200, seed=1)

    n = len(result.history)
    assert result.history["solution"].tolist() == expected["solution"].tolist()[:n]
    assert result.history["budget"].tolist() == expected["budget"].tolist()[:n]


def test_load_solvers_finds_all_solvers() -> None:
    solvers = load_solvers()

    assert set(solvers) == {
        "ADAM",
        "ALOE",
        "ASTRODF",
        "DASSO",
        "FCSA",
        "NELDMD",
        "RNDSRCH",
        "SPSA",
        "STRONG",
    }
    assert all(issubclass(cls, simopt_solvers.Solver) for cls in solvers.values())
