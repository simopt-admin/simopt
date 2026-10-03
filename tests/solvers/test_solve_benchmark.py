"""Tests that ``solve`` and the ``simopt`` re-exports agree with the benchmark."""

import simopt_solvers
from simopt.base import Solution, Solver
from simopt.directory import load_solvers, problem_directory
from simopt.experiment.run_solver import run_solver
from simopt.solver import solve
from simopt.solvers import ADAM, RandomSearch


def test_solve_matches_benchmark_macroreplication() -> None:
    problem = problem_directory["CNTNEWS-1"](fixed_factors={"budget": 200})
    solver = RandomSearch()

    expected, _ = run_solver(solver, problem, n_macroreps=2, n_jobs=1)
    expected = expected[expected["mrep"] == 1]

    result = solve(problem.optimization_problem, RandomSearch(), budget=200, seed=1)

    n = len(result.history)
    assert result.history["solution"].tolist() == expected["solution"].tolist()[:n]
    assert result.history["budget"].tolist() == expected["budget"].tolist()[:n]


def test_simopt_paths_reexport_simopt_solvers() -> None:
    assert Solver is simopt_solvers.Solver
    assert Solution is simopt_solvers.Solution
    assert ADAM is simopt_solvers.ADAM
    assert solve is simopt_solvers.solve


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
