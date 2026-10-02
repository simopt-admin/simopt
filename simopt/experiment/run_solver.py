"""Functions for running solvers and collecting their outputs."""

import logging
import time
from copy import deepcopy

import pandas as pd
from joblib import Parallel, delayed

from simopt.problem import Problem
from simopt.solver import Solver, solve


def _trim(df: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Trim solution history beyond the problem's budget."""
    df = df.loc[df["budget"] <= budget].copy()

    # Add the latest solution as the final row
    if df["budget"].iloc[-1] < budget:
        row = pd.DataFrame.from_records(
            [{"step": len(df), "solution": df["solution"].iloc[-1], "budget": budget}]
        )
        df = pd.concat([df, row], ignore_index=True)

    return df


def _run_solver(solver: Solver, problem: Problem, mrep: int) -> pd.DataFrame:
    result = solve(problem.optimization_problem, solver, problem.factors["budget"], seed=mrep)
    return result.history


def _run_mrep(solver: Solver, problem: Problem, mrep: int) -> tuple[pd.DataFrame, float]:
    """Run one macroreplication of the solver on the problem."""
    logging.debug(
        f"Macroreplication {mrep}: starting solver {solver.name} on problem {problem.name}."
    )

    # Run solver
    start = time.perf_counter()
    df = _run_solver(solver, problem, mrep)
    elapsed = time.perf_counter() - start
    logging.debug(
        f"Macroreplication {mrep}: "
        f"finished solver {solver.name} on problem {problem.name} "
        f"in {elapsed:0.4f} seconds."
    )

    # Trim results to the problem budget and add macroreplication index
    df = _trim(df, problem.factors["budget"])
    df["mrep"] = mrep

    return df, elapsed


def run_solver(
    solver: Solver, problem: Problem, n_macroreps: int, n_jobs: int = -1
) -> tuple[pd.DataFrame, list[float]]:
    """Runs the solver on the problem for a given number of macroreplications.

    Args:
        solver (Solver): The solver to run.
        problem (Problem): The problem to solve.
        n_macroreps (int): Number of macroreplications to run.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
            -1: use all available cores
            1: run sequentially

    Raises:
        ValueError: If `n_macroreps` is not positive.
    """
    if n_macroreps <= 0:
        raise ValueError("number of macroreplications must be positive.")

    logging.info(f"Running solver {solver.name} on problem {problem.name}.")
    logging.debug("Starting macroreplications")

    # TODO: Long-term fix is to make Solver/Problem immutable (stateless)
    # so macroreps can share instances safely without deepcopy.
    if n_jobs == 1:
        results: list[tuple] = [
            _run_mrep(deepcopy(solver), deepcopy(problem), i) for i in range(n_macroreps)
        ]
    else:
        results: list[tuple] = Parallel(n_jobs=n_jobs)(
            delayed(_run_mrep)(deepcopy(solver), deepcopy(problem), i) for i in range(n_macroreps)
        )

    dfs = []
    elapsed_times = []
    for df, elapsed in results:
        dfs.append(df)
        elapsed_times.append(elapsed)
    df = pd.concat(dfs, ignore_index=True)

    return df, elapsed_times


def _to_list(df: pd.DataFrame, column: str) -> list[list]:
    df = df.sort_values(["mrep", "step"])
    return [group[column].tolist() for _, group in df.groupby("mrep")]


def _from_list(data: list[list], column: str) -> pd.DataFrame:
    records = [
        {"mrep": mrep, "step": step, column: value}
        for mrep, steps in enumerate(data)
        for step, value in enumerate(steps)
    ]
    return pd.DataFrame.from_records(records, columns=["mrep", "step", column])
