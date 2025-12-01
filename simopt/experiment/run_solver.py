import logging
import time

import pandas as pd
from joblib import Parallel, delayed

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.problem import Problem
from simopt.solver import Solver


def trim_solver_results(
    problem: Problem,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Trim solver-recommended solutions beyond the problem's maximum budget.

    Args:
        problem (Problem): The problem the solver was run on.
        df (pd.DataFrame): DataFrame with columns ``step``, ``solution`` (Solution)
            and ``budget`` (int) ordered by recommendation time.

    Returns:
        pd.DataFrame: Filtered DataFrame with budgets within the max budget,
        and with a final row at the max budget if one was missing.
    """
    max_budget = problem.factors["budget"]
    trimmed_df = df[df["budget"] <= max_budget].copy()

    # Re-recommend the latest solution at the final budget if needed for plotting.
    if not trimmed_df.empty and trimmed_df["budget"].iloc[-1] < max_budget:
        trimmed_df.loc[len(trimmed_df)] = {
            "step": len(trimmed_df),
            "solution": trimmed_df["solution"].iloc[-1],
            "budget": max_budget,
        }

    return trimmed_df


def to_list(df: pd.DataFrame, column: str) -> list[list]:
    """Convert solver output DataFrame to a nested list grouped by macrorep.

    Args:
        df (pd.DataFrame): DataFrame with columns ``mrep`` and ``step``.
        column (str): Column name to extract.

    Returns:
        list[list]: Outer list ordered by macrorep, inner list ordered by step.
    """
    ordered_df = df.sort_values(["mrep", "step"])
    return [group[column].tolist() for _, group in ordered_df.groupby("mrep")]


def run_multithread(
    mrep: int, solver: Solver, problem: Problem
) -> tuple[pd.DataFrame, float]:
    """Runs one macroreplication of the solver on the problem.

    Args:
        mrep (int): Index of the macroreplication.
        solver (Solver): The simulation-optimization solver to run.
        problem (Problem): The problem to solve.

    Returns:
        tuple: DataFrame of solver output (mrep, step, solution, budget) and runtime
        in seconds.

    Raises:
        ValueError: If `mrep` is negative.
    """
    # Value checking
    if mrep < 0:
        error_msg = "Macroreplication index must be non-negative."
        raise ValueError(error_msg)

    logging.debug(
        f"Macroreplication {mrep + 1}: "
        f"Starting Solver {solver.name} on Problem {problem.name}."
    )
    # Create, initialize, and attach RNGs used for simulating solutions.
    progenitor_rngs = [
        MRG32k3a(s_ss_sss_index=[mrep + 3, ss, 0]) for ss in range(problem.model.n_rngs)
    ]
    # Create a new set of RNGs for the solver based on the current macroreplication.
    # Tried re-using the progentior RNGs, but we need to match the number needed by
    # the solver, not the problem
    solver_rngs = [
        MRG32k3a(
            s_ss_sss_index=[
                mrep + 3,
                problem.model.n_rngs + rng_index,
                0,
            ]
        )
        for rng_index in range(len(solver.rng_list))
    ]

    # Set progenitor_rngs and rng_list for solver.
    solver.solution_progenitor_rngs = progenitor_rngs
    solver.rng_list = solver_rngs

    # logging.debug([rng.s_ss_sss_index for rng in progenitor_rngs])
    # Run the solver on the problem.
    tic = time.perf_counter()
    df = solver.run(problem=problem)
    toc = time.perf_counter()
    runtime = toc - tic
    logging.debug(
        f"Macroreplication {mrep + 1}: "
        f"Finished Solver {solver.name} on Problem {problem.name} "
        f"in {runtime:0.4f} seconds."
    )

    df = trim_solver_results(problem=problem, df=df)

    # Sometimes we end up with numpy scalar values in the solutions,
    # so we convert them to Python scalars. This is especially problematic
    # when trying to dump the solutions to human-readable files as numpy
    # scalars just spit out binary data.
    # TODO: figure out where numpy scalars are coming from and fix it
    df = df.reset_index(drop=True)
    df["solution"] = df["solution"].apply(lambda soln: tuple(float(x) for x in soln.x))
    df["mrep"] = mrep
    df["step"] = df.index
    return df, runtime


def run_solver(
    solver: Solver, problem: Problem, n_macroreps: int, n_jobs: int = -1
) -> tuple[pd.DataFrame, list[float]]:
    """Runs the solver on the problem for a given number of macroreplications.

    Note:
        RNGs for random problem instances are reserved but currently unused.
        This method is under development.

    Args:
        n_macroreps (int): Number of macroreplications to run.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
            -1: use all available cores
            1: run sequentially

    Raises:
        ValueError: If `n_macroreps` is not positive.
    """
    # Local Imports

    # Value checking
    if n_macroreps <= 0:
        error_msg = "Number of macroreplications must be positive."
        raise ValueError(error_msg)

    msg = f"Running Solver {solver.name} on Problem {problem.name}."
    logging.info(msg)

    # Create, initialize, and attach random number generators
    #     Stream 0: reserved for taking post-replications
    #     Stream 1: reserved for bootstrapping
    #     Stream 2: reserved for overhead ...
    #         Substream 0: rng for random problem instance
    #         Substream 1: rng for random initial solution x0 and
    #                      restart solutions
    #         Substream 2: rng for selecting random feasible solutions
    #         Substream 3: rng for solver's internal randomness
    #     Streams 3, 4, ..., n_macroreps + 2: reserved for
    #                                         macroreplications
    # rng0 = MRG32k3a(s_ss_sss_index=[2, 0, 0])  # Currently unused.
    rng_list = [MRG32k3a(s_ss_sss_index=[2, i + 1, 0]) for i in range(3)]
    solver.attach_rngs(rng_list)

    # Start a timer
    function_start = time.time()

    logging.debug("Starting macroreplications")

    if n_jobs == 1:
        results: list[tuple] = [
            run_multithread(i, solver, problem) for i in range(n_macroreps)
        ]
    else:
        results: list[tuple] = Parallel(n_jobs=n_jobs)(
            delayed(run_multithread)(i, solver, problem) for i in range(n_macroreps)
        )  # type: ignore

    timings: list[float] = [0.0 for _ in range(n_macroreps)]
    dfs: list[pd.DataFrame] = []
    for i, (df, timing) in enumerate(results):
        timings[i] = timing
        if not df.empty:
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    runtime = round(time.time() - function_start, 3)
    logging.info(f"Finished running {n_macroreps} mreps in {runtime} seconds.")

    return df, timings
