"""Functions for performing independent evaluations of solutions."""

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.problem import Problem, Solution

from .data import SolverHistorySchema


def _set_up_rngs(
    problem: Problem, mrep: int, crn_across_macroreps: bool
) -> list[MRG32k3a]:
    """Set up RNGs for the macroreplication."""
    if crn_across_macroreps:
        # Use the same RNGs for all macroreps.
        rngs = [
            MRG32k3a(s_ss_sss_index=[0, problem.model.n_rngs + i, 0])
            for i in range(problem.model.n_rngs)
        ]
    else:
        # Use different RNGs for each macroreplication.
        rngs = [
            MRG32k3a(s_ss_sss_index=[0, problem.model.n_rngs * (mrep + 1) + i, 0])
            for i in range(problem.model.n_rngs)
        ]

    return rngs


def _post_replicate_solution(
    problem: Problem,
    x: tuple,
    rngs: list[MRG32k3a],
    n_postreps: int,
    crn_across_budget: bool,
) -> pd.DataFrame:
    """Run post-replications for a single solution.

    Args:
        problem: The simulation-optimization problem instance.
        x: The decision variable vector (solution) to evaluate.
        rngs: List of random number generators to use.
        n_postreps: Number of post-replications to perform.
        crn_across_budget: If True, reuse the RNG state (copy) to maintain CRN
            across different budgets. If False, use the RNGs directly (advancing state).

    Returns:
        pd.DataFrame: A DataFrame containing the results of the post-replications,
            including objective values and stochastic constraints.
    """
    # Attach RNGs for postreplications.
    # If CRN is used across budgets, then we should use a copy rather
    # than passing in the original RNGs.
    solution = Solution(x, problem)
    if crn_across_budget:
        solution.attach_rngs(rngs, copy=True)
    else:
        solution.attach_rngs(rngs, copy=False)
    problem.simulate(solution, num_macroreps=n_postreps)

    data = {
        "rep": list(range(n_postreps)),
        "objective": list(solution.objectives[:, 0]),
        "stochastic_constraints": list(solution.stoch_constraints),
    }
    return pd.DataFrame(data)


def _post_replicate_mrep(
    problem: Problem,
    df: pd.DataFrame,
    n_postreps: int,
    crn_across_macroreps: bool,
    crn_across_budget: bool,
) -> pd.DataFrame:
    """Post-replicate a macroreplication of the problem.

    Iterates through all solutions within a single macroreplication dataframe,
    sets up the appropriate RNGs, and performs post-replications.

    Args:
        problem: The simulation-optimization problem instance.
        df: DataFrame containing solutions for a single macroreplication.
            Must contain 'mrep', 'solution', and 'step' columns.
        n_postreps: Number of post-replications to run.
        crn_across_macroreps: If True, use CRN across macroreplications.
        crn_across_budget: If True, use CRN across time budgets.

    Returns:
        pd.DataFrame: A DataFrame containing combined results for all solutions
            in the macroreplication.
    """
    mrep = df["mrep"].iloc[0]
    rngs = _set_up_rngs(problem, mrep, crn_across_macroreps)

    dfs = []
    for _, row in df.iterrows():
        result = _post_replicate_solution(
            problem, row["solution"], rngs, n_postreps, crn_across_budget
        )
        result["step"] = row["step"]
        dfs.append(result)
    df = pd.concat(dfs, ignore_index=True)
    df["mrep"] = mrep
    return df


def post_replicate(
    problem: Problem,
    df: pd.DataFrame,
    n_postreps: int,
    crn_across_macroreps: bool = False,
    crn_across_budget: bool = True,
) -> tuple[pd.DataFrame, list[float]]:
    """Run independent post-replications for solutions across macroreplications.

    Performs independent evaluations (post-replications) of solutions from a
    solver's run. The function groups solutions by macroreplication and runs
    post-replications in parallel for each macroreplication. Each solution is
    evaluated with the specified number of post-replications to obtain more
    accurate estimates of objective values and stochastic constraints.

    Args:
        problem (Problem): The simulation-optimization problem instance to
            evaluate solutions for.
        df (pd.DataFrame): DataFrame containing solutions to post-replicate.
        n_postreps (int): Number of independent post-replications to run for
            each solution.
        crn_across_macroreps (bool, optional): If True, use Common Random
            Numbers (CRN) across solutions from different macroreplications.
        crn_across_budget (bool, optional): If True, use Common Random Numbers
            (CRN) across solutions from different time budgets/steps within
            the same macroreplication. This reduces variance when comparing
            solutions. Defaults to True.

    Returns:
        tuple[pd.DataFrame, list[float]]: A tuple containing:
            - DataFrame with post-replication results.
            - list[float]: Elapsed time (in seconds) for processing each
                macroreplication, in the same order as macroreplications appear
                in the input DataFrame.
    """
    df = SolverHistorySchema.validate(df)

    def process_macroreplication(
        macroreplication: pd.DataFrame,
    ) -> tuple[pd.DataFrame, float]:
        start = time.perf_counter()
        df = _post_replicate_mrep(
            problem,
            macroreplication,
            n_postreps,
            crn_across_macroreps,
            crn_across_budget,
        )
        elapsed = time.perf_counter() - start
        return df, elapsed

    macroreplications = df.groupby("mrep")
    results = Parallel(n_jobs=-1)(
        delayed(process_macroreplication)(macroreplication)
        for _, macroreplication in macroreplications
    )
    dfs = [df for df, _ in results]
    df = pd.concat(dfs, ignore_index=True)
    elapsed_times = [elapsed for _, elapsed in results]

    return df, elapsed_times


def _to_list(df: pd.DataFrame, column: str) -> list[list]:
    return (
        df.groupby(["mrep", "step"])[column]
        .mean()
        .sort_index()
        .groupby(level="mrep")
        .apply(lambda x: x.to_list())
        .to_list()
    )


def _from_list(data: list[list], column: str) -> pd.DataFrame:
    records = [
        {"mrep": mrep, "step": step, column: value}
        for mrep, steps in enumerate(data)
        for step, value in enumerate(steps)
    ]
    return pd.DataFrame.from_records(records, columns=["mrep", "step", column])


def _to_list_reps(df: pd.DataFrame, column: str) -> list[list[np.ndarray]]:
    return (
        df.sort_values(by=["mrep", "step", "rep"])
        .groupby(["mrep", "step"])[column]
        .apply(np.array)
        .groupby(level="mrep")
        .apply(list)
        .to_list()
    )


def _from_list_reps(data: list[list[np.ndarray]], column: str) -> pd.DataFrame:
    records = []
    for mrep, steps in enumerate(data):
        for step, reps in enumerate(steps):
            for rep, value in enumerate(reps):
                records.append({"mrep": mrep, "step": step, "rep": rep, column: value})
    return pd.DataFrame.from_records(records, columns=["mrep", "step", "rep", column])
