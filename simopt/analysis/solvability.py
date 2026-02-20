"""Shared helpers for solvability analyses."""

import copy
from collections import defaultdict
from collections.abc import Callable
from statistics import quantiles

import numpy as np
import pandas as pd

from simopt.analysis.bootstrap import bootstrap
from simopt.experiment.api import AnalysisInput
from simopt.experiment.single import ProblemSolver
from simopt.options import CrnOptions


def cdf(
    df: pd.DataFrame,
    solve_tolerance: float,
) -> pd.DataFrame:
    """Compute an empirical solvability CDF over normalized budget."""
    solved = df.loc[df["objective"] < solve_tolerance, ["mrep", "budget"]]
    n = df["mrep"].nunique()

    first_hit = solved.groupby("mrep")["budget"].min().reset_index()
    counts = first_hit.groupby("budget")["mrep"].size().rename("value").to_frame().sort_index()
    counts["value"] = counts["value"].cumsum() / n

    budget_points = [0.0, *counts.index, 1.0]
    counts = counts.reindex(budget_points).ffill().fillna(0.0)
    return counts.reset_index()


def _crossing_times(df: pd.DataFrame, solve_tolerance: float) -> list[float]:
    solved = df.loc[df["objective"] < solve_tolerance, ["mrep", "budget"]]
    first_hit = solved.groupby("mrep")["budget"].min()
    # The reindex is needed to handle the case where some mreps did not solve the
    # problem.
    return first_hit.reindex(df["mrep"].unique(), fill_value=np.inf)


def quantile(
    df: pd.DataFrame,
    solve_tolerance: float,
    beta: float,
    budget: float,
) -> pd.DataFrame:
    """Compute a step curve for the beta-quantile solvability profile."""
    crossings = _crossing_times(df, solve_tolerance)
    q = quantiles(crossings, n=100)[int(beta * 99)]
    if np.isinf(q) or np.isnan(q):
        x = [0.0, budget]
        y = [0.0, 0.0]
    elif q == budget:
        x = [0.0, budget]
        y = [0.0, 1.0]
    else:
        x = [0.0, q, budget]
        y = [0.0, 1.0, 1.0]
    return pd.DataFrame({"budget": x, "value": y})


def select_df(analysis_input: AnalysisInput, normalize: bool) -> pd.DataFrame:
    """Select mean data with budget/objective columns for solvability profiles."""
    df = copy.copy(analysis_input.mean_df)
    if not normalize:
        return df
    df["budget"] = df["normalized_budget"]
    df["objective"] = df["normalized_objective"]
    return df


def mean(curves: list[pd.DataFrame]) -> pd.DataFrame:
    """Average step curves over a common set of budgets."""
    stacked = pd.concat(
        [curve.assign(curve_id=i) for i, curve in enumerate(curves)], ignore_index=True
    )
    # TODO: aggfunc="max": if repeated budget rows occur in data, keep the highest
    # attained value. But this may not be necessary.
    pivot = stacked.pivot_table(index="budget", columns="curve_id", values="value", aggfunc="max")
    pivot = pivot.sort_index().ffill().fillna(0.0)
    return pd.DataFrame({"budget": pivot.index.to_numpy(), "value": pivot.mean(axis=1).to_numpy()})


def problem_bootstraps(
    inputs: list[AnalysisInput],
    n_bootstraps: int,
    estimator: Callable[[AnalysisInput], pd.DataFrame],
    crn_options: CrnOptions,
) -> list[list[pd.DataFrame]]:
    """Generate bootstrap curve samples for each problem input."""
    problem_bootstraps = []
    for analysis_input in inputs:
        bootstraps = bootstrap(
            analysis_input,
            n_bootstraps,
            estimator,
            crn_options.across_budget,
            crn_options.across_macroreps,
            crn_options.across_x0_xstar,
        )
        problem_bootstraps.append(bootstraps)
    return problem_bootstraps


def solver_bootstraps(
    problem_bootstraps_data: list[list[pd.DataFrame]], n_bootstraps: int
) -> list[pd.DataFrame]:
    """Aggregate per-problem bootstraps into solver-level bootstrap curves."""
    solver_bootstraps = []
    for i in range(n_bootstraps):
        bootstrap = [problem_bootstraps_data[p][i] for p in range(len(problem_bootstraps_data))]
        solver_bootstraps.append(mean(bootstrap))
    return solver_bootstraps


def group_by_solver(
    analysis_inputs: list[AnalysisInput], experiments: list[ProblemSolver]
) -> tuple[list[str], dict[str, list[AnalysisInput]]]:
    """Group analysis inputs by solver name."""
    solver_to_inputs = defaultdict(list)
    for experiment, analysis_input in zip(experiments, analysis_inputs, strict=True):
        solver = experiment.solver.name
        solver_to_inputs[solver].append(analysis_input)
    return list(solver_to_inputs), solver_to_inputs
