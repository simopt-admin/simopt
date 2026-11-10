"""Merit-based post normalization (objective + penalty * feasibility violation)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.deterministic_feasibility import (
    deterministic_violation,
    merit_from_objective_and_violation,
)
from simopt.experiment.post_normalize import _simulate
from simopt.options import MeritOptions
from simopt.problem import Problem


@dataclass(frozen=True)
class MeritNormalization:
    """Merit normalization constants shared by experiments on the same problem.

    Attributes:
        options: Merit options the constants were computed with.
        x0_violation: Deterministic violation at x0.
        best_merit: (Proxy for) the best merit m*; fixed if `xstar_merit` is None.
        xstar_merit: Empirically best merit solution, or None if m* was supplied.
        xstar_merit_violation: Deterministic violation at `xstar_merit`.
        proxy_init_merit: Supplied merit at x0, overriding the one from `x0_sample`.
    """

    options: MeritOptions
    x0_violation: float
    best_merit: float
    xstar_merit: tuple | None = None
    xstar_merit_violation: float | None = None
    proxy_init_merit: float | None = None


def has_deterministic_constraints(problem: Problem) -> bool:
    """Whether the problem overrides the deterministic constraint hooks of `Problem`.

    `Problem` defines the hooks returning None (no constraints), so merely having the
    attributes does not mean the problem has constraints.
    """
    names = ("get_deterministic_equality_constraints", "get_deterministic_inequality_constraints")
    return any(
        getattr(type(problem), name, None) is not None
        and getattr(type(problem), name) is not getattr(Problem, name)
        for name in names
    )


def attach_deterministic_violation(
    problem: Problem, df: pd.DataFrame, required: bool = False
) -> pd.DataFrame:
    """Add a `deterministic_violation` column (violation of each row's solution) to `df`."""
    if not has_deterministic_constraints(problem):
        if required:
            raise ValueError("problem does not provide deterministic constraints.")
        return df
    violations = {
        solution: deterministic_violation(problem, solution) for solution in df["solution"].unique()
    }
    df = df.copy()
    df["deterministic_violation"] = df["solution"].map(violations).astype(float)
    return df


def _best_merit_with_feasibility(
    problem: Problem,
    dfs: list[pd.DataFrame],
    rngs: list[MRG32k3a],
    n_reps: int,
    options: MeritOptions,
) -> tuple[tuple, float, float, np.ndarray]:
    """Find the empirically best merit solution and re-simulate it at high precision.

    Rather than trusting the single noisy objective estimate that made a solution look
    best across every (experiment, mrep, step) triple, re-simulate that winning solution
    with `n_reps` fresh replications to get a low-bias estimate of its true merit.
    """
    best_merit_so_far = np.inf
    best_x = None

    for df in dfs:
        mean_df = (
            df.groupby(["mrep", "step"])
            .agg({"objective": "mean", "solution": "first", "deterministic_violation": "first"})
            .reset_index()
        )
        for obj, x, feas in zip(
            mean_df["objective"],
            mean_df["solution"],
            mean_df["deterministic_violation"],
            strict=True,
        ):
            m = merit_from_objective_and_violation(
                float(obj), feas, options.obj_const, options.feas_tol_lower, options.feas_tol_upper
            )
            if m < best_merit_so_far:
                best_merit_so_far = m
                best_x = x

    if best_x is None or not np.isfinite(best_merit_so_far):
        error_msg = "No feasible-enough solutions found for which to estimate proxy m*."
        raise RuntimeError(error_msg)

    # Re-simulate the winning solution at high precision, correcting the optimistic
    # bias from taking a min() over many noisy objective estimates.
    xstar_merit_postreps = _simulate(problem, best_x, rngs, n_reps)

    refined_obj = float(np.mean(xstar_merit_postreps))
    refined_feas = deterministic_violation(problem, best_x)
    refined_merit = merit_from_objective_and_violation(
        refined_obj, refined_feas, options.obj_const, options.feas_tol_lower, options.feas_tol_upper
    )

    return best_x, refined_feas, refined_merit, xstar_merit_postreps


def normalize_merit(
    problem: Problem,
    dfs: list[pd.DataFrame],
    x0: tuple | np.ndarray,
    n_reps: int,
    options: MeritOptions,
    proxy_init_merit: float | None = None,
    proxy_opt_merit: float | None = None,
) -> tuple[MeritNormalization, np.ndarray | None]:
    """Computes merit normalization constants for experiments on the same problem.

    Args:
        problem: The shared problem.
        dfs: Per-experiment data indexed by (mrep, step, rep), with a
            `deterministic_violation` column.
        x0: Initial solution.
        n_reps: Number of post-replications at the best merit solution.
        options: Penalty multiplier and violation tolerances.
        proxy_init_merit: Known/override merit at x0.
        proxy_opt_merit: Known/override best merit m*. If provided, skips the
            empirical search and re-simulation step entirely.

    Returns:
        The normalization constants and the post-replications at the best merit
        solution (None if `proxy_opt_merit` was provided).
    """
    x0_violation = deterministic_violation(problem, tuple(x0))

    xstar_merit = None
    xstar_merit_violation = None
    xstar_merit_sample = None
    if proxy_opt_merit is not None:
        best_merit = proxy_opt_merit
    else:
        # Stream 2: reserved for merit-optimal re-simulation (stream 0 is used by
        # normalize for x0/xstar, stream 1 by bootstrapping).
        rngs = [
            MRG32k3a(s_ss_sss_index=[2, problem.model.n_rngs + rng_index, 0])
            for rng_index in range(problem.model.n_rngs)
        ]
        xstar_merit, xstar_merit_violation, best_merit, xstar_merit_sample = (
            _best_merit_with_feasibility(problem, dfs, rngs, n_reps, options)
        )

    return (
        MeritNormalization(
            options=options,
            x0_violation=x0_violation,
            best_merit=best_merit,
            xstar_merit=xstar_merit,
            xstar_merit_violation=xstar_merit_violation,
            proxy_init_merit=proxy_init_merit,
        ),
        xstar_merit_sample,
    )
