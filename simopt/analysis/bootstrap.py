"""Bootstrapping procedures."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.experiment.api import AnalysisInput


def _resample_macroreps(
    rng: MRG32k3a, indices: list[int], disable_macrorep_bootstrap: bool
) -> list[int]:
    # Uniformly resample M macroreplications (with replacement) from 0, 1, ..., M - 1.
    # Subsubstream 0: reserved for outer-level bootstrapping.
    # Advance RNG subsubstream to prepare for inner-level bootstrapping.
    if not disable_macrorep_bootstrap:
        indices = rng.choices(indices, k=len(indices))
    rng.advance_subsubstream()
    return indices


def _bootstrap_resample(rng: MRG32k3a, sample: np.ndarray) -> np.ndarray:
    n_samples = len(sample)
    index = np.array(rng.choices(range(n_samples), k=n_samples))
    return sample[index]


def _bootstrap_common(
    rng: MRG32k3a,
    x0_sample: np.ndarray,
    xstar_sample: np.ndarray,
    crn_across_x0_xstar: bool,
) -> tuple[np.ndarray, np.ndarray]:
    # Subsubstream 1: reserved for bootstrapping at x0 and x*.
    # Bootstrap sample post-replicates at common x0.
    # Reset subsubstream if using CRN across budgets.
    # Bootstrap sample postreplicates at reference optimal solution x*.
    # Compute initial optimality gap.
    x0_sample = _bootstrap_resample(rng, x0_sample)
    if crn_across_x0_xstar:
        rng.reset_subsubstream()
    xstar_sample = _bootstrap_resample(rng, xstar_sample)

    # Advance RNG subsubstream to prepare for inner-level bootstrapping.
    rng.advance_subsubstream()

    return x0_sample, xstar_sample


def _bootstrap_mrep(
    df: pd.DataFrame,
    rng: MRG32k3a,
    n_preps: int,
    crn_across_budget: bool,
    crn_across_macroreps: bool,
) -> pd.DataFrame:
    """Bootstrap a single macroreplication."""
    n_steps = len(df) // n_preps
    budgets = df["budget"].to_numpy().reshape(n_steps, n_preps)[:, 0]
    solutions = df["solution"].to_numpy().reshape(n_steps, n_preps)[:, 0]
    objectives = df["objective"].to_numpy().reshape(n_steps, n_preps)
    stochastic_constraints = (
        df["stochastic_constraints"].to_numpy().reshape(n_steps, n_preps)
    )

    use_special_indices = crn_across_budget and not crn_across_macroreps
    special_indices = (
        rng.choices(range(n_preps), k=n_preps) if use_special_indices else None
    )

    data = []
    for i in range(n_steps):
        indices = (
            special_indices
            if use_special_indices
            else rng.choices(range(n_preps), k=n_preps)
        )

        bootstrap_objective = float(np.mean(objectives[i, indices]))

        datum = {
            "step": i,
            "budget": budgets[i],
            "solution": solutions[i],
            "objective": bootstrap_objective,
            "stochastic_constraints": np.stack(stochastic_constraints[i, indices]).mean(
                axis=0
            ),
        }
        data.append(datum)

        if crn_across_budget and not use_special_indices:
            rng.reset_subsubstream()

    if not use_special_indices:
        if crn_across_macroreps:
            rng.reset_subsubstream()
        else:
            rng.advance_subsubstream()

    return pd.DataFrame.from_records(data)


def _bootstrap_sample(
    analysis_input: AnalysisInput,
    rng: MRG32k3a,
    n_preps: int,
    crn_across_budget: bool,
    crn_across_macroreps: bool,
    crn_across_x0_xstar: bool,
    disable_macrorep_bootstrap: bool,
) -> AnalysisInput:
    df = analysis_input.full_df
    mreps = df.index.get_level_values("mrep").unique().to_list()
    bootstrap_mreps = _resample_macroreps(rng, mreps, disable_macrorep_bootstrap)

    x0_sample, xstar_sample = _bootstrap_common(
        rng,
        analysis_input.x0_sample,
        analysis_input.xstar_sample,
        crn_across_x0_xstar,
    )

    dfs = []
    for i, mrep in enumerate(bootstrap_mreps):
        df_mrep = df.xs(mrep, level="mrep")
        bootstrap_df = _bootstrap_mrep(
            df_mrep,
            rng,
            n_preps,
            crn_across_budget,
            crn_across_macroreps,
        )
        bootstrap_df["mrep"] = i
        dfs.append(bootstrap_df)

    full_df = pd.concat(dfs, ignore_index=True)
    return AnalysisInput(
        full_df=full_df,
        x0=analysis_input.x0,
        x0_sample=x0_sample,
        xstar=analysis_input.xstar,
        xstar_sample=xstar_sample,
    )


def _get_n_preps(df: pd.DataFrame) -> int:
    values = df.reset_index().groupby(["mrep", "step"])["rep"].max().add(1).unique()
    if len(values) > 1:
        raise ValueError(
            "number of post replication is not consistent across macroreps and steps."
        )
    return int(values[0])


def bootstrap(
    analysis_input: AnalysisInput,
    n_bootstraps: int,
    f: Callable[[AnalysisInput], Any],
    crn_across_budget: bool,
    crn_across_macroreps: bool,
    crn_across_x0_xstar: bool,
    disable_macrorep_bootstrap: bool = False,
) -> list[Any]:
    """Perform bootstrapping on post-normalization results.

    Args:
        analysis_input: Analysis input containing post-replication data and references.
        n_bootstraps: Number of bootstrap samples to generate.
        f: Function to apply to each bootstrap sample.
            It takes the bootstrap AnalysisInput and returns any result.
        crn_across_budget: Whether to use CRN across budgets.
        crn_across_macroreps: Whether to use CRN across macroreplications.
        crn_across_x0_xstar: Whether to use CRN across initial and optimal solutions.
        disable_macrorep_bootstrap: Whether to disable macroreplication-level
            bootstrapping.

    Returns:
        List of results from applying `f` to each bootstrap sample.
    """
    # Compute n_preps directly from df
    df = analysis_input.full_df
    n_preps = _get_n_preps(df)

    results = []
    for i in range(n_bootstraps):
        # Create random number generator for bootstrap sampling.
        # Stream 1 dedicated for bootstrapping.
        rng = MRG32k3a(s_ss_sss_index=[1, i, 0])
        bootstrap_input = _bootstrap_sample(
            analysis_input,
            rng,
            n_preps,
            crn_across_budget,
            crn_across_macroreps,
            crn_across_x0_xstar,
            disable_macrorep_bootstrap,
        )
        result = f(bootstrap_input)
        results.append(result)

    return results
