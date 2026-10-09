"""Bootstrapping procedures."""

from collections.abc import Callable, Sequence
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
    mrep_data: dict[str, np.ndarray],
    rng: MRG32k3a,
    crn_across_budget: bool,
    crn_across_macroreps: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap a single macroreplication."""
    objectives = mrep_data["objective"]
    stochastic_constraints = mrep_data["stochastic_constraints"]
    n_steps = objectives.shape[0]
    n_preps = objectives.shape[1]

    use_special_indices = crn_across_budget and not crn_across_macroreps
    special_indices = rng.choices(range(n_preps), k=n_preps) if use_special_indices else None

    all_indices = []
    for _ in range(n_steps):
        indices = special_indices if use_special_indices else rng.choices(range(n_preps), k=n_preps)
        all_indices.append(indices)

        if crn_across_budget and not use_special_indices:
            rng.reset_subsubstream()

    if not use_special_indices:
        if crn_across_macroreps:
            rng.reset_subsubstream()
        else:
            rng.advance_subsubstream()

    all_indices = np.array(all_indices)
    steps = np.arange(n_steps)[:, None]
    bootstrap_objective = objectives[steps, all_indices].mean(axis=1)
    bootstrap_stochastic_constraints = stochastic_constraints[steps, all_indices].mean(axis=1)
    return bootstrap_objective, bootstrap_stochastic_constraints


def _transform_bootstrap_data(
    df: pd.DataFrame,
    n_preps: int,
) -> list[dict[str, np.ndarray]]:
    """Precompute per-macrorep arrays used by bootstrap sampling."""
    data: list[dict[str, np.ndarray]] = []
    for mrep in df.index.get_level_values("mrep").unique().to_list():
        df_mrep = df.xs(mrep, level="mrep")
        n_steps = len(df_mrep) // n_preps
        budgets = df_mrep["budget"].to_numpy().reshape(n_steps, n_preps)[:, 0]
        solutions = df_mrep["solution"].to_numpy().reshape(n_steps, n_preps)[:, 0]
        objectives = df_mrep["objective"].to_numpy().reshape(n_steps, n_preps)
        stochastic_constraint_dim = len(df_mrep["stochastic_constraints"].iloc[0])
        stochastic_constraints = np.vstack(df_mrep["stochastic_constraints"].to_numpy()).reshape(
            n_steps, n_preps, stochastic_constraint_dim
        )
        data.append(
            {
                "budget": budgets,
                "solution": solutions,
                "objective": objectives,
                "stochastic_constraints": stochastic_constraints,
            }
        )
    return data


def _bootstrap_sample(
    analysis_input: AnalysisInput,
    data: list[dict[str, np.ndarray]],
    rng: MRG32k3a,
    crn_across_budget: bool,
    crn_across_macroreps: bool,
    crn_across_x0_xstar: bool,
    disable_macrorep_bootstrap: bool,
) -> AnalysisInput:
    mreps = list(range(len(data)))
    bootstrap_mreps = _resample_macroreps(rng, mreps, disable_macrorep_bootstrap)

    x0_sample, xstar_sample = _bootstrap_common(
        rng,
        analysis_input.x0_sample,
        analysis_input.xstar_sample,
        crn_across_x0_xstar,
    )

    objectives = []
    stochastic_constraints = []
    for mrep in bootstrap_mreps:
        bootstrap_objective, bootstrap_stochastic_constraints = _bootstrap_mrep(
            data[mrep],
            rng,
            crn_across_budget,
            crn_across_macroreps,
        )
        objectives.append(bootstrap_objective)
        stochastic_constraints.append(bootstrap_stochastic_constraints)

    n_steps = [len(data[mrep]["budget"]) for mrep in bootstrap_mreps]
    full_df = pd.DataFrame(
        {
            "step": np.concatenate([np.arange(n) for n in n_steps]),
            "budget": np.concatenate([data[mrep]["budget"] for mrep in bootstrap_mreps]),
            "solution": np.concatenate([data[mrep]["solution"] for mrep in bootstrap_mreps]),
            "objective": np.concatenate(objectives),
            "stochastic_constraints": list(np.concatenate(stochastic_constraints)),
            "mrep": np.repeat(np.arange(len(bootstrap_mreps)), n_steps),
        }
    )
    return AnalysisInput(
        full_df=full_df,
        x0=analysis_input.x0,
        x0_sample=x0_sample,
        xstar=analysis_input.xstar,
        xstar_sample=xstar_sample,
        skip_aggregation=True,
    )


def _get_n_preps(df: pd.DataFrame) -> int:
    values = df.reset_index().groupby(["mrep", "step"])["rep"].max().add(1).unique()
    if len(values) > 1:
        raise ValueError("number of post replication is not consistent across macroreps and steps.")
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
    data = _transform_bootstrap_data(df, n_preps)

    results = []
    for i in range(n_bootstraps):
        # Create random number generator for bootstrap sampling.
        # Stream 1 dedicated for bootstrapping.
        rng = MRG32k3a(s_ss_sss_index=[1, i, 0])
        bootstrap_input = _bootstrap_sample(
            analysis_input,
            data,
            rng,
            crn_across_budget,
            crn_across_macroreps,
            crn_across_x0_xstar,
            disable_macrorep_bootstrap,
        )
        result = f(bootstrap_input)
        results.append(result)

    return results


def compute_bootstrap_conf_int(
    observations: Sequence[float] | np.ndarray,
    conf_level: float,
    bias_correction: bool = True,
    overall_estimator: float | None = None,
) -> tuple[float, float]:
    """Construct a bootstrap confidence interval for an estimator.

    Args:
        observations (Sequence[float] | np.ndarray): Estimators from all bootstrap instances.
        conf_level (float): Confidence level for confidence intervals, i.e., 1 - gamma;
            must be in (0, 1).
        bias_correction (bool, optional): Whether to use bias-corrected bootstrap CIs
            (via the percentile method). Defaults to True.
        overall_estimator (float | None, optional): The estimator to compute the CI
            around. Required if`bias_correction` is True.

    Returns:
        tuple[float, float]: A tuple containing the lower and upper bounds of
            the bootstrap confidence interval.

    Raises:
        ValueError: If `conf_level` is not in (0, 1), or if `overall_estimator` is None
            when `bias_correction` is True.
    """
    # Value checking
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)
    if bias_correction and overall_estimator is None:
        error_msg = "Overall estimator must be provided for bias correction."
        raise ValueError(error_msg)

    # This assertion is used during refactoring to ensure that observations is a 1D
    # array.
    values = np.asarray(observations, dtype=float)
    assert values.ndim == 1

    # Compute bootstrapping confidence interval via percentile method.
    # See Efron (1981) "Nonparameteric Standard Errors and Confidence Intervals."
    if bias_correction:
        if overall_estimator is None:
            error_msg = "Overall estimator must be provided for bias correction."
            raise ValueError(error_msg)
        # Lazy imports
        from scipy.stats import norm

        # For biased-corrected CIs, see equation (4.4) on page 146.
        z0 = norm.ppf(np.mean([obs < overall_estimator for obs in values]))
        zconflvl = norm.ppf(conf_level)
        q_lower = norm.cdf(2 * z0 - zconflvl)
        q_upper = norm.cdf(2 * z0 + zconflvl)
    else:
        # For uncorrected CIs, see equation (4.3) on page 146.
        q_lower = (1 - conf_level) / 2
        q_upper = 1 - (1 - conf_level) / 2
    bs_conf_int_lower_bound = np.quantile(values, q=q_lower)
    bs_conf_int_upper_bound = np.quantile(values, q=q_upper)
    return bs_conf_int_lower_bound, bs_conf_int_upper_bound
