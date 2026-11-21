"""Merit and deterministic feasibility curves computed from analysis inputs."""

import numpy as np
import pandas as pd

from simopt.deterministic_feasibility import merit_from_objective_and_violation
from simopt.experiment.api import AnalysisInput
from simopt.options import MeritOptions
from simopt.utils import make_nonzero


def require_deterministic_violation(analysis_input: AnalysisInput) -> pd.DataFrame:
    """Return the mean data sorted by (mrep, step), requiring recorded violations."""
    df = analysis_input.mean_df
    if "deterministic_violation" not in df:
        raise ValueError(
            "analysis input has no deterministic violations; run the experiment on a problem "
            "with deterministic constraints or use api.add_deterministic_violation."
        )
    return df.sort_values(["mrep", "step"], kind="stable")


def merit_progress(analysis_input: AnalysisInput, options: MeritOptions) -> pd.DataFrame:
    """Normalized merit progress curves with columns mrep, step, budget and objective.

    Per macroreplication, merit is objective + obj_const * violation (inf above
    `feas_tol_upper`), made monotone by a running best, then normalized by the merit gap
    between x0 and the best merit solution. Budget is the fraction of the budget used.
    """
    normalization = analysis_input.merit_normalization
    if normalization is None:
        raise ValueError("run post_normalize_merit before using merit progress curves.")
    if normalization.options != options:
        raise ValueError("merit options differ from those used in post_normalize_merit.")

    def merit(objective: float, violation: float) -> float:
        return merit_from_objective_and_violation(
            objective, violation, options.obj_const, options.feas_tol_lower, options.feas_tol_upper
        )

    df = require_deterministic_violation(analysis_input)

    # Initial merit at x0 from its (possibly bootstrapped) post-replications.
    if normalization.proxy_init_merit is not None:
        initial_merit = normalization.proxy_init_merit
    else:
        initial_merit = merit(float(np.mean(analysis_input.x0_sample)), normalization.x0_violation)

    # Best merit m*, from the (possibly bootstrapped) post-replications at its solution.
    if analysis_input.xstar_merit_sample is not None:
        if normalization.xstar_merit_violation is None:
            raise ValueError("xstar_merit_sample is set but xstar_merit_violation is missing.")
        best_merit = merit(
            float(np.mean(analysis_input.xstar_merit_sample)),
            float(normalization.xstar_merit_violation),
        )
    else:
        best_merit = normalization.best_merit

    initial_merit_gap = make_nonzero(float(initial_merit - best_merit), "initial_merit_gap")

    raw_merit = np.array(
        [
            merit(float(obj), float(feas))
            for obj, feas in zip(df["objective"], df["deterministic_violation"], strict=True)
        ]
    )
    # Merit must not increase after (no points for a momentary blip of good value).
    raw_merit = pd.Series(raw_merit, index=df.index).groupby(df["mrep"]).cummin().to_numpy()

    finite = np.isfinite(raw_merit)
    norm_merit = np.ones_like(raw_merit)
    norm_merit[finite] = np.maximum((raw_merit[finite] - best_merit) / initial_merit_gap, 0.0)

    return pd.DataFrame(
        {
            "mrep": df["mrep"].to_numpy(),
            "step": df["step"].to_numpy(),
            "budget": df["normalized_budget"].to_numpy(),
            "objective": norm_merit,
        }
    )


def deterministic_feasibility_scores(
    analysis_input: AnalysisInput,
    score_type: str,
    obj_const: float,
    feas_tol_upper: float,
    feas_tol_lower: float,
    floor_feas: bool = False,
) -> pd.DataFrame:
    """Deterministic feasibility history with columns mrep, step, budget, objective, score.

    score_type "norm" is the aggregate violation (floored at `feas_tol_lower` if
    `floor_feas`); "objective" is the penalized objective with violations at or below
    `feas_tol_lower` counted as 0 and above `feas_tol_upper` as infinite.
    """
    df = require_deterministic_violation(analysis_input)
    feas = df["deterministic_violation"].to_numpy(dtype=float)
    if score_type == "norm":
        score = np.where(feas <= feas_tol_lower, feas_tol_lower, feas) if floor_feas else feas
    elif score_type == "objective":
        filtered = np.where(
            feas <= feas_tol_lower, 0.0, np.where(feas > feas_tol_upper, np.inf, feas)
        )
        score = df["objective"].to_numpy(dtype=float) + obj_const * filtered
    else:
        raise ValueError("score_type must be 'norm' or 'objective'.")
    return pd.DataFrame(
        {
            "mrep": df["mrep"].to_numpy(),
            "step": df["step"].to_numpy(),
            "budget": df["budget"].to_numpy(),
            "objective": df["objective"].to_numpy(),
            "score": score,
        }
    )
