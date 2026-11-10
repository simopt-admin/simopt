import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import zstandard as zstd
from structlog.testing import capture_logs

from simopt.experiment import post_replicate, run_solver
from simopt.experiment.api import AnalysisInput
from simopt.experiment.data import ManyPostReplicateSchema, ManySolverHistorySchema
from simopt.experiment.experiment import Experiment, from_names
from simopt.options import CrnOptions

EXPECTED_RESULTS_DIR = Path(__file__).parent / "expected_results"

# Pickle-loaded results carry no CRN settings; plot-log baselines were recorded with all flags off.
PICKLE_CRN_OPTIONS = CrnOptions(False, False, False)


def assert_logs_match(actual, desired):
    """Compare nonempty plot logs record by record, including ragged curves."""
    assert len(actual) > 0, "actual plot logs are empty"
    assert len(desired) > 0, "desired plot logs are empty"
    assert len(actual) == len(desired), (
        f"plot log record count differs: actual {len(actual)}, desired {len(desired)}"
    )
    for index, (a, d) in enumerate(zip(actual, desired, strict=True)):
        _assert_log_values_match(a, d, f"record {index}")


def _assert_log_values_match(a, d, message):
    try:
        actual = np.asarray(a)
        desired = np.asarray(d)
    except ValueError:
        actual = desired = None
    if actual is None or desired is None or actual.dtype == object or desired.dtype == object:
        assert len(a) == len(d), f"{message}: component count differs: {len(a)} != {len(d)}"
        for index, (actual_value, desired_value) in enumerate(zip(a, d, strict=True)):
            _assert_log_values_match(actual_value, desired_value, f"{message}, component {index}")
        return
    assert actual.shape == desired.shape, (
        f"{message}: shape differs: actual {actual.shape}, desired {desired.shape}"
    )
    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5, err_msg=message)


def build_analysis_input(
    all_intermediate_budgets,
    all_recommended_xs,
    all_post_replicates,
    all_stoch_constraints,
    has_stochastic_constraints,
    x0,
    xstar,
    x0_postreps,
    xstar_postreps,
) -> AnalysisInput:
    """Build an AnalysisInput from per-macrorep lists."""
    budget = run_solver._from_list(all_intermediate_budgets, "budget")
    solution = run_solver._from_list(all_recommended_xs, "solution")
    solver_history_df = budget.merge(solution, on=["mrep", "step"])
    solver_history_df["experiment"] = 0
    solver_history_df = ManySolverHistorySchema.validate(solver_history_df)

    objective = post_replicate._from_list_reps(all_post_replicates, "objective")
    if has_stochastic_constraints:
        stochastic_constraints = post_replicate._from_list_reps(
            all_stoch_constraints, "stochastic_constraints"
        )
    else:
        stochastic_constraints = pd.DataFrame(
            {
                "mrep": objective["mrep"],
                "step": objective["step"],
                "rep": objective["rep"],
                "stochastic_constraints": [np.array([])] * len(objective),
            }
        )
    post_replicate_df = objective.merge(stochastic_constraints, on=["mrep", "step", "rep"])
    post_replicate_df["experiment"] = 0
    post_replicate_df = ManyPostReplicateSchema.validate(post_replicate_df)

    full_df = solver_history_df.merge(post_replicate_df, on=["experiment", "mrep", "step"])
    full_df = full_df.set_index(["experiment", "mrep", "step", "rep"])
    # Slice to remove experiment index level (there's only one experiment)
    full_df = full_df.loc[0]

    return AnalysisInput(
        full_df=full_df,
        x0=np.array(x0),
        x0_sample=np.array(x0_postreps, dtype=float),
        xstar=np.array(xstar),
        xstar_sample=np.array(xstar_postreps, dtype=float),
    )


def load_experiment_pair(path: str | Path) -> tuple[Experiment, AnalysisInput]:
    """Load an expected-result pickle as an Experiment and its AnalysisInput."""
    with zstd.open(path, "rb") as f:
        data = pickle.load(f)

    experiment = from_names(data["solver_name"], data["problem_name"])
    analysis_input = build_analysis_input(
        data["all_intermediate_budgets"],
        data["all_recommended_xs"],
        data["all_post_replicates"],
        data.get("all_stoch_constraints", []),
        experiment.problem.n_stochastic_constraints > 0,
        data["x0"],
        data["xstar"],
        data["x0_postreps"],
        data["xstar_postreps"],
    )
    return experiment, analysis_input


def split_pairs(pairs):
    """Split (Experiment, AnalysisInput) pairs into a list of each."""
    return [experiment for experiment, _ in pairs], [data for _, data in pairs]


def capture_log_data(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    logger = structlog.get_logger()
    with capture_logs() as logs:
        func(*args, logger=logger, **kwargs)
    return [log["data"] for log in logs if "data" in log]
