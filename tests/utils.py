import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import zstandard as zstd
from structlog.testing import capture_logs

from simopt.curve import Curve
from simopt.experiment import ProblemSolver

EXPECTED_RESULTS_DIR = Path(__file__).parent / "expected_results"


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


def load_problem_solver(path: str | Path) -> ProblemSolver:
    with zstd.open(path, "rb") as f:
        data = pickle.load(f)

    problem_solver = ProblemSolver(data["solver_name"], data["problem_name"])
    problem_solver.n_macroreps = data["num_macroreps"]
    problem_solver.n_postreps = data["num_postreps"]
    problem_solver.n_postreps_init_opt = data["n_postreps_init_opt"]
    problem_solver.all_recommended_xs = data["all_recommended_xs"]
    problem_solver.all_intermediate_budgets = data["all_intermediate_budgets"]
    problem_solver.all_post_replicates = data["all_post_replicates"]
    problem_solver.all_stoch_constraints = data.get("all_stoch_constraints", [])
    problem_solver.all_est_lhs = data.get("all_est_lhs", [])
    problem_solver.x0 = data["x0"]
    problem_solver.xstar = data["xstar"]
    problem_solver.x0_postreps = data["x0_postreps"]
    problem_solver.xstar_postreps = data["xstar_postreps"]
    problem_solver.objective_curves = [Curve(x, y) for x, y in data["objective_curves"]]
    problem_solver.progress_curves = [Curve(x, y) for x, y in data["progress_curves"]]
    problem_solver.has_run = True
    problem_solver.has_postreplicated = True
    problem_solver.has_postnormalized = True
    return problem_solver


def capture_log_data(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    logger = structlog.get_logger()
    with capture_logs() as logs:
        func(*args, logger=logger, **kwargs)
    return [log["data"] for log in logs if "data" in log]
