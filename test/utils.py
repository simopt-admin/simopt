import pickle
from collections.abc import Callable
from typing import Any

import structlog
import zstandard as zstd
from structlog.testing import capture_logs

from simopt.curve import Curve
from simopt.experiment import ProblemSolver


def load_problem_solver(path: str) -> ProblemSolver:
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
