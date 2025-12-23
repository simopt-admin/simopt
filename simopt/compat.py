"""Compatibility functions between old and new experiment APIs."""

import numpy as np
import pandas as pd

from simopt.experiment import post_replicate, run_solver
from simopt.experiment.api import Result
from simopt.experiment.data import ManyPostReplicateSchema, ManySolverHistorySchema
from simopt.experiment.post_normalize import NormalizationResult
from simopt.experiment.single import ProblemSolver
from simopt.utils import make_nonzero


def _validate_experiment(experiment: ProblemSolver) -> None:
    if not experiment.has_run:
        raise ValueError("experiment must be run before conversion.")
    if not experiment.has_postreplicated:
        raise ValueError("experiment must be post-replicated before conversion.")
    if not experiment.has_postnormalized:
        raise ValueError("experiment must be post-normalized before conversion.")


def _build_solver_history_df(experiment: ProblemSolver) -> pd.DataFrame:
    budget = run_solver._from_list(experiment.all_intermediate_budgets, "budget")
    solution = run_solver._from_list(experiment.all_recommended_xs, "solution")
    df = budget.merge(solution, on=["mrep", "step"])
    df["experiment"] = 0
    return ManySolverHistorySchema.validate(df)


def _build_post_replicate_df(experiment: ProblemSolver) -> pd.DataFrame:
    objective = post_replicate._from_list_reps(
        experiment.all_post_replicates, "objective"
    )

    if experiment.problem.n_stochastic_constraints > 0:
        stochastic_constraints = post_replicate._from_list_reps(
            experiment.all_stoch_constraints, "stochastic_constraints"
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

    df = objective.merge(stochastic_constraints, on=["mrep", "step", "rep"])
    df["experiment"] = 0
    return ManyPostReplicateSchema.validate(df)


def _build_normalization_result(experiment: ProblemSolver) -> NormalizationResult:
    x0_sample = np.array(experiment.x0_postreps, dtype=float)
    xstar_sample = np.array(experiment.xstar_postreps, dtype=float)
    initial_objective = float(np.mean(x0_sample))
    optimal_objective = float(np.mean(xstar_sample))
    initial_gap = make_nonzero(initial_objective - optimal_objective, "initial_gap")

    return NormalizationResult(
        x0=experiment.x0,
        x0_sample=x0_sample,
        xstar=experiment.xstar,
        xstar_sample=xstar_sample,
        initial_objective=initial_objective,
        optimal_objective=optimal_objective,
        initial_gap=initial_gap,
    )


def convert(experiment: ProblemSolver) -> Result:
    """Convert a ProblemSolver object to a Result object."""
    _validate_experiment(experiment)

    solver_history_df = _build_solver_history_df(experiment)
    post_replicate_df = _build_post_replicate_df(experiment)

    full_df = solver_history_df.merge(
        post_replicate_df, on=["experiment", "mrep", "step"]
    )
    full_df = full_df.set_index(["experiment", "mrep", "step", "rep"])

    normalization_result = _build_normalization_result(experiment)

    return Result(
        experiments=[experiment],
        solver_history_df=solver_history_df,
        post_replicate_df=post_replicate_df,
        full_df=full_df,
        normalization_result=normalization_result,
    )
