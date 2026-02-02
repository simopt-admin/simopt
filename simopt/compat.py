"""Compatibility functions between old and new experiment APIs."""

import numpy as np
import pandas as pd

from simopt.experiment import post_replicate, run_solver
from simopt.experiment.api import AnalysisInput
from simopt.experiment.data import ManyPostReplicateSchema, ManySolverHistorySchema
from simopt.experiment.single import ProblemSolver


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


def convert(experiment: ProblemSolver) -> AnalysisInput:
    """Convert a ProblemSolver object to an AnalysisInput object."""
    _validate_experiment(experiment)

    solver_history_df = _build_solver_history_df(experiment)
    post_replicate_df = _build_post_replicate_df(experiment)

    full_df = solver_history_df.merge(
        post_replicate_df, on=["experiment", "mrep", "step"]
    )
    full_df = full_df.set_index(["experiment", "mrep", "step", "rep"])
    # Slice to remove experiment index level (there's only one experiment)
    full_df = full_df.loc[0]

    x0_sample = np.array(experiment.x0_postreps, dtype=float)
    xstar_sample = np.array(experiment.xstar_postreps, dtype=float)

    return AnalysisInput(
        full_df=full_df,
        x0=np.array(experiment.x0),
        x0_sample=x0_sample,
        xstar=np.array(experiment.xstar),
        xstar_sample=xstar_sample,
    )
