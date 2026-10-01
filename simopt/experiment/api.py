# ruff: noqa: D101, D103
"""API for running simulation optimization experiments."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydantic import BaseModel

from simopt.directory import problem_directory, solver_directory
from simopt.experiment.data import (
    ManyPostReplicateSchema,
    ManySolverHistorySchema,
)
from simopt.experiment.post_normalize import normalize
from simopt.experiment.post_replicate import _post_replicate_mrep
from simopt.experiment.run_solver import _run_mrep
from simopt.experiment.single import ProblemSolver
from simopt.options import DEFAULT_CRN_OPTIONS, CrnOptions
from simopt.problem import Problem
from simopt.solver import Solver
from simopt.utils import make_nonzero


class SolverConfig(BaseModel):
    name: str
    id: str | None = None
    fixed_factors: dict | None = None


class ProblemConfig(BaseModel):
    name: str
    id: str | None = None
    fixed_factors: dict | None = None
    model_fixed_factors: dict | None = None


class SimulationConfig(BaseModel):
    n_mreps: int
    n_preps: int
    n_preps_x0_xstar: int


class ProxyValues(BaseModel):
    initial_objective: float | None = None
    xstar: tuple | None = None
    optimal_objective: float | None = None


DEFAULT_PROXY_VALUES = ProxyValues()


@dataclass(frozen=True)
class PlotConfig:
    """Base class for plot configuration dataclasses."""


def to_solver(config: dict) -> Solver:
    config_model = SolverConfig(**config)
    return solver_directory[config_model.name](fixed_factors=config_model.fixed_factors)


def to_problem(config: dict) -> Problem:
    config_model = ProblemConfig(**config)
    return problem_directory[config_model.name](
        name=config_model.name,
        fixed_factors=config_model.fixed_factors,
        model_fixed_factors=config_model.model_fixed_factors,
    )


def validate_solvers(solvers: list[dict]) -> list[Solver]:
    return [to_solver(solver) for solver in solvers]


def validate_problems(problems: list[dict]) -> list[Problem]:
    return [to_problem(problem) for problem in problems]


def create_matrix(solvers: list[Solver], problems: list[Problem]) -> list[ProblemSolver]:
    return [
        ProblemSolver(solver=solver, problem=problem) for solver in solvers for problem in problems
    ]


def _mean(
    full_df: pd.DataFrame,
    x0: np.ndarray,
    x0_sample: np.ndarray,
    xstar: np.ndarray,
    xstar_sample: np.ndarray,
    skip_aggregation: bool = False,
) -> tuple[float, pd.DataFrame]:
    if skip_aggregation:
        df_mean = full_df
    else:
        df_mean = (
            full_df.groupby(["mrep", "step"])
            .agg(
                {
                    "budget": "first",
                    "solution": "first",
                    "objective": "mean",
                    "stochastic_constraints": "mean",
                }
            )
            .reset_index()
        )

    initial_objective = float(np.mean(x0_sample))
    optimal_objective = float(np.mean(xstar_sample))
    initial_gap = make_nonzero(initial_objective - optimal_objective, "initial_gap")
    df_mean.loc[df_mean["solution"] == tuple(x0), "objective"] = initial_objective
    df_mean.loc[df_mean["solution"] == tuple(xstar), "objective"] = optimal_objective

    budget = float(df_mean["budget"].max())
    df_mean["normalized_budget"] = df_mean["budget"] / budget
    df_mean["normalized_objective"] = (df_mean["objective"] - optimal_objective) / initial_gap

    return budget, df_mean


class AnalysisInput:
    def __init__(
        self,
        full_df: pd.DataFrame,
        x0: np.ndarray,
        x0_sample: np.ndarray,
        xstar: np.ndarray,
        xstar_sample: np.ndarray,
        skip_aggregation: bool = False,
    ) -> None:
        """Initialize AnalysisInput with data and compute mean statistics."""
        self.full_df = full_df
        self.x0 = x0
        self.x0_sample = x0_sample
        self.xstar = xstar
        self.xstar_sample = xstar_sample
        self.budget, self.mean_df = _mean(
            full_df, x0, x0_sample, xstar, xstar_sample, skip_aggregation
        )


@dataclass(frozen=True)
class ExperimentPlan:
    """A problem comparison group ready for macroreplication scheduling."""

    problem: Problem
    solvers: tuple[Solver, ...]
    simulation_config: SimulationConfig
    crn_options: CrnOptions
    proxy_values: ProxyValues


def _copy_problem(problem: Problem) -> Problem:
    copied_problem = deepcopy(problem)
    # Cached DSL callbacks close over the source Problem; rebuild them on the copy.
    copied_problem._optimization_problem = None
    return copied_problem


def plan_experiment(
    problem: Problem,
    solvers: list[Solver],
    simulation_config: SimulationConfig,
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS,
    proxy_values: ProxyValues | None = None,
) -> ExperimentPlan:
    """Snapshot a problem, solvers, and settings for one comparison group.

    Planning does not run the solvers or simulate the problem. All solvers in
    a plan share the normalization result computed after their jobs finish.
    """
    if not solvers:
        raise ValueError("solvers must not be empty.")
    if simulation_config.n_mreps <= 0:
        raise ValueError("number of macroreplications must be positive.")
    if simulation_config.n_preps <= 0 or simulation_config.n_preps_x0_xstar <= 0:
        raise ValueError("numbers of post-replications must be positive.")

    problem_snapshot = _copy_problem(problem)

    return ExperimentPlan(
        problem=problem_snapshot,
        solvers=tuple(deepcopy(solver) for solver in solvers),
        simulation_config=simulation_config.model_copy(deep=True),
        crn_options=deepcopy(crn_options),
        proxy_values=(proxy_values or DEFAULT_PROXY_VALUES).model_copy(deep=True),
    )


def _run_planned_mrep(
    problem: Problem,
    solver: Solver,
    mrep: int,
    n_preps: int,
    crn_options: CrnOptions,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    solver_history_df, _ = _run_mrep(deepcopy(solver), _copy_problem(problem), mrep)
    post_replicate_df = _post_replicate_mrep(
        _copy_problem(problem),
        solver_history_df,
        n_preps,
        crn_options.across_macroreps,
        crn_options.across_budget,
    )
    return solver_history_df, post_replicate_df


def _analyze_plan(
    plan: ExperimentPlan,
    solver_history_dfs: list[list[pd.DataFrame]],
    post_replicate_dfs: list[list[pd.DataFrame]],
) -> list[AnalysisInput]:
    solver_histories = []
    post_replicates = []
    for i in range(len(plan.solvers)):
        solver_history_df = pd.concat(solver_history_dfs[i], ignore_index=True)
        solver_history_df["experiment"] = i
        solver_histories.append(solver_history_df)

        post_replicate_df = pd.concat(post_replicate_dfs[i], ignore_index=True)
        post_replicate_df["experiment"] = i
        post_replicates.append(post_replicate_df)

    many_solver_history_df = pd.concat(solver_histories, ignore_index=True)
    many_solver_history_df = ManySolverHistorySchema.validate(many_solver_history_df)
    many_post_replicate_df = pd.concat(post_replicates, ignore_index=True)
    many_post_replicate_df = ManyPostReplicateSchema.validate(many_post_replicate_df)

    full_df = many_solver_history_df.merge(
        many_post_replicate_df, on=["experiment", "mrep", "step"]
    )
    full_df = full_df.set_index(["experiment", "mrep", "step", "rep"])

    normalization_result = normalize(
        _copy_problem(plan.problem),
        full_df,
        plan.simulation_config.n_preps_x0_xstar,
        plan.crn_options.across_x0_xstar,
        plan.proxy_values.initial_objective,
        plan.proxy_values.xstar,
        plan.proxy_values.optimal_objective,
    )

    # Build list of AnalysisInput objects
    analysis_inputs = []
    for i in range(len(plan.solvers)):
        sliced_full_df = full_df.loc[i]
        analysis_inputs.append(
            AnalysisInput(
                full_df=sliced_full_df,
                x0=np.array(normalization_result.x0),
                x0_sample=normalization_result.x0_sample,
                xstar=np.array(normalization_result.xstar),
                xstar_sample=normalization_result.xstar_sample,
            )
        )
    return analysis_inputs


def _run_plans(plans: list[ExperimentPlan], n_jobs: int) -> list[AnalysisInput]:
    if not plans:
        return []

    jobs = [
        (plan_index, solver_index, mrep)
        for plan_index, plan in enumerate(plans)
        for solver_index in range(len(plan.solvers))
        for mrep in range(plan.simulation_config.n_mreps)
    ]
    if n_jobs == 1:
        results = [
            _run_planned_mrep(
                plans[plan_index].problem,
                plans[plan_index].solvers[solver_index],
                mrep,
                plans[plan_index].simulation_config.n_preps,
                plans[plan_index].crn_options,
            )
            for plan_index, solver_index, mrep in jobs
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_planned_mrep)(
                plans[plan_index].problem,
                plans[plan_index].solvers[solver_index],
                mrep,
                plans[plan_index].simulation_config.n_preps,
                plans[plan_index].crn_options,
            )
            for plan_index, solver_index, mrep in jobs
        )

    solver_history_dfs = [[[] for _ in plan.solvers] for plan in plans]
    post_replicate_dfs = [[[] for _ in plan.solvers] for plan in plans]
    for (plan_index, solver_index, _), (solver_history_df, post_replicate_df) in zip(
        jobs, results, strict=True
    ):
        solver_history_dfs[plan_index][solver_index].append(solver_history_df)
        post_replicate_dfs[plan_index][solver_index].append(post_replicate_df)

    analysis_inputs = []
    for plan_index, plan in enumerate(plans):
        analysis_inputs.extend(
            _analyze_plan(plan, solver_history_dfs[plan_index], post_replicate_dfs[plan_index])
        )
    return analysis_inputs


def run(
    experiments: list[ProblemSolver],
    simulation_config: SimulationConfig,
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS,
    proxy_values: ProxyValues | None = None,
    n_jobs: int = -1,
) -> list[AnalysisInput]:
    """Run experiments and return analysis inputs.

    Experiments are grouped by problem instance and planned internally before
    dispatch through one shared macroreplication pool. Each group shares a
    normalization result.

    Args:
        experiments: List of ProblemSolver experiments to run.
        simulation_config: Configuration for the simulation (n_mreps, n_preps, etc.).
        crn_options: Options for common random numbers.
        proxy_values: Optional proxy values for normalization.
        n_jobs: Number of parallel jobs to run.

    Returns:
        A list of AnalysisInput objects, one per experiment, in the same order as input.
    """
    if not experiments:
        return []

    # Group experiments by problem
    problem_groups = {}
    for i, exp in enumerate(experiments):
        problem_key = id(exp.problem)  # Group by problem instance
        if problem_key not in problem_groups:
            problem_groups[problem_key] = []
        problem_groups[problem_key].append((i, exp))

    # Run experiments for each problem group
    plans = []
    indices_by_group = []
    for group in problem_groups.values():
        indices, exps = zip(*group, strict=True)
        plan = plan_experiment(
            exps[0].problem,
            [experiment.solver for experiment in exps],
            simulation_config,
            crn_options,
            proxy_values,
        )
        plans.append(plan)
        indices_by_group.append(indices)

    analysis_inputs = _run_plans(plans, n_jobs)
    results: list[tuple[int, AnalysisInput]] = []
    offset = 0
    for indices in indices_by_group:
        group_inputs = analysis_inputs[offset : offset + len(indices)]
        offset += len(indices)
        for idx, ai in zip(indices, group_inputs, strict=True):
            results.append((idx, ai))

    # Sort by original index and return
    results.sort(key=lambda x: x[0])
    return [ai for _, ai in results]
