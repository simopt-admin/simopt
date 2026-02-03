# ruff: noqa: D101, D103
"""API for running simulation optimization experiments."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel

from simopt.directory import problem_directory, solver_directory
from simopt.experiment.data import (
    ManyPostReplicateSchema,
    ManySolverHistorySchema,
)
from simopt.experiment.post_normalize import normalize
from simopt.experiment.post_replicate import post_replicate
from simopt.experiment.run_solver import run_solver
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


def create_matrix(
    solvers: list[Solver], problems: list[Problem]
) -> list[ProblemSolver]:
    return [
        ProblemSolver(solver=solver, problem=problem)
        for solver in solvers
        for problem in problems
    ]


def _mean(
    full_df: pd.DataFrame,
    x0: np.ndarray,
    x0_sample: np.ndarray,
    xstar: np.ndarray,
    xstar_sample: np.ndarray,
) -> tuple[float, pd.DataFrame]:
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
    df_mean["normalized_objective"] = (
        df_mean["objective"] - optimal_objective
    ) / initial_gap

    return budget, df_mean


class AnalysisInput:
    def __init__(
        self,
        full_df: pd.DataFrame,
        x0: np.ndarray,
        x0_sample: np.ndarray,
        xstar: np.ndarray,
        xstar_sample: np.ndarray,
    ) -> None:
        """Initialize AnalysisInput with data and compute mean statistics."""
        self.full_df = full_df
        self.x0 = x0
        self.x0_sample = x0_sample
        self.xstar = xstar
        self.xstar_sample = xstar_sample
        self.budget, self.mean_df = _mean(full_df, x0, x0_sample, xstar, xstar_sample)


def run_experiment(
    experiments: list[ProblemSolver],
    simulation_config: SimulationConfig,
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS,
    proxy_values: ProxyValues | None = None,
    n_jobs: int = -1,
) -> list[AnalysisInput]:
    """Run experiments and return analysis inputs.

    Args:
        experiments: List of ProblemSolver experiments to run. All must share
            the same problem.
        simulation_config: Configuration for the simulation (n_mreps, n_preps, etc.).
        crn_options: Options for common random numbers.
        proxy_values: Optional proxy values for normalization.
        n_jobs: Number of parallel jobs to run.

    Returns:
        A list of AnalysisInput objects, one per experiment.
    """
    if not experiments:
        raise ValueError("experiments must not be empty.")

    problem = experiments[0].problem
    for experiment in experiments[1:]:
        if experiment.problem != problem:
            raise ValueError("all experiments must share the same problem.")

    solver_history_dfs = []
    post_replicate_dfs = []
    for i, experiment in enumerate(experiments):
        solver_history_df, _ = run_solver(
            experiment.solver, experiment.problem, simulation_config.n_mreps, n_jobs
        )
        solver_history_df["experiment"] = i
        solver_history_dfs.append(solver_history_df)

        post_replicate_df, _ = post_replicate(
            experiment.problem,
            solver_history_df,
            simulation_config.n_preps,
            crn_options.across_macroreps,
            crn_options.across_budget,
        )
        post_replicate_df["experiment"] = i
        post_replicate_dfs.append(post_replicate_df)

    many_solver_history_df = pd.concat(solver_history_dfs, ignore_index=True)
    many_solver_history_df = ManySolverHistorySchema.validate(many_solver_history_df)
    many_post_replicate_df = pd.concat(post_replicate_dfs, ignore_index=True)
    many_post_replicate_df = ManyPostReplicateSchema.validate(many_post_replicate_df)

    full_df = many_solver_history_df.merge(
        many_post_replicate_df, on=["experiment", "mrep", "step"]
    )
    full_df = full_df.set_index(["experiment", "mrep", "step", "rep"])

    proxy_values = proxy_values or DEFAULT_PROXY_VALUES
    normalization_result = normalize(
        problem,
        full_df,
        simulation_config.n_preps_x0_xstar,
        crn_options.across_x0_xstar,
        proxy_values.initial_objective,
        proxy_values.xstar,
        proxy_values.optimal_objective,
    )

    # Build list of AnalysisInput objects
    analysis_inputs = []
    for i in range(len(experiments)):
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


def run(
    experiments: list[ProblemSolver],
    simulation_config: SimulationConfig,
    crn_options: CrnOptions = DEFAULT_CRN_OPTIONS,
    proxy_values: ProxyValues | None = None,
    n_jobs: int = -1,
) -> list[AnalysisInput]:
    """Run experiments and return analysis inputs.

    This is a wrapper around run_experiment that allows experiments on different
    problems. Experiments are grouped by problem and run_experiment is called
    for each group.

    Args:
        experiments: List of ProblemSolver experiments to run. Can be on
            different problems.
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
    results: list[tuple[int, AnalysisInput]] = []
    for group in problem_groups.values():
        indices, exps = zip(*group, strict=True)
        analysis_inputs = run_experiment(
            list(exps), simulation_config, crn_options, proxy_values, n_jobs
        )
        for idx, ai in zip(indices, analysis_inputs, strict=True):
            results.append((idx, ai))

    # Sort by original index and return
    results.sort(key=lambda x: x[0])
    return [ai for _, ai in results]
