"""Functions for normalizing objective curves."""

import dataclasses
import logging

import numpy as np
import pandas as pd
import pandera.pandas as pa

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import Solution
from simopt.curve import Curve
from simopt.problem import Problem
from simopt.utils import make_nonzero

from .post_replicate import _from_list
from .single import ProblemSolver


class MeanReplicateSchema(pa.DataFrameModel):
    """Schema for macroreplication means."""

    mrep: int
    step: int
    objective: float
    stochastic_constraints: object  # np.ndarray


def _check_experiment(
    ref_experiment: ProblemSolver, experiment: ProblemSolver, new_api: bool = False
) -> None:
    """Check if the experiment matches the reference experiment configuration.

    Args:
        ref_experiment: The reference experiment to compare against.
        experiment: The experiment to check.
        new_api: Whether using the new API.

    Raises:
        RuntimeError: If there is a mismatch in problem, macro-replications,
            or if the experiment has not been run or post-replicated.
    """
    # Check if problems are the same.
    if experiment.problem != ref_experiment.problem:
        raise RuntimeError("At least two experiments have different problems.")

    # Check if experiments have common number of macroreps.
    if experiment.n_macroreps != ref_experiment.n_macroreps:
        raise RuntimeError(
            "At least two experiments have different numbers of macro-replications."
        )

    # Check if experiments have common number of post-replications.
    if experiment.n_postreps != ref_experiment.n_postreps:
        raise RuntimeError(
            "At least two experiments have different numbers of post-replications. "
            "Estimation of optimal solution x* may be based on different numbers "
            "of post-replications."
        )

    # New API does not set has_run and has_postreplicated attributes.
    if new_api:
        return

    # Check if experiment has been run
    if not experiment.has_run:
        raise RuntimeError(
            f"The experiment of {experiment.solver.name} on "
            f"{experiment.problem.name} has not been run."
        )

    # Check if experiment has been post-replicated
    if not experiment.has_postreplicated:
        raise RuntimeError(
            f"The experiment of {experiment.solver.name} on "
            f"{experiment.problem.name} has not been post-replicated."
        )


def _set_up_rngs(problem: Problem) -> list[MRG32k3a]:
    """Set up RNGs.

    Args:
        problem: The problem instance.

    Returns:
        A list of initialized MRG32k3a random number generators.
    """
    # TODO: it seems that the RNGs are the shared with post replications step.
    return [
        MRG32k3a(s_ss_sss_index=[0, problem.model.n_rngs + i, 0])
        for i in range(problem.model.n_rngs)
    ]


def _simulate(
    problem: Problem, x: tuple, rngs: list[MRG32k3a], n_reps: int
) -> np.ndarray:
    """Simulate a solution x.

    Args:
        problem: The problem instance.
        x: The decision variable tuple.
        rngs: List of RNGs.
        n_reps: Number of replications to run.

    Returns:
        A list of objective values (first objective only).
    """
    solution = Solution(x, problem)
    solution.attach_rngs(rngs, copy=False)
    # TODO: the naming of `num_macroreps` is wrong. It should be number of replications.
    problem.simulate(solution, n_reps)
    return solution.objectives[:, 0]  # Assuming only one objective


def _get_x0_and_sample(
    problem: Problem,
    rngs: list[MRG32k3a],
    n_reps: int,
    proxy_initial_objective: float | None = None,
) -> tuple[tuple, np.ndarray]:
    """Determine initial solution x0 and its post-replication values.

    Args:
        problem: The problem instance.
        rngs: List of RNGs.
        n_reps: Number of replications.
        proxy_initial_objective: Optional known value for f(x).

    Returns:
        A tuple (x, sample).
    """
    logging.info(f"Normalizing problem {problem.name}.")

    x = problem.factors["initial_solution"]

    if proxy_initial_objective is not None:
        sample = np.full(n_reps, proxy_initial_objective)
    else:
        sample = _simulate(problem, x, rngs, n_reps)

    return x, sample


def _best_with_feasibility(
    problem: Problem, df: pd.DataFrame, rngs: list[MRG32k3a], n_reps: int
) -> tuple[tuple, np.ndarray]:
    """Determine the best solution among experiments, considering feasibility.

    Args:
        problem: The problem being solved.
        df: DataFrame containing the data.
        rngs: List of RNGs for simulation.
        n_reps: Number of post-replications.

    Returns:
        A tuple containing the best solution (xstar) and its post-replications.

    Raises:
        RuntimeError: If no feasible solutions are found.
    """
    df = (
        df.groupby(["experiment", "mrep", "step"])
        .agg(
            {"objective": "mean", "stochastic_constraints": "mean", "solution": "first"}
        )
        .reset_index()
    )
    df = MeanReplicateSchema.validate(df)

    has_stochastic_constraints = df["stochastic_constraints"].iloc[0].shape[0] > 0
    if has_stochastic_constraints:
        indices = df["stochastic_constraints"].apply(lambda x: np.all(x <= 0))
    else:
        indices = np.ones(len(df), dtype=bool)

    # Filter out infeasible solutions.
    df = df.loc[indices].copy()
    sense = problem.minmax[0]
    df.loc[:, "objective"] *= sense

    if len(df) == 0:
        raise RuntimeError(
            "No feasible solutions found for which to estimate proxy for x*."
        )

    best_index = df["objective"].idxmax()
    xstar = df.loc[best_index, "solution"]
    sample = _simulate(problem, xstar, rngs, n_reps)

    return xstar, sample


def _get_xstar_and_sample(
    problem: Problem,
    df: pd.DataFrame,
    rngs: list[MRG32k3a],
    n_reps: int,
    proxy_xstar: tuple | None = None,
    proxy_optimal_value: float | None = None,
) -> tuple[tuple | None, np.ndarray]:
    """Determine optimal solution x* (or proxy) and its post-replication values.

    Args:
        problem: The problem instance.
        df: DataFrame containing the data.
        rngs: List of RNGs.
        n_reps: Number of post-replications.
        proxy_optimal_value: Optional known value for f(x*).
        proxy_xstar: Optional known x*.

    Returns:
        A tuple (xstar, xstar_postreps). xstar can be None if only value is known.
    """
    # Determine (proxy for) optimal solution and/or (proxy for) its
    # objective function value. If deterministic (proxy for) f(x*),
    # create duplicate post-replicates to facilitate later bootstrapping.

    # TODO: investigate what happens 1) if proxy_optimal_value is provided but
    # proxy_xstar is not. 2) if optimal_value is provided but optimal_solution is not.

    # Proxy for f(x*) is specified
    if proxy_optimal_value is not None:
        logging.info("Finding f(x*) using provided proxy f(x*).")
        # TODO: should revisit this question. Do we always assume that both proxy_xstar
        # and proxy_optimal_value are provided?
        assert proxy_xstar is not None
        return proxy_xstar, np.full(n_reps, proxy_optimal_value)

    # Proxy for x* is specified.
    if proxy_xstar is not None:
        # TODO: verify the logic here. If the user provides a proxy for x*, we should
        # trust it is feasible.
        # # If stochastic constraints exist, ensure provided xstar is feasible.
        # if problem.n_stochastic_constraints >= 1 and any(
        #     opt_soln.stoch_constraints_mean > 0
        # ):
        #     x, objectives = _best_with_feasibility(
        #         experiments,
        #         problem,
        #         rngs,
        #         n_postreps_init_opt,
        #     )
        logging.info("Finding f(x*) using provided proxy x*.")
        return proxy_xstar, _simulate(problem, proxy_xstar, rngs, n_reps)

    # f(x*) is known
    if problem.optimal_value is not None:
        logging.info("Finding f(x*) using coded f(x*).")
        # NOTE: optimal_value is a tuple.
        # Currently hard-coded for single objective case, i.e., optimal_value[0].

        # TODO: should revisit this question. Do we always assume that both
        # problem.optimal_solution and problem.optimal_value are provided?
        assert problem.optimal_solution is not None
        return problem.optimal_solution, np.full(n_reps, problem.optimal_value)

    # x* is known...
    if problem.optimal_solution is not None:
        logging.info("Finding f(x*) using coded x*.")
        return problem.optimal_solution, _simulate(
            problem, problem.optimal_solution, rngs, n_reps
        )

    # If nothing is known, estimate x* empirically as estimated best solution
    # found by any solver on any macroreplication.
    logging.info("Finding f(x*) using best postreplicated solution as proxy for x*.")
    return _best_with_feasibility(problem, df, rngs, n_reps)


@dataclasses.dataclass
class NormalizationResult:
    """Class to store normalization results."""

    x0: tuple
    x0_sample: np.ndarray
    xstar: tuple
    xstar_sample: np.ndarray
    initial_objective: float
    optimal_objective: float
    initial_gap: float


def normalize(
    problem: Problem,
    df: pd.DataFrame,
    n_reps: int,
    crn_across_x0_xstar: bool = True,
    proxy_initial_objective: float | None = None,
    proxy_xstar: tuple | None = None,
    proxy_optimal_objective: float | None = None,
) -> NormalizationResult:
    """Computes normalization constants for a set of experiments.

    It assumes that all experiments have the same problem.
    """
    rngs = _set_up_rngs(problem)

    x0, x0_sample = _get_x0_and_sample(problem, rngs, n_reps, proxy_initial_objective)

    if crn_across_x0_xstar:
        for rng in rngs:
            rng.reset_substream()

    xstar, xstar_sample = _get_xstar_and_sample(
        problem, df, rngs, n_reps, proxy_xstar, proxy_optimal_objective
    )

    # Compute signed initial optimality gap = f(x0) - f(x*).
    initial_obj = float(np.mean(x0_sample))
    optimal_obj = float(np.mean(xstar_sample))
    initial_gap = float(initial_obj - optimal_obj)
    initial_gap = make_nonzero(initial_gap, "initial_gap")

    return NormalizationResult(
        x0,
        x0_sample,
        xstar,  # type: ignore
        xstar_sample,
        initial_obj,
        optimal_obj,
        initial_gap,
    )


def _get_curves(
    df: pd.DataFrame,
    optimal_objective: float,
    initial_gap: float,
    total_budget: float,
) -> tuple[list[Curve], list[Curve]]:
    objective_curves = []
    progress_curves = []

    for _, mrep in df.groupby("mrep"):
        budget = mrep["budget"].to_numpy()
        objective = mrep["objective"].to_numpy()
        objective_curves.append(Curve(budget.tolist(), objective.tolist()))

        normalized_budget = budget / total_budget
        normalized_objective = (objective - optimal_objective) / initial_gap
        progress_curves.append(
            Curve(normalized_budget.tolist(), normalized_objective.tolist())
        )

    return objective_curves, progress_curves


def _normalize_experiment(
    normalization_result: NormalizationResult,
    experiment: ProblemSolver,
    df: pd.DataFrame,
    index: int,
    n_reps: int,
    crn_across_x0_xstar: bool,
) -> None:
    """Normalize an objective curve using x0 and x*.

    Args:
        normalization_result: Normalization result containing x0, xstar,
            and related data.
        experiment: The experiment to normalize.
        df: The DataFrame containing the experiment data.
        index: The index of the experiment.
        n_reps: Number of post-replications.
        crn_across_x0_xstar: Whether CRN was used.
    """
    # DOUBLE-CHECK FOR SHALLOW COPY ISSUES.
    experiment.n_postreps_init_opt = n_reps
    experiment.crn_across_init_opt = crn_across_x0_xstar
    experiment.x0 = normalization_result.x0
    experiment.x0_postreps = normalization_result.x0_sample.tolist()
    experiment.xstar = normalization_result.xstar
    experiment.xstar_postreps = normalization_result.xstar_sample.tolist()

    df = df.loc[df["experiment"] == index]
    df.loc[df["solution"] == normalization_result.x0, "objective"] = (
        normalization_result.initial_objective
    )
    df.loc[df["solution"] == normalization_result.xstar, "objective"] = (
        normalization_result.optimal_objective
    )

    # Construct objective and progress curves.
    experiment.objective_curves, experiment.progress_curves = _get_curves(
        df,
        normalization_result.optimal_objective,
        normalization_result.initial_gap,
        experiment.problem.factors["budget"],
    )

    experiment.has_postnormalized = True


def _to_df(experiments: list[ProblemSolver]) -> pd.DataFrame:
    dfs = []
    for i, experiment in enumerate(experiments):
        solution = _from_list(experiment.all_recommended_xs, "solution")
        objective = _from_list(experiment.all_est_objectives, "objective")
        budget = _from_list(experiment.all_intermediate_budgets, "budget")
        stochastic_constraints = _from_list(
            experiment.all_est_lhs, "stochastic_constraints"
        )
        if stochastic_constraints.empty:
            stochastic_constraints = pd.DataFrame(
                {
                    "mrep": objective["mrep"],
                    "step": objective["step"],
                    "stochastic_constraints": [np.array([])] * len(objective),
                }
            )

        df = (
            solution.merge(objective, on=["mrep", "step"])
            .merge(budget, on=["mrep", "step"])
            .merge(stochastic_constraints, on=["mrep", "step"])
        )
        df["experiment"] = i
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).reset_index(drop=True)


def post_normalize(
    experiments: list[ProblemSolver],
    n_postreps_init_opt: int,
    crn_across_init_opt: bool = True,
    proxy_init_val: float | None = None,
    proxy_opt_x: tuple | None = None,
    proxy_opt_val: float | None = None,
    create_pair_pickles: bool = False,
) -> None:
    """Constructs objective and normalized progress curves for a set of experiments.

    Args:
        experiments (list[ProblemSolver]): Problem-solver pairs for different solvers on
            the same problem.
        n_postreps_init_opt (int): Number of postreplications at initial (x0) and
            optimal (x*) solutions.
        crn_across_init_opt (bool, optional): If True, use CRN for postreplications at
            x0 and x*. Defaults to True.
        proxy_init_val (float, optional): Known objective value of the initial solution.
        proxy_opt_x (tuple, optional): Proxy for the optimal solution.
        proxy_opt_val (float, optional): Proxy or bound for the optimal objective value.
        create_pair_pickles (bool, optional): If True, create a pickle file for each
            problem-solver pair. Defaults to False.
    """
    # Check that all experiments have the same problem and same
    # post-experimental setup.
    ref_experiment = experiments[0]
    for experiment in experiments:
        _check_experiment(ref_experiment, experiment)
    problem = ref_experiment.problem

    df = _to_df(experiments)

    normalization_result = normalize(
        problem,
        df,
        n_postreps_init_opt,
        crn_across_init_opt,
        proxy_init_val,
        proxy_opt_x,
        proxy_opt_val,
    )

    # Store x0 and x* info and compute progress curves for each ProblemSolver.
    for i, experiment in enumerate(experiments):
        _normalize_experiment(
            normalization_result,
            experiment,
            df,
            i,
            n_postreps_init_opt,
            crn_across_init_opt,
        )

        # Save ProblemSolver object to .pickle file if specified.
        if create_pair_pickles:
            file_name = experiment.file_name_path.name
            experiment.record_experiment_results(file_name=file_name)
