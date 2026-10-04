"""Immutable solver-problem pairs."""

from __future__ import annotations

from dataclasses import dataclass

import simopt.directory as directory
from simopt.base import ConstraintType, ObjectiveType, Problem, Solver, VariableType


@dataclass(frozen=True, eq=False)
class Experiment:
    """One solver paired with one problem.

    Experiments compare by identity: `simopt.experiment.api.run` groups experiments
    that share the same problem instance and normalizes them together.
    """

    solver: Solver
    problem: Problem


def from_names(
    solver_name: str,
    problem_name: str,
    *,
    solver_fixed_factors: dict | None = None,
    problem_fixed_factors: dict | None = None,
    model_fixed_factors: dict | None = None,
) -> Experiment:
    """Create an experiment by looking up solver and problem names in `directory.py`.

    Args:
        solver_name (str): Name of the solver.
        problem_name (str): Name of the problem.
        solver_fixed_factors (dict, optional): Fixed solver parameters.
        problem_fixed_factors (dict, optional): Fixed problem parameters.
        model_fixed_factors (dict, optional): Fixed model parameters.

    Returns:
        Experiment: The solver-problem pair.
    """
    # Default arguments
    if solver_fixed_factors is None:
        solver_fixed_factors = {}
    if problem_fixed_factors is None:
        problem_fixed_factors = {}
    if model_fixed_factors is None:
        model_fixed_factors = {}

    if solver_name not in directory.solver_directory:
        error_msg = "Solver name not found in solver directory."
        raise ValueError(error_msg)
    if problem_name not in directory.problem_directory:
        error_msg = "Problem name not found in problem directory."
        raise ValueError(error_msg)
    solver = directory.solver_directory[solver_name](fixed_factors=solver_fixed_factors)
    problem = directory.problem_directory[problem_name](
        fixed_factors=problem_fixed_factors,
        model_fixed_factors=model_fixed_factors,
    )
    return Experiment(solver=solver, problem=problem)


# TODO: Convert this to throwing exceptions?
# TODO: Convert this functionality to run automatically
def check_compatibility(experiment: Experiment) -> str:
    """Check whether the experiment's solver and problem are compatible.

    Args:
        experiment (Experiment): The solver-problem pair to check.

    Returns:
        str: Error message in the event problem and solver are incompatible.
    """
    solver = experiment.solver
    problem = experiment.problem
    # make a string builder
    error_messages = []
    # Check number of objectives.
    if solver.objective_type == ObjectiveType.SINGLE and problem.n_objectives > 1:
        error_message = "Solver cannot solve a multi-objective problem"
        error_messages.append(error_message)
    elif solver.objective_type == ObjectiveType.MULTI and problem.n_objectives == 1:
        error_message = "Solver cannot solve a single-objective problem"
        error_messages.append(error_message)
    # Check constraint types.
    if solver.constraint_type.value < problem.constraint_type.value:
        solver_str = solver.constraint_type.name.lower()
        problem_str = problem.constraint_type.name.lower()
        error_message = (
            f"Solver can handle {solver_str} constraints, "
            f"but problem has {problem_str} constraints."
        )
        error_messages.append(error_message)

    if (
        # TODO: this is a hack to get around the fact that compatibility checks are
        # not implemented correctly.
        solver.class_name_abbr == "FCSA"
        and solver.constraint_type == ConstraintType.STOCHASTIC
        and problem.constraint_type != ConstraintType.STOCHASTIC
    ):
        error_message = (
            "Solver is designed for problems with stochastic constraints, but "
            "the problem does not have any."
        )
        error_messages.append(error_message)

    if (
        solver.constraint_type != ConstraintType.STOCHASTIC
        and problem.constraint_type == ConstraintType.STOCHASTIC
    ):
        error_message = (
            "Solver is not designed for problems with stochastic constraints, but "
            "the problem have some."
        )
        error_messages.append(error_message)

    # Check variable types.
    if (
        solver.variable_type == VariableType.DISCRETE
        and problem.variable_type != VariableType.DISCRETE
    ):
        problem_type = problem.variable_type.name.lower()
        error_message = (
            f"Solver is for discrete variables, but problem variables are {problem_type}."
        )
        error_messages.append(error_message)
    elif (
        solver.variable_type == VariableType.CONTINUOUS
        and problem.variable_type != VariableType.CONTINUOUS
    ):
        problem_type = problem.variable_type.name.lower()
        error_message = (
            f"Solver is for continuous variables, but problem variables are {problem_type}."
        )
        error_messages.append(error_message)
    # Check for existence of gradient estimates.
    if solver.gradient_needed and not problem.gradient_available:
        error_message = "Solver requires gradient estimates but problem does not have them."
        error_messages.append(error_message)
    # Strip trailing newline character.
    return "\n".join(error_messages)
