"""Experimental framework for running experiments."""

from simopt.experiment_base import ProblemSolver, post_normalize


class ProblemConfig:
    """Class to hold problem config information."""

    def __init__(
        self,
        name: str,
        rename: str | None = None,
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the Problem with name, rename, and problem/model fixed factors."""
        self.name = name
        self.rename = rename if rename else name
        self.fixed_factors = fixed_factors if fixed_factors else {}
        self.model_fixed_factors = model_fixed_factors if model_fixed_factors else {}


class SolverConfig:
    """Class to hold solver config information."""

    def __init__(
        self, name: str, rename: str | None = None, fixed_factors: dict | None = None
    ) -> None:
        """Initialize the Solver with name, rename, and solver fixed factors."""
        self.name = name
        self.rename = rename if rename else name
        self.fixed_factors = fixed_factors if fixed_factors else {}


def run_experiment(
    problems: list[dict],
    solvers: list[dict],
    num_macroreps: int,
    num_postreps: int,
    num_postnorms: int,
) -> list[list[ProblemSolver]]:
    """Run an experiment using the provided configurations.

    Args:
        problems: List of ProblemConfig instances.
        solvers: List of SolverConfig instances.
        num_macroreps: Number of macroreplications.
        num_postreps: Number of post-replications.
        num_postnorms: Number of post-normalizations.

    Returns:
        List[list[ProblemSolver]]: A list of lists containing ProblemSolver instances,
        grouped by problem.
    """
    problem_configs = [ProblemConfig(**p) for p in problems]
    solver_configs = [SolverConfig(**s) for s in solvers]

    all_experiments = []
    total = len(problem_configs)
    for problem_idx, problem in enumerate(problem_configs):
        print(
            f"Running Problem {problem_idx + 1}/{total}: {problem.rename}...",
            end="",
            flush=True,
        )
        # Keep track of experiments on the same problem for post-processing.
        experiments_same_problem = []
        # Create each ProblemSolver and run it.
        for solver in solver_configs:
            new_experiment = ProblemSolver(
                solver_name=solver.name,
                solver_rename=solver.rename,
                solver_fixed_factors=solver.fixed_factors,
                problem_name=problem.name,
                problem_rename=problem.rename,
                problem_fixed_factors=problem.fixed_factors,
                model_fixed_factors=problem.model_fixed_factors,
            )
            # Run and post-replicate the experiment.
            new_experiment.run(n_macroreps=num_macroreps)
            new_experiment.post_replicate(n_postreps=num_postreps)
            experiments_same_problem.append(new_experiment)

        # Post-normalize experiments with L.
        # Provide NO proxies for f(x0), f(x*), or f(x).
        post_normalize(
            experiments=experiments_same_problem,
            n_postreps_init_opt=num_postnorms,
        )
        all_experiments.append(experiments_same_problem)
        print("Done.")
    print("All experiments completed.")
    return all_experiments


def group_experiments_by_solver(
    all_experiments: list[list[ProblemSolver]],
) -> list[list[ProblemSolver]]:
    """Group experiments by solver name."""
    experiment_dict = {}
    for exp_problem_list in all_experiments:
        for experiment in exp_problem_list:
            # Use the solver name as the key and append the ProblemSolver instance.
            key = experiment.solver.name
            if key not in experiment_dict:
                experiment_dict[key] = []
            experiment_dict[key].append(experiment)
    # Turn the dictionary into a list of lists.
    return list(experiment_dict.values())
