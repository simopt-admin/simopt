from copy import deepcopy

import matplotlib.pyplot as plt
import pytest

from simopt.experiment.post_normalize import post_normalize
from simopt.experiment.single import ProblemSolver
from tests.analysis.options_helpers import option_experiment, option_pair  # noqa: F401
from tests.utils import EXPECTED_RESULTS_DIR, load_problem_solver


@pytest.fixture(autouse=True)
def close_plot_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def experiment(request):
    """Fixture that loads an experiment based on the parametrized path."""
    # Default to CNTNEWS1_ADAM if no parameter is provided
    filename = getattr(request, "param", "CNTNEWS1_ADAM.pickle.zst")
    return load_problem_solver(EXPECTED_RESULTS_DIR / filename)


@pytest.fixture(scope="session")
def same_problem_experiments():
    """Fixture that manually runs two experiments on CNTNEWS-1 with RNDSRCH and ADAM."""
    experiment1 = ProblemSolver(solver_name="RNDSRCH", problem_name="CNTNEWS-1")
    experiment2 = ProblemSolver(solver_name="ADAM", problem_name="CNTNEWS-1")

    n_macroreps = 2
    experiment1.run(n_macroreps)
    experiment2.run(n_macroreps)

    n_postreps = 10
    experiment1.post_replicate(n_postreps)
    experiment2.post_replicate(n_postreps)

    n_postreps_init_opt = 10
    post_normalize([experiment1, experiment2], n_postreps_init_opt)

    return [experiment1, experiment2]


@pytest.fixture(scope="session")
def san2_experiments():
    """Fixture that runs SAN-2 experiments for FCSA and RNDSRCH."""
    experiment1 = ProblemSolver(solver_name="FCSA", problem_name="SAN-2")
    experiment2 = ProblemSolver(solver_name="RNDSRCH", problem_name="SAN-2")

    n_macroreps = 2
    experiment1.run(n_macroreps)
    experiment2.run(n_macroreps)

    n_postreps = 10
    experiment1.post_replicate(n_postreps)
    experiment2.post_replicate(n_postreps)

    n_postreps_init_opt = 10
    post_normalize([experiment1, experiment2], n_postreps_init_opt)

    return [experiment1, experiment2]


@pytest.fixture(scope="session")
def different_problem_experiments():
    """Fixture that runs experiments on different problems.

    Uses CNTNEWS-1 with RNDSRCH and CONTAM-1 with RNDSRCH.
    """
    experiment1 = ProblemSolver(solver_name="RNDSRCH", problem_name="CNTNEWS-1")
    experiment2 = ProblemSolver(solver_name="RNDSRCH", problem_name="CONTAM-1")

    n_macroreps = 2
    experiment1.run(n_macroreps)
    experiment2.run(n_macroreps)

    n_postreps = 10
    experiment1.post_replicate(n_postreps)
    experiment2.post_replicate(n_postreps)

    # Post-normalize each experiment separately since they have different problems
    n_postreps_init_opt = 10
    post_normalize([experiment1], n_postreps_init_opt)
    post_normalize([experiment2], n_postreps_init_opt)

    return [experiment1, experiment2]


@pytest.fixture(scope="session")
def different_problem_experiments_stochastic_constraints():
    """Fixture that runs experiments on different problems with stochastic constraints.

    Uses CONTAM-2 with RNDSRCH and SAN-2 with RNDSRCH.
    """
    experiment1 = ProblemSolver(solver_name="RNDSRCH", problem_name="CONTAM-2")
    experiment2 = ProblemSolver(solver_name="RNDSRCH", problem_name="SAN-2")

    n_macroreps = 2
    experiment1.run(n_macroreps)
    experiment2.run(n_macroreps)

    n_postreps = 10
    experiment1.post_replicate(n_postreps)
    experiment2.post_replicate(n_postreps)

    # Post-normalize each experiment separately since they have different problems
    n_postreps_init_opt = 10
    post_normalize([experiment1], n_postreps_init_opt)
    post_normalize([experiment2], n_postreps_init_opt)

    return [experiment1, experiment2]


@pytest.fixture(scope="session")
def plot_experiment_grid(same_problem_experiments):
    experiments = [
        ProblemSolver(solver_name=experiment.solver.name, problem_name="SAN-1")
        for experiment in same_problem_experiments
    ]
    for experiment in experiments:
        experiment.run(2)
        experiment.post_replicate(10)
    post_normalize(experiments, 10)
    return [list(pair) for pair in zip(same_problem_experiments, experiments, strict=True)]


@pytest.fixture(scope="session")
def stochastic_plot_experiment_grid(san2_experiments):
    experiments = [
        ProblemSolver(solver_name=experiment.solver.name, problem_name="CONTAM-2")
        for experiment in san2_experiments
    ]
    for experiment in experiments:
        experiment.run(2)
        experiment.post_replicate(10)
    post_normalize(experiments, 10)
    return [list(pair) for pair in zip(san2_experiments, experiments, strict=True)]


@pytest.fixture(
    scope="session", params=[(1, 1), (1, 2), (2, 1), (2, 2)], ids=["1p1s", "1p2s", "2p1s", "2p2s"]
)
def plot_experiments(request, plot_experiment_grid):
    n_problems, n_solvers = request.param
    experiments = deepcopy([row[:n_problems] for row in plot_experiment_grid[:n_solvers]])
    # Normalize each problem using only the solvers included in this case.
    for group in zip(*experiments, strict=True):
        post_normalize(list(group), 10)
    return experiments


@pytest.fixture(
    scope="session", params=[(1, 1), (1, 2), (2, 1), (2, 2)], ids=["1p1s", "1p2s", "2p1s", "2p2s"]
)
def stochastic_plot_experiments(request, stochastic_plot_experiment_grid):
    n_problems, n_solvers = request.param
    experiments = deepcopy(
        [row[:n_problems] for row in stochastic_plot_experiment_grid[:n_solvers]]
    )
    # Normalize each problem using only the solvers included in this case.
    for group in zip(*experiments, strict=True):
        post_normalize(list(group), 10)
    return experiments
