import pytest

from simopt.experiment.post_normalize import post_normalize
from simopt.experiment.single import ProblemSolver
from test.utils import load_problem_solver


@pytest.fixture(scope="session")
def experiment(request):
    """Fixture that loads an experiment based on the parametrized path."""
    # Default to CNTNEWS1_ADAM if no parameter is provided
    path = getattr(request, "param", "test/expected_results/CNTNEWS1_ADAM.pickle.zst")
    return load_problem_solver(path)


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
