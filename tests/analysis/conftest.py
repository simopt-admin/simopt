import matplotlib.pyplot as plt
import pytest

from simopt.experiment import api
from simopt.experiment.api import SimulationConfig
from simopt.experiment.experiment import Experiment, from_names
from simopt.options import CrnOptions
from tests.analysis.options_helpers import option_experiment, option_pair  # noqa: F401
from tests.utils import EXPECTED_RESULTS_DIR, load_experiment_pair

SIMULATION_CONFIG = SimulationConfig(n_mreps=2, n_preps=10, n_preps_x0_xstar=10)


@pytest.fixture(autouse=True)
def close_plot_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def run_crn_options():
    """CRN options passed to api.run by the run fixtures; analysis reuses them."""
    return CrnOptions(across_budget=True, across_macroreps=False, across_x0_xstar=True)


def run_pairs(experiments, crn_options):
    """Run experiments through api.run and pair each with its AnalysisInput."""
    inputs = api.run(experiments, SIMULATION_CONFIG, crn_options, n_jobs=1)
    return list(zip(experiments, inputs, strict=True))


def shared_problem_experiments(problem_name, solver_names):
    """One experiment per solver, sharing a problem instance so api.run normalizes them together."""
    problem = from_names(solver_names[0], problem_name).problem
    return [
        Experiment(solver=from_names(name, problem_name).solver, problem=problem)
        for name in solver_names
    ]


@pytest.fixture(scope="session")
def experiment_pair(request):
    """Fixture that loads an (Experiment, AnalysisInput) pair based on the parametrized path."""
    # Default to CNTNEWS1_ADAM if no parameter is provided
    filename = getattr(request, "param", "CNTNEWS1_ADAM.pickle.zst")
    return load_experiment_pair(EXPECTED_RESULTS_DIR / filename)


@pytest.fixture(scope="session")
def same_problem_experiments(run_crn_options):
    """Fixture that runs two experiments on CNTNEWS-1 with RNDSRCH and ADAM."""
    experiments = shared_problem_experiments("CNTNEWS-1", ["RNDSRCH", "ADAM"])
    return run_pairs(experiments, run_crn_options)


@pytest.fixture(scope="session")
def san2_experiments(run_crn_options):
    """Fixture that runs SAN-2 experiments for FCSA and RNDSRCH."""
    experiments = shared_problem_experiments("SAN-2", ["FCSA", "RNDSRCH"])
    return run_pairs(experiments, run_crn_options)


@pytest.fixture(scope="session")
def different_problem_experiments(run_crn_options):
    """Fixture that runs experiments on different problems.

    Uses CNTNEWS-1 with RNDSRCH and CONTAM-1 with RNDSRCH.
    """
    experiments = [from_names("RNDSRCH", "CNTNEWS-1"), from_names("RNDSRCH", "CONTAM-1")]
    # The problems are distinct instances, so api.run normalizes each separately.
    return run_pairs(experiments, run_crn_options)


@pytest.fixture(scope="session")
def different_problem_experiments_stochastic_constraints(run_crn_options):
    """Fixture that runs experiments on different problems with stochastic constraints.

    Uses CONTAM-2 with RNDSRCH and SAN-2 with RNDSRCH.
    """
    experiments = [from_names("RNDSRCH", "CONTAM-2"), from_names("RNDSRCH", "SAN-2")]
    # The problems are distinct instances, so api.run normalizes each separately.
    return run_pairs(experiments, run_crn_options)


def solver_problem_grid(first_column, problem_name, run_crn_options):
    """Add a second problem column, shared across the solvers of `first_column`."""
    solver_names = [experiment.solver.name for experiment, _ in first_column]
    experiments = shared_problem_experiments(problem_name, solver_names)
    return [
        [first, second]
        for first, second in zip(first_column, run_pairs(experiments, run_crn_options), strict=True)
    ]


def sub_grid(grid, n_problems, n_solvers, crn_options):
    """Take a sub-grid and normalize each problem using only the solvers included."""
    # Deliberately re-runs api.run on the subset to get per-subset normalization; this relies
    # on solver runs being deterministic across re-runs.
    columns = []
    for problem_index in range(n_problems):
        experiments = [row[problem_index][0] for row in grid[:n_solvers]]
        columns.append(run_pairs(experiments, crn_options))
    return [list(row) for row in zip(*columns, strict=True)]


@pytest.fixture(scope="session")
def plot_experiment_grid(same_problem_experiments, run_crn_options):
    return solver_problem_grid(same_problem_experiments, "SAN-1", run_crn_options)


@pytest.fixture(scope="session")
def stochastic_plot_experiment_grid(san2_experiments, run_crn_options):
    return solver_problem_grid(san2_experiments, "CONTAM-2", run_crn_options)


@pytest.fixture(
    scope="session", params=[(1, 1), (1, 2), (2, 1), (2, 2)], ids=["1p1s", "1p2s", "2p1s", "2p2s"]
)
def plot_experiments(request, plot_experiment_grid, run_crn_options):
    n_problems, n_solvers = request.param
    return sub_grid(plot_experiment_grid, n_problems, n_solvers, run_crn_options)


@pytest.fixture(
    scope="session", params=[(1, 1), (1, 2), (2, 1), (2, 2)], ids=["1p1s", "1p2s", "2p1s", "2p2s"]
)
def stochastic_plot_experiments(request, stochastic_plot_experiment_grid, run_crn_options):
    n_problems, n_solvers = request.param
    return sub_grid(stochastic_plot_experiment_grid, n_problems, n_solvers, run_crn_options)


@pytest.fixture
def plot_baseline(request):
    from tests.analysis.baselines import load_plot_baseline

    return load_plot_baseline(request.node.nodeid)
