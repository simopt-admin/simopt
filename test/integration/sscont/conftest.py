from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import zstandard as zstd

from simopt.directory import problem_directory, solver_directory
from simopt.experiment.api import AnalysisInput, SimulationConfig, run
from simopt.experiment.single import ProblemSolver

NUM_MACROREPS = 10
NUM_POSTREPS = 100
NUM_POSTNORMS = 200
FAST_EXPERIMENT_COUNT = 4

DEMAND_MEANS = [25.0, 50.0, 100.0, 200.0, 400.0]
LEAD_MEANS = [1.0, 3.0, 6.0, 9.0]

SOLVERS = [
    ("RNDSRCH", {"sample_size": 10}, "RNDSRCH_ss=10"),
    ("RNDSRCH", {"sample_size": 50}, "RNDSRCH_ss=50"),
    ("ASTRODF", None, None),
    ("NELDMD", None, None),
    ("STRONG", None, None),
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-full",
        action="store_true",
        default=False,
        help="Run both full and reduced SSCont integration datasets.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "dataset" not in metafunc.fixturenames:
        return
    run_full = metafunc.config.getoption("--run-full")
    modes = ["full", "fast"] if run_full else ["fast"]
    metafunc.parametrize("dataset", modes, scope="session")


@pytest.fixture(scope="session")
def sscont_artifacts_dir() -> Path:
    path = Path(__file__).parent / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def dataset(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def sscont_plot_dir(dataset: str, sscont_artifacts_dir: Path) -> Path:
    dirname = "plots_fast" if dataset == "fast" else "plots"
    path = sscont_artifacts_dir / dirname
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def sscont_simulation_config() -> SimulationConfig:
    return SimulationConfig(
        n_mreps=NUM_MACROREPS,
        n_preps=NUM_POSTREPS,
        n_preps_x0_xstar=NUM_POSTNORMS,
    )


@pytest.fixture
def sscont_experiments(dataset: str) -> list[ProblemSolver]:
    solvers = []
    for solver_name, fixed_factors, name in SOLVERS:
        solver_class = solver_directory[solver_name]
        solver_name = name if name is not None else solver_name
        solver = solver_class(solver_name, fixed_factors)
        solvers.append(solver)

    problems = [
        problem_directory["SSCONT-1"](
            fixed_factors={"budget": 1000},
            model_fixed_factors={"demand_mean": demand_mean, "lead_mean": lead_mean},
        )
        for demand_mean in DEMAND_MEANS
        for lead_mean in LEAD_MEANS
    ]

    experiments = [
        ProblemSolver(solver=solver, problem=problem)
        for problem in problems
        for solver in solvers
    ]
    if dataset == "fast":
        return experiments[:FAST_EXPERIMENT_COUNT]
    return experiments


@pytest.fixture
def sscont_results(
    dataset: str,
    sscont_artifacts_dir: Path,
    sscont_experiments: list[ProblemSolver],
    sscont_simulation_config: SimulationConfig,
) -> list[AnalysisInput]:
    cache_filename = "run_fast.pkl.zst" if dataset == "fast" else "run.pkl.zst"
    cache_path = sscont_artifacts_dir / cache_filename
    if cache_path.exists():
        with zstd.open(cache_path, "rb") as f:
            return pickle.load(f)

    results = run(sscont_experiments, sscont_simulation_config)
    with zstd.open(cache_path, "wb") as f:
        pickle.dump(results, f)
    return results
