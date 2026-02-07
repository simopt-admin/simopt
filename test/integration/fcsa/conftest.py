from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import zstandard as zstd

from simopt.experiment.api import AnalysisInput, SimulationConfig, run
from simopt.experiment.single import ProblemSolver
from simopt.models.san import SANLongestPathStochastic
from simopt.solvers.fcsa import FCSA

NUM_MACROREPS = 10
NUM_POSTREPS = 100

PROBLEM_FIXED_FACTORS = {
    "constraint_nodes": [6, 8],
    "length_to_node_constraint": [5, 5],
    "initial_solution": (5,) * 13,
    "budget": 10000,
}


@pytest.fixture(scope="session")
def fcsa_artifacts_dir() -> Path:
    path = Path(__file__).parent / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def fcsa_plot_dir(fcsa_artifacts_dir: Path) -> Path:
    path = fcsa_artifacts_dir / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def fcsa_simulation_config() -> SimulationConfig:
    return SimulationConfig(
        n_mreps=NUM_MACROREPS,
        n_preps=NUM_POSTREPS,
        n_preps_x0_xstar=NUM_POSTREPS,
    )


@pytest.fixture(scope="session")
def fcsa_problem() -> SANLongestPathStochastic:
    return SANLongestPathStochastic(fixed_factors=PROBLEM_FIXED_FACTORS)


@pytest.fixture(scope="session")
def fcsa_all_solutions_experiments(
    fcsa_problem: SANLongestPathStochastic,
) -> list[ProblemSolver]:
    solvers = [
        FCSA(
            fixed_factors={
                "search_direction": "CSA",
                "normalize_grads": False,
                "report_all_solns": True,
                "crn_across_solns": False,
            },
            name="CSA",
        ),
        FCSA(
            fixed_factors={
                "search_direction": "CSA",
                "normalize_grads": True,
                "report_all_solns": True,
                "crn_across_solns": False,
            },
            name="CSA-N",
        ),
        FCSA(
            fixed_factors={
                "search_direction": "FCSA",
                "normalize_grads": True,
                "report_all_solns": True,
                "crn_across_solns": False,
            },
            name="FCSA",
        ),
    ]
    return [ProblemSolver(solver=solver, problem=fcsa_problem) for solver in solvers]


@pytest.fixture(scope="session")
def fcsa_all_solutions_results(
    fcsa_artifacts_dir: Path,
    fcsa_all_solutions_experiments: list[ProblemSolver],
    fcsa_simulation_config: SimulationConfig,
) -> list[AnalysisInput]:
    cache_path = fcsa_artifacts_dir / "run_all_solutions.pkl.zst"
    if cache_path.exists():
        with zstd.open(cache_path, "rb") as f:
            return pickle.load(f)

    results = run(fcsa_all_solutions_experiments, fcsa_simulation_config)
    with zstd.open(cache_path, "wb") as f:
        pickle.dump(results, f)
    return results


@pytest.fixture(scope="session")
def fcsa_recommended_experiments(
    fcsa_problem: SANLongestPathStochastic,
) -> list[ProblemSolver]:
    solvers = [
        FCSA(
            fixed_factors={
                "search_direction": "CSA",
                "normalize_grads": True,
                "report_all_solns": False,
                "crn_across_solns": False,
            },
            name="CSA-N",
        ),
        FCSA(
            fixed_factors={
                "search_direction": "FCSA",
                "normalize_grads": True,
                "report_all_solns": False,
                "crn_across_solns": False,
            },
            name="FCSA",
        ),
    ]
    return [ProblemSolver(solver=solver, problem=fcsa_problem) for solver in solvers]


@pytest.fixture(scope="session")
def fcsa_recommended_results(
    fcsa_artifacts_dir: Path,
    fcsa_recommended_experiments: list[ProblemSolver],
    fcsa_simulation_config: SimulationConfig,
) -> list[AnalysisInput]:
    cache_path = fcsa_artifacts_dir / "run_recommended.pkl.zst"
    if cache_path.exists():
        with zstd.open(cache_path, "rb") as f:
            return pickle.load(f)

    results = run(fcsa_recommended_experiments, fcsa_simulation_config)
    with zstd.open(cache_path, "wb") as f:
        pickle.dump(results, f)
    return results
