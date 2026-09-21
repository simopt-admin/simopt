"""Regression tests for configuration objects passed to model replications."""

import pytest

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import Solution
from simopt.data_farming_base import DataFarmingExperiment, DesignPoint
from simopt.models.contam import (
    ContaminationTotalCostCont,
    ContaminationTotalCostDisc,
)
from simopt.models.example import ExampleProblem
from simopt.models.mm1queue import MM1Queue


@pytest.mark.parametrize(
    "problem_class",
    [ExampleProblem, ContaminationTotalCostDisc, ContaminationTotalCostCont],
)
def test_simulation_passes_config_without_changing_defaults(problem_class, monkeypatch):
    problem = problem_class()
    original = problem.model.config.model_copy(deep=True)
    replicate = problem.model.replicate
    received = []

    def capture(factors, rngs):
        received.append(factors)
        return replicate(factors, rngs)

    monkeypatch.setattr(problem.model, "replicate", capture)
    rngs = [MRG32k3a(s_ss_sss_index=[0, i, 0]) for i in range(problem.model.n_rngs)]
    solution = Solution((0.0,) * problem.dim, rngs)
    problem.simulate(solution)

    assert len(received) == 1
    assert isinstance(received[0], problem.model.config_class)
    assert received[0] is not problem.model.config
    for key, value in problem.vector_to_factor_dict(solution.x).items():
        assert getattr(received[0], key) == value
    assert problem.model.config == original


def test_factor_export_reflects_config_and_preserves_aliases():
    model = MM1Queue({"lambda": 2.0})
    model.config.lambda_ = 4.0

    assert model.factors["lambda"] == 4.0
    assert "lambda_" not in model.factors


def test_design_point_passes_updated_config(monkeypatch):
    point = DesignPoint(MM1Queue())
    point.model_factors["lambda"] = 2.0
    point.attach_rngs([MRG32k3a() for _ in range(point.model.n_rngs)])
    replicate = point.model.replicate
    received = []

    def capture(factors, rngs):
        received.append(factors)
        return replicate(factors, rngs)

    monkeypatch.setattr(point.model, "replicate", capture)
    point.simulate(2)

    assert point.n_reps == 2
    assert len(received) == 2
    assert received[0] is received[1]
    assert received[0].lambda_ == 2.0


def test_data_farming_keeps_distinct_design_configs(tmp_path):
    design_path = tmp_path / "design.txt"
    design_path.write_text("lambda\n2.0\n4.0\n")
    experiment = DataFarmingExperiment("MM1", ["lambda"], design_path=design_path)

    assert [point.model.config.lambda_ for point in experiment.design] == [2.0, 4.0]
    assert [point.model_factors["lambda"] for point in experiment.design] == [2.0, 4.0]
