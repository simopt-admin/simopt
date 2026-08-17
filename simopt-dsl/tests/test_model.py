"""Tests for model declarations and replication evaluation."""

from simopt_dsl import Model


def test_vector_component_does_not_alias_scalar_with_same_name() -> None:
    model = Model()
    vector = model.add_continuous_vector(0, 10, name="x", shape=1)
    scalar = model.add_continuous_variable(0, 10, name="x[0]")
    model.minimize(vector[0])

    assert vector[0] != scalar

    evaluation = model.run_replication((2, 7), [object()])

    assert evaluation.objective == 2
    assert evaluation.objective_gradient == (1, 0)
