import numpy as np

from tests.utils import split_pairs


def test_solvers_share_pooled_normalization(same_problem_experiments):
    experiments, inputs = split_pairs(same_problem_experiments)
    reference_experiment, reference = experiments[0], inputs[0]
    assert reference_experiment.problem.optimal_value is None
    assert reference_experiment.problem.optimal_solution is None

    candidates = [
        (objective * experiment.problem.minmax[0], solution)
        for experiment, data in zip(experiments, inputs, strict=True)
        for objective, solution in data.full_df.groupby(["mrep", "step"])
        .agg({"objective": "mean", "solution": "first"})
        .itertuples(index=False)
    ]
    _, expected_xstar = max(candidates, key=lambda candidate: candidate[0])
    assert tuple(reference.xstar) == expected_xstar

    initial_objective = np.mean(reference.x0_sample)
    optimal_objective = np.mean(reference.xstar_sample)
    initial_gap = initial_objective - optimal_objective
    assert initial_gap != 0

    for analysis_input in inputs:
        np.testing.assert_array_equal(analysis_input.x0, reference.x0)
        np.testing.assert_array_equal(analysis_input.xstar, expected_xstar)
        np.testing.assert_array_equal(analysis_input.x0_sample, reference.x0_sample)
        np.testing.assert_array_equal(analysis_input.xstar_sample, reference.xstar_sample)
        df = analysis_input.mean_df
        np.testing.assert_allclose(
            df["normalized_objective"],
            (df["objective"] - optimal_objective) / initial_gap,
        )
