import numpy as np

from simopt.compat import convert


def test_solvers_share_pooled_normalization(same_problem_experiments):
    experiments = same_problem_experiments
    reference = experiments[0]
    assert reference.problem.optimal_value is None
    assert reference.problem.optimal_solution is None

    candidates = [
        (objective * experiment.problem.minmax[0], solution)
        for experiment in experiments
        for objectives, solutions in zip(
            experiment.all_est_objectives, experiment.all_recommended_xs, strict=True
        )
        for objective, solution in zip(objectives, solutions, strict=True)
    ]
    _, expected_xstar = max(candidates, key=lambda candidate: candidate[0])
    assert reference.xstar == expected_xstar

    initial_objective = np.mean(reference.x0_postreps)
    optimal_objective = np.mean(reference.xstar_postreps)
    initial_gap = initial_objective - optimal_objective
    assert initial_gap != 0

    for experiment in experiments:
        assert experiment.x0 == reference.x0
        assert experiment.xstar == expected_xstar
        np.testing.assert_array_equal(experiment.x0_postreps, reference.x0_postreps)
        np.testing.assert_array_equal(experiment.xstar_postreps, reference.xstar_postreps)

        analysis_input = convert(experiment)
        np.testing.assert_array_equal(analysis_input.xstar, expected_xstar)
        np.testing.assert_array_equal(analysis_input.x0_sample, reference.x0_postreps)
        np.testing.assert_array_equal(analysis_input.xstar_sample, reference.xstar_postreps)
        df = analysis_input.mean_df
        np.testing.assert_allclose(
            df["normalized_objective"],
            (df["objective"] - optimal_objective) / initial_gap,
        )
