"""Tests for the DASSO solver."""

import math
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import norm

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.experiment.api import SimulationConfig, run
from simopt.experiment.experiment import from_names
from simopt_solvers import Budget, BudgetExhaustedError, Context
from simopt_solvers.solvers.dasso import DASSO, _RunState

CONFIG = SimulationConfig(n_mreps=1, n_preps=1, n_preps_x0_xstar=1)


def test_dasso_repeats_same_history_for_same_problem() -> None:
    """Verify DASSO produces repeatable histories for the same problem."""
    (result1,) = run([from_names("DASSO", "EXAMPLE-2")], CONFIG, n_jobs=1)
    (result2,) = run([from_names("DASSO", "EXAMPLE-2")], CONFIG, n_jobs=1)

    for column in ("solution", "budget"):
        assert result1.full_df[column].tolist() == result2.full_df[column].tolist()


@pytest.mark.parametrize("coordinate_ids", [(0,), (0, 1)])
def test_theta_design_covariance_matches_full_precision_inverse(coordinate_ids):
    solver = DASSO()
    mapping = DASSO._Mapping([(0,), (1,), (0, 1)], (2, 5), (5, 8))
    solution_indices = [0, 3, 12, 15]
    theta_design = solver._prepare_theta_design(mapping, coordinate_ids, solution_indices)
    precision_matrix_parameters = [2.0, -0.3, -0.2][: len(coordinate_ids) + 1]

    # Build the full lattice precision matrix independently of the sparse partition.
    shape = (4,) * len(coordinate_ids)
    precision = 2.0 * np.eye(math.prod(shape))
    for coordinate in np.ndindex(shape):
        row = np.ravel_multi_index(coordinate, shape)
        for dimension_id in range(len(coordinate_ids)):
            neighbor = list(coordinate)
            neighbor[dimension_id] += 1
            if neighbor[dimension_id] < shape[dimension_id]:
                column = np.ravel_multi_index(tuple(neighbor), shape)
                precision[row, column] = precision[column, row] = precision_matrix_parameters[
                    dimension_id + 1
                ]
    full_covariance = np.linalg.inv(precision)
    low_dim_comp_indices = [
        mapping.get_low_dim_comp_ind_from_sol_index(sol_ind, coordinate_ids)
        for sol_ind in solution_indices
    ]
    cov_mat_dd = solver._get_covariance_matrix_dd(theta_design, precision_matrix_parameters)
    transformed_cov_mat_dd = (
        theta_design.rearranged_tf_mat_d @ cov_mat_dd @ theta_design.rearranged_tf_mat_d.T
    )

    assert theta_design.num_d == len(set(low_dim_comp_indices))
    assert theta_design.num_u + theta_design.num_d == len(precision)
    np.testing.assert_allclose(
        transformed_cov_mat_dd,
        full_covariance[np.ix_(low_dim_comp_indices, low_dim_comp_indices)],
    )


def test_theta_candidates_are_feasible_and_repeatable():
    solver = DASSO()
    ctx1 = Mock(spec=Context, rng=Mock(return_value=MRG32k3a()))
    ctx2 = Mock(spec=Context, rng=Mock(return_value=MRG32k3a()))

    candidates1 = solver._generate_theta_candidates(ctx1, (0, 1, 2))
    candidates2 = solver._generate_theta_candidates(ctx2, (0, 1, 2))

    assert candidates1.shape == (30, 3)
    assert np.all(candidates1 >= 0)
    assert np.all(candidates1.sum(axis=1) < 0.5)
    np.testing.assert_array_equal(candidates1, candidates2)
    assert ctx1.rng(2).random() == ctx2.rng(2).random()


def test_theta_negative_log_likelihood_matches_diagonal_gaussian():
    actual = DASSO._negative_log_likelihood_function(
        2.0, np.diag([2.0, 6.0]), np.diag([0.5, 0.25]), np.array([1.0, -2.0])
    )

    assert actual == pytest.approx(math.log(1.5 * 3.25) + 1 / 1.5 + 4 / 3.25)


def test_theta_fit_selects_the_best_candidate_and_scales_all_parameters():
    solver = DASSO()
    mapping = DASSO._Mapping([(0,)], (0,), (3,))
    theta_design = solver._prepare_theta_design(mapping, (0,), [0, 3])
    feasible_candidates = np.array([[0.0], [0.25]])
    sample_mean_differences = np.array([1.0, 2.0])

    # With no noise, the optimal scale is n / (y^T C^-1 y).
    objectives = []
    parameters = []
    for (other_theta,) in feasible_candidates:
        precision = np.diag(np.ones(4))
        precision += np.diag(np.full(3, -other_theta), 1)
        precision += np.diag(np.full(3, -other_theta), -1)
        covariance = np.linalg.inv(precision)[np.ix_([0, 3], [0, 3])]
        quadratic = sample_mean_differences @ np.linalg.solve(covariance, sample_mean_differences)
        theta_0 = 2 / quadratic
        objectives.append(np.linalg.slogdet(covariance)[1] - 2 * np.log(theta_0) + 2)
        parameters.append([theta_0, -theta_0 * other_theta])

    actual = solver._fit_theta_candidates(
        theta_design,
        feasible_candidates,
        sparse.eye(2, format="csc"),
        np.zeros((2, 2)),
        sample_mean_differences,
    )

    # SLSQP uses its default stopping tolerance.
    np.testing.assert_allclose(actual, parameters[np.argmin(objectives)], atol=5e-5)
    assert actual[1] == pytest.approx(-actual[0] * 0.25)


@pytest.mark.parametrize(
    ("design_mean", "pareto_mean", "expected"),
    [(10.0, 0.0, [1, 4, 7]), (0.0, 10.0, [2, 5, 8]), (1.0, 1.0, [2, 5, 8])],
    ids=["design", "pareto", "tie-prefers-pareto"],
)
def test_dice_selects_restricted_set_by_expected_improvement(design_mean, pareto_mean, expected):
    solver = DASSO()
    mapping = DASSO._Mapping([(0,), (1,)], (0, 0), (2, 2))
    groups = [DASSO._Group(mapping, (i,), [2.0, -0.3], 1.0) for i in range(2)]
    run_state = _RunState(design_point_indices=[0, 4])
    pareto = {
        0: {"indices": [None], "mean": [0.0], "var": [0.0]},
        1: {"indices": [2], "mean": [pareto_mean], "var": [1.0]},
    }

    restricted_set = solver._select_restricted_set(
        run_state, mapping, groups, 0, 0, np.array([0.0, design_mean]), np.array([0.0, 1.0]), pareto
    )

    assert restricted_set == expected


def test_dice_decodes_pareto_combinations_in_group_order():
    solver = DASSO()
    mapping = DASSO._Mapping([(0,), (1,), (2,)], (0, 0, 0), (2, 2, 2))
    groups = [DASSO._Group(mapping, (i,), [2.0, -0.3], 1.0) for i in range(3)]
    pareto = {
        0: {"indices": [0, 1], "mean": [0.0, 1.0], "var": [1.0, 1.0]},
        1: {"indices": [None], "mean": [0.0], "var": [0.0]},
        2: {"indices": [1, 2], "mean": [4.0, 0.0], "var": [1.0, 1.0]},
    }

    restricted_set = solver._select_restricted_set(
        _RunState(design_point_indices=[0]),
        mapping,
        groups,
        1,
        0,
        np.array([0.0]),
        np.array([0.0]),
        pareto,
    )

    assert restricted_set == [10, 13, 16]


@pytest.mark.parametrize("restricted_set", [[0, 1], [1, 2]])
def test_dice_simulates_an_unseen_slice_only_when_needed(monkeypatch, restricted_set):
    solver = DASSO()
    run_state = _RunState(design_point_indices=[0])
    events = []
    mapping = DASSO._Mapping([(0,), (1,)], (0, 0), (2, 2))
    groups = [DASSO._Group(mapping, (i,), [2.0, -0.3], 1.0) for i in range(2)]
    for group in groups:
        monkeypatch.setattr(
            group, "add_design_points", lambda indices: events.append(("add", indices[0]))
        )
    monkeypatch.setattr(solver, "_dice_conditional_components", Mock(return_value=(None, None, {})))
    monkeypatch.setattr(solver, "_select_restricted_set", Mock(return_value=restricted_set))

    def record(run_state, solution_indices, ctx, mapping):
        solution_index = solution_indices[0]
        events.append(("simulate", solution_index))
        if solution_index not in run_state.design_point_indices:
            run_state.design_point_indices.append(solution_index)

    monkeypatch.setattr(solver, "_record_simulations", record)
    solver_rng = Mock(randrange=Mock(return_value=0), choice=Mock(return_value=2))

    assert solver._dice_stage(run_state, Mock(), mapping, groups, 0, solver_rng) == (
        0,
        restricted_set,
    )
    if 0 in restricted_set:
        assert events == [("simulate", 0)]
        solver_rng.choice.assert_not_called()
    else:
        assert events == [("simulate", 0), ("simulate", 2), ("add", 2), ("add", 2)]
        solver_rng.choice.assert_called_once_with(restricted_set)


def test_slice_distribution_matches_dense_gaussian_conditioning():
    solver = DASSO()
    mapping = DASSO._Mapping([(0,), (1,)], (0, 0), (3, 1))
    group_g = DASSO._Group(mapping, (0,), [2.0, -0.3], 1.0)
    run_state = _RunState(
        design_point_indices=[0, 6],
        sample_means_vec_d=np.array([1.0, 4.0]),
        noise_cov_mat_diagonals=np.array([0.2, 0.4]),
    )

    reordered_solutions, cond_mean_vec, q_bar_inv, sample_best_index = (
        solver._slice_conditional_distribution(run_state, mapping, group_g, [0, 2, 4, 6])
    )

    precision = 2.0 * np.eye(4) - 0.3 * (np.eye(4, k=1) + np.eye(4, k=-1))
    covariance = np.linalg.inv(precision)
    observation_covariance = covariance[np.ix_([0, 3], [0, 3])] + np.diag([0.2, 0.4])
    weights = np.linalg.solve(observation_covariance, np.ones(2))
    beta = weights @ np.array([1.0, 4.0]) / weights.sum()
    posterior_precision = precision.copy()
    posterior_precision[0, 0] += 1 / 0.2
    posterior_precision[3, 3] += 1 / 0.4
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_covariance[:, [0, 3]] @ ((np.array([1.0, 4.0]) - beta) / [0.2, 0.4])

    assert reordered_solutions == [2, 4, 0, 6]
    assert sample_best_index == 2
    np.testing.assert_allclose(q_bar_inv, posterior_covariance[np.ix_([1, 2, 0, 3], [1, 2, 0, 3])])
    np.testing.assert_allclose(cond_mean_vec, posterior_mean[[1, 2, 0, 3]])


def test_slice_expected_improvement_excludes_sample_best():
    solver = DASSO()
    cond_mean_vec = np.array([0.0, 1.0, 3.0])
    q_bar_inv = np.array([[2.0, 0.2, 0.0], [0.2, 1.0, 0.3], [0.0, 0.3, 2.0]])

    cei_values = solver._slice_expected_improvement(cond_mean_vec, q_bar_inv, 0)

    mean = np.array([-1.0, -3.0])
    std = np.sqrt([2.6, 4.0])
    assert cei_values[0] == 0
    np.testing.assert_allclose(
        cei_values[1:], mean * norm.cdf(mean / std) + std * norm.pdf(mean / std)
    )


@pytest.mark.parametrize("remaining", [0, 1, 2])
def test_slice_preserves_group_updates_and_simulation_order_at_budget_exhaustion(
    monkeypatch, remaining
):
    solver = DASSO()
    run_state = _RunState(design_point_indices=[0])
    budget = Budget(remaining)
    events = []
    mapping = DASSO._Mapping([(0,), (1,)], (0, 0), (2, 2))
    groups = [DASSO._Group(mapping, (i,), [2.0, -0.3], 1.0) for i in range(2)]
    for group in groups:
        monkeypatch.setattr(
            group, "add_design_points", lambda indices: events.append(("add", indices[0]))
        )
    monkeypatch.setattr(
        solver,
        "_slice_conditional_distribution",
        Mock(return_value=([2, 0], np.zeros(2), np.eye(2), 1)),
    )
    monkeypatch.setattr(
        solver, "_slice_expected_improvement", Mock(return_value=np.array([1.0, 0.0]))
    )

    def record(run_state, solution_indices, ctx, mapping):
        budget.request(1)
        events.append(("simulate", solution_indices[0]))

    monkeypatch.setattr(solver, "_record_simulations", record)
    if remaining < 2:
        with pytest.raises(BudgetExhaustedError):
            solver._slice_stage(run_state, Mock(), mapping, groups, 0, [0, 2])
    else:
        solver._slice_stage(run_state, Mock(), mapping, groups, 0, [0, 2])

    assert events == [("add", 2), ("add", 2)] + [("simulate", i) for i in [2, 0][:remaining]]
    assert budget.used == remaining
