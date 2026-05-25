"""Dice and Slice Simulation Optimization (DASSO).

An algorithm for large-scale, computationally expensive discrete simulation optimization
problems whose feasible solutions are defined on a finite subset of a high-dimensional
integer lattice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Annotated, ClassVar, Self

import numpy as np
from pydantic import Field, model_validator
from scipy import optimize, sparse
from scipy.special import (
    ndtr,
)  # cumulative distribution of the standard normal distribution
from scipy.stats import qmc

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solver,
    SolverConfig,
    VariableType,
)
from simopt.solvers._dasso import _numpy_seed_from_mrg, build_lhs


class DASSOConfig(SolverConfig):
    """Configuration for DASSO solver."""

    sample_size: Annotated[int, Field(default=10, gt=0, description="sample size per solution")]
    n_points_for_estimation: Annotated[
        int,
        Field(
            default=10,
            gt=0,
            description=("number of reference points to be used for parameter estimation"),
        ),
    ]
    decomposition: Annotated[
        list[tuple[int, ...]],
        Field(default_factory=list, description="decomposition of dimensions"),
    ]

    @model_validator(mode="after")
    def _validate_decomposition(self) -> Self:
        if not self.decomposition:
            return self
        coord_sets = [set(coord_ids) for coord_ids in self.decomposition]
        for i, coord_ids1 in enumerate(coord_sets):
            for j, coord_ids2 in enumerate(coord_sets):
                if i != j and not coord_ids1.isdisjoint(coord_ids2):
                    raise ValueError("Coordinate ids in the decomposition must be disjoint.")
        return self


@dataclass
class _RunState:
    design_point_indices: list[int] = field(default_factory=list)
    design_points_actual: list = field(default_factory=list)
    sample_means_vec_d: np.ndarray = field(default_factory=lambda: np.array([]))
    noise_cov_mat_diagonals: np.ndarray = field(default_factory=lambda: np.array([]))


class DASSO(Solver):
    """Dice and Slice Simulation Optimization (DASSO).

    An algorithm for large-scale, computationally expensive discrete simulation
    optimization problems whose feasible solutions are defined on a finite subset of a
    high-dimensional integer lattice.
    """

    name: str = "DASSO"
    config_class: ClassVar[type[SolverConfig]] = DASSOConfig
    class_name_abbr: ClassVar[str] = "DASSO"
    class_name: ClassVar[str] = "Dice and Slice Simulation Optimization"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_needed: ClassVar[bool] = False

    def __init__(self, name: str = "DASSO", fixed_factors: dict | None = None) -> None:
        """Initialize the DASSO solver.

        Args:
            name (str, optional): The name of the solver. Defaults to "DASSO".
            fixed_factors (dict, optional): Fixed factors of the solver. Defaults to
                None.
        """
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

    def _solver_rng(self) -> MRG32k3a:
        """Return the RNG used for solver-internal randomness."""
        if len(self.rng_list) < 3:
            error_msg = "DASSO requires an internal RNG before solve() is called."
            raise RuntimeError(error_msg)
        return self.rng_list[2]

    def _numpy_rng(self) -> np.random.Generator:
        """Create a NumPy generator derived from the solver RNG."""
        return np.random.default_rng(_numpy_seed_from_mrg(self._solver_rng()))

    def solve(self, problem: Problem) -> None:
        run_state = _RunState()
        solver_rng = self._solver_rng()

        # Problem specifications.
        dim = problem.dim
        lower_bounds = problem.lower_bounds
        upper_bounds = problem.upper_bounds

        # Decomposition.
        if len(self.factors["decomposition"]) >= 1:
            decomposition = self.factors["decomposition"]
        else:
            # Consider two groups unless decomposition is specified.
            decomposition = [
                tuple(range(int(dim / 2))),
                tuple(range(int(dim / 2), dim)),
            ]

        # Mapping.
        mapping = self._Mapping(decomposition, lower_bounds, upper_bounds)

        # Parameter estimation.
        theta_parameters, random_effect_variances, beta_0 = self._hyperparameter_estimation(
            run_state, problem, mapping, decomposition
        )

        # Identify the sample-best solution.
        best_solution_candidate_index = run_state.sample_means_vec_d.argmin()
        self.recommended_solns.append(run_state.design_points_actual[best_solution_candidate_index])
        self.intermediate_budgets.append(self.budget.used)

        # Group creation.
        groups = [
            DASSO._Group(
                mapping,
                coord_ids,
                theta_parameters[coord_ids],
                random_effect_variances[coord_ids],
            )
            for coord_ids in decomposition
        ]

        for group in groups:
            group.add_design_points(run_state.design_point_indices)

        # Iterate each stage until the algorithm terminates.
        while True:
            sample_best_solution_index = run_state.design_point_indices[
                best_solution_candidate_index
            ]

            # -- Dice Stage --
            # Update random effect group.
            random_effect_group_index = solver_rng.randrange(len(groups))

            # Calculate the covariance matrix corresponding to design points, Sigma_DD
            num_design_points = len(run_state.design_point_indices)
            cov_mat_dd = sum(
                group.transformed_cov_mat_dd
                if group_index != random_effect_group_index
                else group.random_effect_variance * np.eye(num_design_points)
                for group_index, group in enumerate(groups)
            )

            # Estimate the location parameter.
            ones_vec = np.ones(num_design_points)  # vector of ones, 1_D
            part1 = (
                1 / sum(np.linalg.solve(cov_mat_dd, ones_vec)) * ones_vec
            )  # (1_D * Sigma^-1 * 1_D^T)^-1 * 1_D^T
            part2 = np.linalg.solve(
                cov_mat_dd, run_state.sample_means_vec_d
            )  # Sigma^-1 * \bar{Y}_D
            beta_0 = part1 @ part2

            # Get vectors needed to compute CEI for solutions in D and find
            # Pareto-efficient points.
            sample_means_vec_d_stand = run_state.sample_means_vec_d - beta_0  # \bar{Y}_D - beta_0
            cond_var_of_diff_vec_d = np.zeros(
                num_design_points
            )  # v(x) + v(x_best) - 2 * c(x_best, x) for x in D
            cond_mean_of_diff_vec_d = np.zeros(num_design_points)  # m(x_best) - m(x) for x in D

            covariance_matrix_d = cov_mat_dd + np.diag(
                run_state.noise_cov_mat_diagonals
            )  # Sigma_DD + Q_epsilon^-1
            eye_d = np.eye(num_design_points)
            covariance_matrix_inv_d = np.linalg.solve(
                covariance_matrix_d, eye_d
            )  # inverse of Sigma_DD + Q_epsilon^-1

            cond_dist_comp_of_diff_pareto_each_group = {}
            for group_index, group in enumerate(groups):
                is_random_effect = group_index == random_effect_group_index
                comp_d, comp_pareto = group.get_cond_dist_components(
                    sample_best_solution_index,
                    covariance_matrix_inv_d,
                    sample_means_vec_d_stand,
                    is_random_effect,
                )
                cond_dist_comp_of_diff_pareto_each_group[group_index] = comp_pareto

                cond_var_of_diff_vec_d += comp_d["var"]
                cond_mean_of_diff_vec_d += comp_d["mean"]

            # Calculate CEI for design points.
            cond_std_of_diff_vec_d = np.sqrt(cond_var_of_diff_vec_d)  # sqrt{v(x_best, x)}
            sample_best_index = run_state.design_point_indices.index(sample_best_solution_index)
            cond_std_of_diff_vec_d[sample_best_index] = np.inf

            stand_mean_of_diff_vec_d = (
                cond_mean_of_diff_vec_d / cond_std_of_diff_vec_d
            )  # [m(x_best) - m(x)] / sqrt{v(x_best, x)}
            cei_values_d = cond_mean_of_diff_vec_d * ndtr(
                stand_mean_of_diff_vec_d
            ) + cond_std_of_diff_vec_d * self._standard_normal_pdf(stand_mean_of_diff_vec_d)
            cei_values_d[sample_best_index] = 0
            max_cei_index_d = np.argmax(cei_values_d)

            # Construct Pareto frontier, set check{F}.
            cond_mean_of_diff_vec_f = np.array([0])
            cond_var_of_diff_vec_f = np.array([0])
            for (
                cond_dist_comp_of_diff_pareto_group
            ) in cond_dist_comp_of_diff_pareto_each_group.values():
                cond_mean_of_diff_vec_f = np.concatenate(
                    [
                        cond_mean_of_diff_vec_f + mean_p
                        for mean_p in cond_dist_comp_of_diff_pareto_group["mean"]
                    ]
                )
                cond_var_of_diff_vec_f = np.concatenate(
                    [
                        cond_var_of_diff_vec_f + var_p
                        for var_p in cond_dist_comp_of_diff_pareto_group["var"]
                    ]
                )

            # Calculate CEI for Pareto frontier.
            cond_std_of_diff_vec_f = np.sqrt(cond_var_of_diff_vec_f)  # sqrt{v(x_best, x)}
            stand_mean_of_diff_vec_f = (
                cond_mean_of_diff_vec_f / cond_std_of_diff_vec_f
            )  # [m(x_best) - m(x)] / sqrt{v(x_best, x)}
            cei_values_f = cond_mean_of_diff_vec_f * ndtr(
                stand_mean_of_diff_vec_f
            ) + cond_std_of_diff_vec_f * self._standard_normal_pdf(stand_mean_of_diff_vec_f)

            max_cei_index_f = np.argmax(cei_values_f)

            # Find a slice with the largest CEI.
            if (
                cei_values_d[max_cei_index_d] > cei_values_f[max_cei_index_f]
            ):  # if it is a design point
                max_cei_solution = run_state.design_point_indices[max_cei_index_d]

                best_non_g_comp = tuple(
                    mapping.get_low_dim_comp_ind_from_sol_index(max_cei_solution, group.coord_ids)
                    if group_index != random_effect_group_index
                    else None
                    for group_index, group in enumerate(groups)
                )
            else:
                best_non_g_comp = ()
                tmp = max_cei_index_f
                for i in range(len(groups)):
                    indices = cond_dist_comp_of_diff_pareto_each_group[i]["indices"]
                    best_non_g_comp += (indices[tmp % len(indices)],)
                    tmp = tmp // len(indices)

            # Find the restricted set.
            non_g_comp_as_tuple = {
                group.coord_ids: comp for group, comp in zip(groups, best_non_g_comp, strict=True)
            }
            restricted_set = mapping.get_solution_indices_from_non_g_component(non_g_comp_as_tuple)

            # Simulate the sample best solution.
            self._record_simulations(run_state, [sample_best_solution_index], problem, mapping)

            # Simulate a solution from the restricted set if it does not have any
            # solution that has been simulated.
            if not any(sol in run_state.design_point_indices for sol in restricted_set):
                solution_index = solver_rng.choice(restricted_set)
                self._record_simulations(run_state, [solution_index], problem, mapping)
                for group in groups:
                    group.add_design_points([solution_index])

            # -- Slice Stage --
            group_g = groups[random_effect_group_index]

            solutions_u, solutions_d = [], []
            for sol in restricted_set:
                solutions_d.append(
                    sol
                ) if sol in run_state.design_point_indices else solutions_u.append(sol)
            reordered_solutions = solutions_u + solutions_d
            low_dim_comp_indices_reordered = [
                mapping.get_low_dim_comp_ind_from_sol_index(i, group_g.coord_ids)
                for i in reordered_solutions
            ]

            num_u = len(solutions_u)
            num_d = len(solutions_d)

            design_point_indices_slice = [
                run_state.design_point_indices.index(i) for i in solutions_d
            ]
            noise_prec_mat = np.diag(
                1 / run_state.noise_cov_mat_diagonals[design_point_indices_slice]
            )
            noise_cov_mat = np.diag(run_state.noise_cov_mat_diagonals[design_point_indices_slice])

            # Conditional precision matrix: \bar{Q}
            low_dim_comp_u = low_dim_comp_indices_reordered[:num_u]
            low_dim_comp_d = low_dim_comp_indices_reordered[num_u:]
            uu, ud, du, dd = group_g.get_precision_matrix_components(low_dim_comp_u, low_dim_comp_d)

            q_bar = np.vstack(
                [
                    np.hstack([uu.toarray(), ud.toarray()]),
                    np.hstack([du.toarray(), dd.toarray() + noise_prec_mat]),
                ]
            )

            # \bar{Q}^{-1}
            q_bar_inv = np.linalg.solve(q_bar, np.eye(num_u + num_d))

            # # Calculate the conditional mean vector.
            sample_means_vec_d_slice = run_state.sample_means_vec_d[
                design_point_indices_slice
            ]  # \bar{Y}_D

            # Estimate mean.
            schur_complement_mat = dd - du @ sparse.linalg.spsolve(uu, ud)
            if schur_complement_mat.size == 1:
                cov_mat = 1 / np.array(schur_complement_mat) + noise_cov_mat
            else:
                cov_mat = (
                    np.linalg.solve(schur_complement_mat.toarray(), np.eye(len(low_dim_comp_d)))
                    + noise_cov_mat
                )

            ones_vec = np.ones(num_d)  # vector of ones, 1_D
            part1 = (
                1 / sum(np.linalg.solve(cov_mat, ones_vec)) * ones_vec
            )  # (1_D * Sigma^-1 * 1_D^T)^-1 * 1_D^T
            part2 = np.linalg.solve(cov_mat, sample_means_vec_d_slice)  # Sigma^-1 * \bar{Y}_D
            beta = part1 @ part2

            comp_d = noise_prec_mat @ (sample_means_vec_d_slice - beta)
            # Vertical stack: 0_|U| and [[B^-1]^-1 * (\bar{Y}_D - mu_D)
            cond_mean_comp = np.concatenate([np.zeros(num_u), comp_d])

            cond_mean_vec = q_bar_inv @ cond_mean_comp

            sample_best_index = num_u + sample_means_vec_d_slice.argmin()

            # Conditional variance and covariance vectors
            cond_var_vec = q_bar_inv.diagonal()  # v(x)
            cond_cov_vec = q_bar_inv[sample_best_index, :]  # c(x_best, x)

            # Calculate CEI values.
            # conditional mean of difference Y(x_best) - Y(x): m(x_best) - m(x)
            cond_mean_of_diff_vec = cond_mean_vec[sample_best_index] - cond_mean_vec
            # Conditional variance of difference Y(x_best) - Y(x): v(x_best, x)
            # = v(x_best) + v(x) - 2 * c(x_best, x)
            cond_var_of_diff_vec = cond_var_vec + cond_var_vec[sample_best_index] - 2 * cond_cov_vec
            cond_std_of_diff_vec = np.sqrt(cond_var_of_diff_vec)  # sqrt{v(x_best, x)}
            cond_std_of_diff_vec[sample_best_index] = np.inf

            # [m(x_best) - m(x)] / sqrt{v(x_best, x)}
            stand_mean_of_diff_vec = cond_mean_of_diff_vec / cond_std_of_diff_vec

            cei_values = cond_mean_of_diff_vec * ndtr(
                stand_mean_of_diff_vec
            ) + cond_std_of_diff_vec * self._standard_normal_pdf(stand_mean_of_diff_vec)
            cei_values[sample_best_index] = 0

            # Simulate max-CEI solution.
            max_cei_index = np.argmax(cei_values)
            max_cei_solution = reordered_solutions[max_cei_index]
            if max_cei_solution not in run_state.design_point_indices:
                for group in groups:
                    group.add_design_points([max_cei_solution])
            self._record_simulations(run_state, [max_cei_solution], problem, mapping)

            # Simulate sample-best solution of the restricted set.
            sample_best_solution_restricted = reordered_solutions[sample_best_index]
            self._record_simulations(run_state, [sample_best_solution_restricted], problem, mapping)

            # Identify the sample best solution.
            best_solution_candidate_index = run_state.sample_means_vec_d.argmin()
            self.recommended_solns.append(
                run_state.design_points_actual[best_solution_candidate_index]
            )
            self.intermediate_budgets.append(self.budget.used)

    class _Mapping:
        """Map between solutions and coordinates."""

        def __init__(self, decomposition: list, lower_bounds: tuple, upper_bounds: tuple) -> None:
            """Initialize a _Mapping object.

            Create a mapping between solutions and coordinates.

            Args:
                decomposition (list): Coordinates for each group.
                lower_bounds (tuple): Lower bound for each coordinate of feasible set.
                upper_bounds (tuple): Upper bound for each coordinate of feasible set.
            """
            # Initialize the attributes.
            self._actual_values = [
                {i: lower_bound + i for i in range(upper_bound - lower_bound + 1)}
                for lower_bound, upper_bound in zip(lower_bounds, upper_bounds, strict=True)
            ]

            self._actual_values_reverse = [
                {v: k for k, v in vals.items()} for vals in self._actual_values
            ]

            num_point_in_coord = [len(values) for values in self._actual_values]
            self.num_points_in_group = {
                coord_ids: math.prod([num_point_in_coord[coord_id] for coord_id in coord_ids])
                for coord_ids in decomposition
            }

            self._index_multipliers = [
                math.prod(num_point_in_coord[i + 1 :]) for i in range(len(num_point_in_coord))
            ]

            self._index_multipliers_group = {}
            for coord_ids in decomposition:
                num_points = [num_point_in_coord[coord_id] for coord_id in coord_ids]
                self._index_multipliers_group[coord_ids] = [
                    math.prod(num_points[i + 1 :]) for i in range(len(coord_ids))
                ]

            # Construct the neighbors for each coordinate (dimension).
            values = [list(vals.keys()) for vals in self._actual_values]
            neighbors_in_coord = [
                {
                    **{
                        node: [left, right]
                        for node, left, right in zip(
                            vals[1:],
                            vals,
                            vals[2:],
                            strict=False,  # TODO: It might be a bug.
                        )
                    },
                    **{vals[0]: [vals[1]]},
                    **{vals[-1]: [vals[-2]]},
                }
                for vals in values
            ]

            self.neighbors_indices_in_group = {}
            for coord_ids in decomposition:
                index_multipliers = self._index_multipliers_group[coord_ids]
                neighbors_indices = []
                for low_dim_comp_ind in range(self.num_points_in_group[coord_ids]):
                    comp = self._get_low_dim_comp_from_index(low_dim_comp_ind, coord_ids)
                    neighbors = {
                        0: [low_dim_comp_ind]
                    }  # the neighbors in 0-th dimension, i.e., itself
                    for ind, coord_id in enumerate(coord_ids):
                        neighbors_comp = [
                            tuple(n if i == ind else v for i, v in enumerate(comp))
                            for n in neighbors_in_coord[coord_id][comp[ind]]
                        ]
                        neighbors[ind + 1] = [
                            sum(
                                multiplier * coord_value
                                for multiplier, coord_value in zip(
                                    index_multipliers, comp, strict=True
                                )
                            )
                            for comp in neighbors_comp
                        ]

                    neighbors_indices.append(neighbors)
                self.neighbors_indices_in_group[coord_ids] = neighbors_indices

        def _get_solution_from_index(self, solution_index: int) -> tuple:
            """Find the solution from the given index from the feasible solution set.

            Args:
                solution_index (int): The index of a solution.

            Returns:
                tuple: The solution from the given index.
            """
            solution = []
            for multiplier in self._index_multipliers:
                coord_value = solution_index // multiplier
                solution += [coord_value]
                solution_index -= multiplier * coord_value
            return tuple(solution)

        def get_solution_index(self, solution_actual: tuple) -> int:
            """Find the index of the given solution with actual values.

            Args:
                solution_actual (tuple): A solution with actual values from the feasible
                    solution set.

            Returns:
                int: The index of the given solution.
            """
            solution = (
                self._actual_values_reverse[coord_ind][coord_value]
                for coord_ind, coord_value in enumerate(solution_actual)
            )
            return sum(
                multiplier * coord_value
                for multiplier, coord_value in zip(self._index_multipliers, solution, strict=True)
            )

        def get_low_dim_comp_ind_from_sol_index(self, solution_index: int, coord_ids: tuple) -> int:
            """Find the lower dimensional component index for the given group.

            Use the provided solution index.

            Args:
                solution_index (int): The index of a solution.
                coord_ids (tuple): The lower dimensional component coordinate ids.

            Returns:
                int: The lower dimensional component index.
            """
            solution = self._get_solution_from_index(solution_index)
            index_multipliers = self._index_multipliers_group[coord_ids]
            low_dim_comp = [solution[coord_id] for coord_id in coord_ids]
            return sum(
                multiplier * coord_value
                for multiplier, coord_value in zip(index_multipliers, low_dim_comp, strict=True)
            )

        def _get_low_dim_comp_from_index(
            self, lower_dimensional_component_index: int, coord_ids: tuple
        ) -> tuple:
            """Find the lower dimensional component from a component index.

            Args:
                lower_dimensional_component_index (int): The index of a lower
                    dimensional component.
                coord_ids (tuple): The lower dimensional component coordinate ids.

            Returns:
                tuple: The lower dimensional component index.
            """
            index_multipliers = self._index_multipliers_group[coord_ids]
            low_dim_comp = []
            for multiplier in index_multipliers:
                coord_value = lower_dimensional_component_index // multiplier
                low_dim_comp += [coord_value]
                lower_dimensional_component_index -= multiplier * coord_value
            return tuple(low_dim_comp)

        def get_actual_values_of_solution_from_index(self, solution_index: int) -> tuple:
            """Find the solution with its actual values for the given solution index.

            Args:
                solution_index (int): The index of a solution.

            Returns:
                tuple: The solution with its actual values in each coordinate for the
                    given solution index.
            """
            solution = self._get_solution_from_index(solution_index)
            return tuple(
                self._actual_values[coord_id][ind] for coord_id, ind in enumerate(solution)
            )

        def get_solution_indices_from_non_g_component(self, non_g_component: dict) -> list:
            """Find the solution indices with the given non-g component.

            Args:
                non_g_component (dict): The lower dimensional component for each group
                    except the random-effect group.

            Returns:
                list: The solution indices with the given non-g component.
            """
            solution: list[int | None] = [None for _ in range(len(self._actual_values))]
            for coord_ids, low_dim_comp_index in non_g_component.items():
                if low_dim_comp_index is not None:
                    low_dim_comp = self._get_low_dim_comp_from_index(low_dim_comp_index, coord_ids)
                    for coord_id, coord_comp in zip(coord_ids, low_dim_comp, strict=True):
                        solution[coord_id] = coord_comp

            solution_indices = []
            coord_ids_random_effect = next(
                coord_ids
                for coord_ids, low_dim_comp_index in non_g_component.items()
                if low_dim_comp_index is None
            )
            for low_dim_comp_index in range(self.num_points_in_group[coord_ids_random_effect]):
                low_dim_comp = self._get_low_dim_comp_from_index(
                    low_dim_comp_index, coord_ids_random_effect
                )
                for coord_id, coord_comp in zip(coord_ids_random_effect, low_dim_comp, strict=True):
                    solution[coord_id] = coord_comp

                solution_index = 0
                for multiplier, coord_value in zip(self._index_multipliers, solution, strict=True):
                    if coord_value is None:
                        raise RuntimeError("Decomposition did not cover every coordinate.")
                    solution_index += multiplier * coord_value
                solution_indices.append(solution_index)

            return solution_indices

    def _hyperparameter_estimation(
        self,
        run_state: _RunState,
        problem: Problem,
        mapping: DASSO._Mapping,
        decomposition: list,
    ) -> tuple[dict, dict, float]:
        """Estimate the hyperparameters for each group and the overall mean.

        Args:
            run_state (_RunState): Run-scoped storage for design point data.
            problem (Problem): The problem instance providing bounds and function
                evaluations.
            mapping (DASSO._Mapping): The mapping instance providing mapping between
                solutions and coordinates.
            decomposition (list): Coordinates for each group.

        Returns:
            dict: The estimated theta values for the precision matrix of each group.
            dict: The estimated random-effect variance for each group.
            float: The estimated location parameter.
        """
        # Problem specifications.
        lower_bound: list[int | float] = list(problem.lower_bounds)
        upper_bound: list[int | float] = list(problem.upper_bounds)
        dim = len(lower_bound)

        # Default values.
        n_init: int = self.factors["n_points_for_estimation"]
        solver_rng = self._solver_rng()

        # Create the reference points.
        bounds: dict[int | str, list[int | float]] = {
            i: [lower_bound[i], upper_bound[i]] for i in range(dim)
        }
        ref_points = (
            build_lhs(bounds, n_init, rng=solver_rng)
            .round()
            .astype(int)
            .apply(tuple, axis=1)
            .unique()
            .tolist()
        )

        # For each reference point and each group, find a point to pair with the
        # reference point.
        paired_points_with_each_ref_point = {}
        for ref_point in ref_points:
            paired_points_ind = {}
            for coord_ids in decomposition:
                paired_point = ()
                for coord_id, ind in enumerate(ref_point):
                    if coord_id not in coord_ids:
                        paired_point += (ind,)
                    elif ind == bounds[coord_id][0]:  # equal to the lowest index value
                        paired_point += (ind + 1,)
                    elif ind == bounds[coord_id][1]:  # equal to the largest index value
                        paired_point += (ind - 1,)
                    else:  # randomly +1 or -1
                        paired_point += (ind + 2 * (solver_rng.random() < 0.5) - 1,)
                paired_points_ind[coord_ids] = paired_point
            paired_points_with_each_ref_point[ref_point] = paired_points_ind

        # Create design points (by avoiding having the same point more than once).
        design_points = []
        for ref_point, paired_points in paired_points_with_each_ref_point.items():
            if ref_point not in design_points:
                design_points.append(ref_point)
            for paired_point in paired_points.values():
                if paired_point not in design_points:
                    design_points.append(paired_point)

        # Prepare difference matrices B^(rho)'s for construction as (row, column, data).
        diff_mat_construction_each_group = {coord_ids: [] for coord_ids in decomposition}
        for ref_point_id, (ref_point, paired_points) in enumerate(
            paired_points_with_each_ref_point.items()
        ):
            ref_point_index_in_list = design_points.index(ref_point)
            for coord_ids, paired_point in paired_points.items():
                paired_point_index_in_list = design_points.index(paired_point)
                diff_mat_construction_each_group[coord_ids] += [
                    (ref_point_id, ref_point_index_in_list, 1)
                ]
                diff_mat_construction_each_group[coord_ids] += [
                    (ref_point_id, paired_point_index_in_list, -1)
                ]

        # Construct difference matrices B^(rho)'s.
        num_ref_points = len(ref_points)
        num_design_points = len(design_points)
        diff_matrices = {}
        for coord_ids, construction_values in diff_mat_construction_each_group.items():
            rows, columns, data = list(zip(*construction_values, strict=True))
            diff_mat = sparse.coo_matrix(
                (data, (rows, columns)), shape=(num_ref_points, num_design_points)
            ).tocsc()
            diff_matrices[coord_ids] = diff_mat

        # Parameter estimation.
        design_point_indices = [mapping.get_solution_index(sol) for sol in design_points]
        self._record_simulations(run_state, design_point_indices, problem, mapping)

        # Estimate hyperparameters for each group.
        random_effect_variances = {}
        theta_parameters = {}
        transformed_cov_matrices_dd = {}
        for coord_ids in decomposition:
            diff_mat = diff_matrices[coord_ids]
            random_effect_variances[coord_ids] = self._estimate_random_effect_variance(
                run_state.sample_means_vec_d,
                run_state.noise_cov_mat_diagonals,
                diff_mat,
            )
            theta_parameters[coord_ids], transformed_cov_matrices_dd[coord_ids] = (
                self._estimate_theta(
                    mapping,
                    coord_ids,
                    design_point_indices,
                    diff_mat,
                    run_state.sample_means_vec_d,
                    run_state.noise_cov_mat_diagonals,
                )
            )

        # Sigma_DD + Q_epsilon^-1
        cov_sum = sum(transformed_cov_matrices_dd.values()) + np.diag(
            run_state.noise_cov_mat_diagonals
        )
        # Estimate the location parameter.
        num_design_points, _ = cov_sum.shape
        ones_vec = np.ones(num_design_points)  # vector of ones, 1_D
        part1 = (
            1 / sum(np.linalg.solve(cov_sum, ones_vec)) * ones_vec
        )  # (1_D * Sigma^-1 * 1_D^T)^-1 * 1_D^T
        part2 = np.linalg.solve(cov_sum, run_state.sample_means_vec_d)  # Sigma^-1 * \bar{Y}_D
        beta_0 = part1 @ part2

        return theta_parameters, random_effect_variances, beta_0

    @staticmethod
    def _estimate_random_effect_variance(
        sample_means_vec: np.ndarray,
        noise_cov_diagonals: np.ndarray,
        diff_mat: sparse.csc_matrix,
    ) -> float:
        """Estimate the random-effect variance via MLE.

        Args:
            sample_means_vec (np.ndarray): The vector of sample means of initial design
                points.
            noise_cov_diagonals (np.ndarray): The diagonal elements of the intrinsic
                covariance matrix of noise.
            diff_mat (sparse.csc_matrix): The matrix with 1's and -1's to be used for
                efficient computation of MLE.

        Returns:
            float: The estimated random effect variance.
        """
        # Diagonals of B^(rho) Q_eps^-1 [B^(rho)]^T, which is a diagonal matrix since
        # Q_eps^-1 is diagonal.
        noise_cov_mat_diff_diagonals = np.abs(diff_mat) @ noise_cov_diagonals  # type: ignore

        sample_mean_differences = diff_mat @ sample_means_vec  # B * \bar{Y}
        diff_for_sigma_g = sample_mean_differences.transpose() ** 2 - noise_cov_mat_diff_diagonals
        sigma_g_upper_bound = np.max(diff_for_sigma_g) / 2
        if sigma_g_upper_bound < 1e-6:  # a small number
            random_effect_variance = 1e-6
        else:
            sigma_g_init = min(1e3, np.mean(diff_for_sigma_g) / 2)
            sigma_g_bounds = [(1e-6, min(1e3, sigma_g_upper_bound))]

            def _negative_log_likelihood_function_random_effect(
                sigma_g: float,
            ) -> float:
                """Calculate the negative log likelihood value."""
                cov_mat_b_inverse_diagonals = 1 / (2 * sigma_g + noise_cov_mat_diff_diagonals)
                if math.prod(cov_mat_b_inverse_diagonals) <= 1e-1000:
                    first_part = -math.inf
                else:
                    first_part = math.log(
                        math.prod(cov_mat_b_inverse_diagonals)
                    )  # log | (Sigma^(rho)_B)^-1 |
                second_part = sum(sample_mean_differences**2 * cov_mat_b_inverse_diagonals)
                return -(first_part - second_part)  # negative likelihood (1/2 is ignored)

            res = optimize.minimize(
                _negative_log_likelihood_function_random_effect,
                sigma_g_init,
                method="SLSQP",
                bounds=sigma_g_bounds,
                options={"maxiter": 50},
            )
            random_effect_variance = res.x[0]

        return random_effect_variance

    def _estimate_theta(
        self,
        mapping: DASSO._Mapping,
        coordinate_ids: tuple,
        solution_indices: list,
        diff_mat: sparse.csc_matrix,
        sample_means_vec_d: np.ndarray,
        noise_cov_mat_diagonals: np.ndarray,
    ) -> tuple[list, np.ndarray]:
        """Estimate hyperparameters of the precision matrix via MLE.

        Args:
            mapping (DASSO._Mapping): The mapping instance providing mapping between
                solutions and coordinates.
            coordinate_ids (tuple): The lower dimensional component coordinate ids of a
                group.
            solution_indices (list[int]): The solution indices to be simulated.
            diff_mat (sparse.csc_matrix): The matrix with 1's and -1's to be used for
                efficient computation of MLE.
            sample_means_vec_d (np.ndarray): The vector of sample means of design
                points.
            noise_cov_mat_diagonals (np.ndarray): The diagonal noise covariance values.

        Returns:
            list: The estimated theta values for the precision matrix.
            np.ndarray: The transformed covariance matrix component corresponding to
                design points.
        """
        # Use simulated solutions to obtain the corresponding lower dimensional
        # component of the design points.
        low_dim_comp_indices_d = list(
            {
                mapping.get_low_dim_comp_ind_from_sol_index(sol_ind, coordinate_ids)
                for sol_ind in solution_indices
            }
        )
        low_dim_comp_indices_u = [
            low_dim_comp
            for low_dim_comp in range(mapping.num_points_in_group[coordinate_ids])
            if low_dim_comp not in low_dim_comp_indices_d
        ]

        # Set up the data to more efficiently compute the component of the covariance
        # matrix.
        neighbors_indices = mapping.neighbors_indices_in_group[coordinate_ids]
        num_u = len(low_dim_comp_indices_u)
        num_d = len(low_dim_comp_indices_d)

        mapping_d = {
            low_dim_comp_ind: ind for ind, low_dim_comp_ind in enumerate(low_dim_comp_indices_d)
        }
        mapping_u = {
            low_dim_comp_ind: ind for ind, low_dim_comp_ind in enumerate(low_dim_comp_indices_u)
        }

        rows_uu, columns_uu, dimension_ids_uu = [], [], []
        rows_ud, columns_ud, dimension_ids_ud = [], [], []
        for low_dim_com_ind, ind in mapping_u.items():
            for dimension_id, indices in neighbors_indices[low_dim_com_ind].items():
                for neighbor_low_dim_comp_ind in indices:
                    if neighbor_low_dim_comp_ind in mapping_u:
                        rows_uu.append(ind)
                        columns_uu.append(mapping_u[neighbor_low_dim_comp_ind])
                        dimension_ids_uu.append(dimension_id)
                    else:
                        rows_ud.append(ind)
                        columns_ud.append(mapping_d[neighbor_low_dim_comp_ind])
                        dimension_ids_ud.append(dimension_id)

        rows_dd, columns_dd, dimension_ids_dd = [], [], []
        for low_dim_com_ind, ind in mapping_d.items():
            for dimension_id, indices in neighbors_indices[low_dim_com_ind].items():
                for neighbor_low_dim_comp_ind in indices:
                    if neighbor_low_dim_comp_ind in mapping_d:
                        rows_dd.append(ind)
                        columns_dd.append(mapping_d[neighbor_low_dim_comp_ind])
                        dimension_ids_dd.append(dimension_id)

        # Rearrange the transformation matrix component corresponding to design points.
        low_dim_comp_indices = [
            mapping.get_low_dim_comp_ind_from_sol_index(sol_ind, coordinate_ids)
            for sol_ind in solution_indices
        ]
        num_solutions_d = len(solution_indices)

        rows = list(range(num_solutions_d))
        columns = [low_dim_comp_indices_d.index(ind) for ind in low_dim_comp_indices]
        data = [1] * num_solutions_d
        rearranged_tf_mat_d = sparse.coo_matrix(
            (data, (rows, columns)), shape=(num_solutions_d, num_d)
        ).tocsc()

        # B^(rho) Q_eps^-1 [B^(rho)]^T, which is a diagonal matrix since Q_eps^-1 is
        # diagonal.
        noise_cov_mat_diff = np.diag(np.abs(diff_mat) @ noise_cov_mat_diagonals)  # type: ignore

        diff_tf_mat_d = diff_mat @ rearranged_tf_mat_d  # B^(rho) T^(rho)_DD

        # Generate Sobol points to create a feasible candidate parameters.
        solver_rng = self._solver_rng()
        sampler = qmc.Sobol(len(coordinate_ids), rng=self._numpy_rng())
        candidates = sampler.random(2**12)  # without theta_0, all non-negative

        # Restrict the set of candidates for efficiency.
        candidates = candidates[:30]
        # Make the sum of candidate parameters less than 0.5
        feasible_candidates = np.array(
            [solver_rng.uniform(0, 0.5) * candidate / sum(candidate) for candidate in candidates]
        )

        # Find the best candidate
        sample_mean_differences = diff_mat @ sample_means_vec_d  # B^(rho) * \bar{Y}
        theta_0_init = 1 / np.var(sample_mean_differences)

        def _get_covariance_matrix_dd(precision_matrix_parameters: list) -> np.ndarray:
            """Compute the covariance matrix component for design points.

            Args:
                precision_matrix_parameters (list): The theta values representing the
                    correlation.

            Returns:
                np.ndarray: The covariance matrix component for design points.
            """
            data_uu = [precision_matrix_parameters[dim_id] for dim_id in dimension_ids_uu]
            data_ud = [precision_matrix_parameters[dim_id] for dim_id in dimension_ids_ud]
            data_dd = [precision_matrix_parameters[dim_id] for dim_id in dimension_ids_dd]

            uu = sparse.coo_matrix((data_uu, (rows_uu, columns_uu)), shape=(num_u, num_u)).tocsc()
            ud = sparse.coo_matrix((data_ud, (rows_ud, columns_ud)), shape=(num_u, num_d)).tocsc()
            du = ud.transpose().tocsc()
            dd = sparse.coo_matrix((data_dd, (rows_dd, columns_dd)), shape=(num_d, num_d)).tocsc()

            schur_complement_mat = dd - du @ sparse.linalg.spsolve(uu, ud)

            return np.linalg.solve(schur_complement_mat.toarray(), np.eye(num_d))

        objectives = {}
        for other_thetas in feasible_candidates:
            # Construct the components of precision matrix with theta_0 = 1
            prec_mat_inv_dd = _get_covariance_matrix_dd([1] + [-theta for theta in other_thetas])
            # theta_0 B^(rho) T^(rho)_DD Sigma^(rho)_DD [B^(rho) T^(rho)_DD]^T
            cov_mat_temp = diff_tf_mat_d @ prec_mat_inv_dd @ diff_tf_mat_d.transpose()

            def _negative_log_likelihood_function(
                theta_0: float, cov_mat_temp: np.ndarray = cov_mat_temp
            ) -> float:
                """Calculate the negative log likelihood value."""
                cov_mat_b = cov_mat_temp / float(theta_0) + noise_cov_mat_diff  # Sigma^(rho)_B

                # log | sigma_c^-1 | = - log | sigma_c |
                first_part = -np.prod(np.linalg.slogdet(cov_mat_b))
                second_part = sample_mean_differences.transpose() @ np.linalg.solve(
                    cov_mat_b, sample_mean_differences
                )
                return -(first_part - second_part)  # negative likelihood (1/2 is ignored)

            res = optimize.minimize(
                _negative_log_likelihood_function,
                theta_0_init,
                method="SLSQP",
                bounds=[(1e-6, None)],
                options={"maxiter": 50},
            )
            objectives[(res.x[0], *(-res.x[0] * theta for theta in other_thetas))] = res.fun

        # Get the hyperparameters with the lowest negative log likelihood value
        prec_mat_parameters = list(min(objectives, key=objectives.__getitem__))

        cov_mat_dd = _get_covariance_matrix_dd(prec_mat_parameters)
        transformed_cov_mat_dd = rearranged_tf_mat_d @ cov_mat_dd @ rearranged_tf_mat_d.transpose()

        return prec_mat_parameters, transformed_cov_mat_dd

    class _Group:
        """A _Group object contains group information for given hyperparameters."""

        def __init__(
            self,
            mapping: DASSO._Mapping,
            coordinate_ids: tuple,
            theta_parameters: list,
            random_effect_variance: float,
        ) -> None:
            """Initialize a _Group object, which contains the information of a group.

            Args:
                mapping (DASSO._Mapping): The mapping instance providing mapping between
                    solutions and coordinates.
                coordinate_ids (tuple): The lower dimensional component coordinate ids
                    of a group.
                theta_parameters (list): The theta values for the precision matrix.
                random_effect_variance (float): The variance of the random effect.
            """
            self._mapping = mapping
            self.coord_ids = coordinate_ids
            self._theta_parameters = theta_parameters

            num_low_dim_comp = mapping.num_points_in_group[coordinate_ids]

            # Use simulated solutions to obtain the corresponding lower dimensional
            # component of the design points.
            self._sol_indices_d = []
            self._low_dim_comp_indices_d = []
            self._low_dim_comp_indices_u = list(range(num_low_dim_comp))

            # Reorder indices.
            self._low_dim_comp_indices_reordered = list(range(num_low_dim_comp))

            # For initialization purposes.
            self._cov_mat_dd = np.array([[]])
            self._rearranged_tf_mat_d = sparse.csc_matrix((0, 0))
            self.transformed_cov_mat_dd = np.array([[]])

            self._Q_uu_inv = np.array([[]])
            self._Q_uu_inv_times_Q_ud = np.array([[]])

            self.random_effect_variance = random_effect_variance

        def add_design_points(self, new_solution_indices: list[int]) -> None:
            """Update the components of matrices with given a set of new design points.

            Args:
                new_solution_indices (list): The indices of new design points.
            """
            self._sol_indices_d += new_solution_indices
            new_low_dim_comp_added = False
            for sol_ind in new_solution_indices:
                low_dim_comp_ind = self._mapping.get_low_dim_comp_ind_from_sol_index(
                    sol_ind, self.coord_ids
                )
                if low_dim_comp_ind not in self._low_dim_comp_indices_d:
                    self._low_dim_comp_indices_d.append(low_dim_comp_ind)
                    self._low_dim_comp_indices_u.remove(low_dim_comp_ind)
                    new_low_dim_comp_added = True

            if new_low_dim_comp_added:
                self._low_dim_comp_indices_reordered = (
                    self._low_dim_comp_indices_u + self._low_dim_comp_indices_d
                )

                # Set up the data to more efficiently compute the component of the
                # covariance matrix.
                num_u = len(self._low_dim_comp_indices_u)
                num_d = len(self._low_dim_comp_indices_d)

                prec_mat_uu, prec_mat_ud, prec_mat_du, prec_mat_dd = (
                    self.get_precision_matrix_components(
                        self._low_dim_comp_indices_u, self._low_dim_comp_indices_d
                    )
                )

                self._Q_uu_inv = np.linalg.solve(prec_mat_uu.toarray(), np.eye(num_u))
                self._Q_uu_inv_times_Q_ud = self._Q_uu_inv @ prec_mat_ud

                schur_complement_mat = prec_mat_dd - prec_mat_du @ sparse.linalg.spsolve(
                    prec_mat_uu, prec_mat_ud
                )
                self._cov_mat_dd = np.linalg.solve(schur_complement_mat.toarray(), np.eye(num_d))

            if len(new_solution_indices) > 0:
                low_dim_comp_indices = [
                    self._mapping.get_low_dim_comp_ind_from_sol_index(sol_ind, self.coord_ids)
                    for sol_ind in self._sol_indices_d
                ]

                num_solutions_d = len(self._sol_indices_d)
                num_low_dim_comp_d = len(self._low_dim_comp_indices_d)

                rows = list(range(num_solutions_d))
                columns = [self._low_dim_comp_indices_d.index(ind) for ind in low_dim_comp_indices]
                data = [1] * num_solutions_d
                self._rearranged_tf_mat_d = sparse.coo_matrix(
                    (data, (rows, columns)), shape=(num_solutions_d, num_low_dim_comp_d)
                ).tocsc()
                self.transformed_cov_mat_dd = (
                    self._rearranged_tf_mat_d
                    @ self._cov_mat_dd
                    @ self._rearranged_tf_mat_d.transpose()
                )

        def get_cond_dist_components(
            self,
            sample_best_solution_index: int,
            covariance_matrix_inv_d: np.ndarray,
            sample_means_vec_d_stand: np.ndarray,
            is_random_effect: bool,
        ) -> tuple[dict, dict]:
            """Compute conditional mean/variance of differences for D and Pareto points.

            Args:
                sample_best_solution_index (int): The index of the current sample-best
                    solution.
                covariance_matrix_inv_d (np.ndarray): The inverse of covariance matrix
                    of Y_D + noise.
                sample_means_vec_d_stand (np.ndarray): The standardized sample mean
                    vector for solutions in D.
                is_random_effect (bool): True if it is the random-effect group.

            Returns:
                dict: The conditional means and variances of the differences for
                    solutions in D.
                dict: The conditional means and variances of the differences for
                    Pareto-efficient points.
            """
            if not is_random_effect:
                num_u = len(self._low_dim_comp_indices_u)
                num_d = len(self._low_dim_comp_indices_d)

                t_rho_d = self._rearranged_tf_mat_d.toarray()

                # Lower dimensional component of the sample-best solution
                low_dim_comp_sample_best = self._mapping.get_low_dim_comp_ind_from_sol_index(
                    sample_best_solution_index, self.coord_ids
                )
                best_ind = self._low_dim_comp_indices_reordered.index(low_dim_comp_sample_best)

                # S^(rho): Schur complement = cond_cov_mat_dd
                cov_mat_d_inv_tr = covariance_matrix_inv_d @ t_rho_d
                cov_mat_d_inv_tr_tr = t_rho_d.transpose() @ cov_mat_d_inv_tr
                tmp = np.linalg.solve(self._cov_mat_dd, np.eye(num_d)) - cov_mat_d_inv_tr_tr
                e_rho = covariance_matrix_inv_d + cov_mat_d_inv_tr @ np.linalg.solve(
                    tmp, cov_mat_d_inv_tr.transpose()
                )
                s_rho = self._cov_mat_dd - self._cov_mat_dd @ cov_mat_d_inv_tr_tr @ self._cov_mat_dd

                cond_cov_mat_ud = -self._Q_uu_inv_times_Q_ud @ s_rho
                cond_cov_mat_uu = (
                    self._Q_uu_inv - self._Q_uu_inv_times_Q_ud @ cond_cov_mat_ud.transpose()
                )

                # alculate the conditional variance and covariance.
                cond_var_vec = np.concatenate((cond_cov_mat_uu.diagonal(), s_rho.diagonal()))
                cond_cov_vec = np.concatenate(
                    (cond_cov_mat_ud[:, best_ind - num_u], s_rho[:, best_ind - num_u])
                )

                # Calculate the conditional mean vector.
                tmp_vec = t_rho_d.transpose() @ e_rho @ sample_means_vec_d_stand
                cond_mean_vec = np.concatenate((cond_cov_mat_ud @ tmp_vec, s_rho @ tmp_vec))

                # Conditional distribution components of the differences for each
                # lower dimensional component.
                cond_var_of_diff_vec = cond_var_vec[best_ind] + cond_var_vec - 2 * cond_cov_vec
                cond_mean_of_diff_vec = cond_mean_vec[best_ind] - cond_mean_vec

                # Conditional distribution components of the differences for solution in
                # D.
                cond_var_of_diff_vec_d = (
                    t_rho_d @ cond_var_of_diff_vec[-len(self._low_dim_comp_indices_d) :]
                )
                cond_mean_of_diff_vec_d = (
                    t_rho_d @ cond_mean_of_diff_vec[-len(self._low_dim_comp_indices_d) :]
                )

                # Pareto frontier as indices of the Pareto-efficient lower dimensional
                # components.
                num_points = len(cond_mean_of_diff_vec)
                pareto_frontier_indices = np.arange(num_points)

                next_point_index = 0  # Next index in the efficient_points array to search for
                while next_point_index < len(cond_mean_of_diff_vec):
                    nondominated_point_mask = ~(
                        (cond_mean_of_diff_vec < cond_mean_of_diff_vec[next_point_index])
                        * (cond_var_of_diff_vec < cond_var_of_diff_vec[next_point_index])
                    )
                    pareto_frontier_indices = pareto_frontier_indices[
                        nondominated_point_mask
                    ]  # Remove dominated points
                    cond_mean_of_diff_vec = cond_mean_of_diff_vec[nondominated_point_mask]
                    cond_var_of_diff_vec = cond_var_of_diff_vec[nondominated_point_mask]
                    next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

                pareto_points = [
                    self._low_dim_comp_indices_reordered[ind] for ind in pareto_frontier_indices
                ]
                cond_dist_of_diff_comp_f = {
                    "indices": pareto_points,
                    "mean": cond_mean_of_diff_vec,
                    "var": cond_var_of_diff_vec,
                }

            else:  # if random-effect group
                # Calculate the conditional covariance matrix.
                num_design_points, _ = covariance_matrix_inv_d.shape
                eye_d = np.eye(num_design_points)
                sigma_eye_d = self.random_effect_variance * eye_d

                # S^(g): Schur complement
                s_g = sigma_eye_d - self.random_effect_variance**2 * covariance_matrix_inv_d

                best_ind = self._sol_indices_d.index(sample_best_solution_index)

                # Calculate the conditional mean vector: S^(g) * H * (\bar{Y}_D - mu_D).
                cond_mean_vec_d = (
                    eye_d - s_g / self.random_effect_variance
                ) @ sample_means_vec_d_stand

                # Conditional distribution components of the differences for solutions
                # in D.
                cond_var_of_diff_vec_d = (
                    s_g[best_ind, best_ind] + s_g.diagonal() - 2 * s_g[best_ind, :]
                )
                cond_mean_of_diff_vec_d = cond_mean_vec_d[best_ind] - cond_mean_vec_d

                # Conditional distribution components of the differences for solutions
                # in U.
                cond_dist_of_diff_comp_f = {
                    "indices": np.array([None]),
                    "mean": np.array([cond_mean_vec_d[best_ind] - 0]),
                    "var": np.array(
                        [self.random_effect_variance + s_g[best_ind, best_ind] - 2 * 0]
                    ),
                }

            cond_dist_comp_of_diff_d = {
                "mean": cond_mean_of_diff_vec_d,
                "var": cond_var_of_diff_vec_d,
            }
            return cond_dist_comp_of_diff_d, cond_dist_of_diff_comp_f

        def get_precision_matrix_components(
            self, indices_u: list, indices_d: list
        ) -> list[sparse.coo_matrix]:
            """Compute the components of the precision matrix.

            Args:
                indices_u (list): The indices of the matrix corresponding to non-design
                    points.
                indices_d (list): The indices of the matrix corresponding to design
                    points.

            Returns:
                list[sparse.coo_matrix]: The components of the precision matrix.
            """
            neighbors_indices = self._mapping.neighbors_indices_in_group[self.coord_ids]

            mapping_d = {component_ind: ind for ind, component_ind in enumerate(indices_d)}
            mapping_u = {component_ind: ind for ind, component_ind in enumerate(indices_u)}

            rows_uu, columns_uu, data_uu = [], [], []
            rows_ud, columns_ud, data_ud = [], [], []
            for low_dim_com_ind, ind in mapping_u.items():
                for dimension_id, indices in neighbors_indices[low_dim_com_ind].items():
                    for neighbor_low_dim_comp_ind in indices:
                        if neighbor_low_dim_comp_ind in mapping_u:
                            rows_uu.append(ind)
                            columns_uu.append(mapping_u[neighbor_low_dim_comp_ind])
                            data_uu.append(self._theta_parameters[dimension_id])
                        else:
                            rows_ud.append(ind)
                            columns_ud.append(mapping_d[neighbor_low_dim_comp_ind])
                            data_ud.append(self._theta_parameters[dimension_id])

            rows_dd, columns_dd, data_dd = [], [], []
            for low_dim_com_ind, ind in mapping_d.items():
                for dimension_id, indices in neighbors_indices[low_dim_com_ind].items():
                    for neighbor_low_dim_comp_ind in indices:
                        if neighbor_low_dim_comp_ind in mapping_d:
                            rows_dd.append(ind)
                            columns_dd.append(mapping_d[neighbor_low_dim_comp_ind])
                            data_dd.append(self._theta_parameters[dimension_id])

            num_u = len(indices_u)
            num_d = len(indices_d)

            prec_mat_uu = sparse.coo_matrix(
                (data_uu, (rows_uu, columns_uu)), shape=(num_u, num_u)
            ).tocsc()
            prec_mat_ud = sparse.coo_matrix(
                (data_ud, (rows_ud, columns_ud)), shape=(num_u, num_d)
            ).tocsc()
            prec_mat_du = prec_mat_ud.transpose().tocsc()
            prec_mat_dd = sparse.coo_matrix(
                (data_dd, (rows_dd, columns_dd)), shape=(num_d, num_d)
            ).tocsc()

            return [prec_mat_uu, prec_mat_ud, prec_mat_du, prec_mat_dd]

    @staticmethod
    def _standard_normal_pdf(values: np.ndarray) -> np.ndarray:
        """Computes the probability distribution of the standard normal distribution.

        Args:
            values (np.ndarray): An array of values to calculate the probability
                distribution for.

        Returns:
            np.ndarray: The standard normal distribution probability distribution as an
                array.
        """
        return (2 * math.pi) ** (-0.5) * np.exp(-0.5 * values**2)

    def _record_simulations(
        self,
        run_state: _RunState,
        solution_indices: list[int],
        problem: Problem,
        mapping: DASSO._Mapping,
    ) -> None:
        """Record the simulation results.

        Args:
            run_state (_RunState): Run-scoped storage for design point data.
            solution_indices (list[int]): The solution indices to be simulated.
            problem (Problem): The problem instance providing bounds and function
                evaluations.
            mapping (DASSO._Mapping): The mapping instance providing mapping between
                solutions and coordinates.
        """
        additional_sample_means_vec_d = []
        additional_noise_cov_mat_diagonals = []
        sample_size = self.factors["sample_size"]
        for solution_index in solution_indices:
            if solution_index in run_state.design_point_indices:
                array_index = run_state.design_point_indices.index(solution_index)
                solution = run_state.design_points_actual[array_index]
                solution = self.evaluate(solution, problem, sample_size)
                run_state.sample_means_vec_d[array_index] = (
                    solution.objectives_mean[0] * problem.minmax[0] * (-1)
                )  # minimization problem
                run_state.noise_cov_mat_diagonals[array_index] = (
                    solution.objectives_var[0] / solution.n_reps
                )
            else:
                x_actual = mapping.get_actual_values_of_solution_from_index(solution_index)
                solution = self.evaluate(x_actual, problem, sample_size)
                additional_sample_means_vec_d += [
                    solution.objectives_mean[0] * problem.minmax[0] * (-1)
                ]  # minimization problem
                additional_noise_cov_mat_diagonals += [solution.objectives_var[0] / solution.n_reps]
                run_state.design_point_indices.append(solution_index)
                run_state.design_points_actual.append(solution)

        run_state.sample_means_vec_d = np.append(
            run_state.sample_means_vec_d, additional_sample_means_vec_d
        )
        run_state.noise_cov_mat_diagonals = np.append(
            run_state.noise_cov_mat_diagonals, additional_noise_cov_mat_diagonals
        )
