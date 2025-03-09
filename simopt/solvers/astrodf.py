"""
Summary
-------
The ASTRO-DF solver progressively builds local models (quadratic with diagonal Hessian) using interpolation on a set of points on the coordinate bases of the best (incumbent) solution. Solving the local models within a trust region (closed ball around the incumbent solution) at each iteration suggests a candidate solution for the next iteration. If the candidate solution is worse than the best interpolation point, it is replaced with the latter (a.k.a. direct search). The solver then decides whether to accept the candidate solution and expand the trust-region or reject it and shrink the trust-region based on a success ratio test. The sample size at each visited point is determined adaptively and based on closeness to optimality.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/astrodf.html>`__.

This version does not require a delta_max, instead it estimates the maximum step size using get_random_solution(). Parameter tuning on delta_max is therefore not needed and removed from this version as well.
- Delta_max is so longer a factor, instead the maximum step size is estimated using get_random_solution().
- Parameter tuning on delta_max is therefore not needed and removed from this version as well.
- No upper bound on sample size may be better - testing
- It seems for SAN we always use pattern search - why? because the problem is convex and model may be misleading at the beginning
- Added sufficient reduction for the pattern search
"""

from __future__ import annotations
from simopt.utils import classproperty

import logging
from math import ceil, log
from typing import Callable

import numpy as np
from numpy.linalg import inv, norm, pinv, LinAlgError
from scipy.optimize import NonlinearConstraint, minimize

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    VariableType,
)


class ASTRODF(Solver):
    """The ASTRO-DF solver.

    Attributes
    ----------
    name : string
        name of solver
    objective_type : string
        description of objective types:
            "single" or "multi"
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_needed : bool
        indicates if gradient of objective function is needed
    factors : dict
        changeable factors (i.e., parameters) of the solver
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used for the solver's internal purposes

    Arguments
    ---------
    name : str
        user-specified name for solver
    fixed_factors : dict
        fixed_factors of the solver
    See also
    --------
    base.Solver
    """

    @classproperty
    def class_name(cls) -> str:
        return "ASTRO-DF"

    @classproperty
    def objective_type(cls) -> ObjectiveType:
        return ObjectiveType.SINGLE

    @classproperty
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS

    @classproperty
    def gradient_needed(cls) -> bool:
        return False

    @classproperty
    def specifications(cls) -> dict[str, dict]:
        return {
            "crn_across_solns": {
                "description": "use CRN across solutions",
                "datatype": bool,
                "default": True,
            },
            "eta_1": {
                "description": "threshhold for a successful iteration",
                "datatype": float,
                "default": 0.1,
            },
            "eta_2": {
                "description": "threshhold for a very successful iteration",
                "datatype": float,
                "default": 0.8,
            },
            "gamma_1": {
                "description": "trust-region radius increase rate after a very successful iteration",
                "datatype": float,
                "default": 2.5,
            },
            "gamma_2": {
                "description": "trust-region radius decrease rate after an unsuccessful iteration",
                "datatype": float,
                "default": 0.5,
            },
            "lambda_min": {
                "description": "minimum sample size",
                "datatype": int,
                "default": 5,
            },
            "easy_solve": {
                "description": "solve the subproblem approximately with Cauchy point",
                "datatype": bool,
                "default": True,
            },
            "reuse_points": {
                "description": "reuse the previously visited points",
                "datatype": bool,
                "default": True,
            },
            "ps_sufficient_reduction": {
                "description": "use pattern search if with sufficient reduction, 0 always allows it, large value never does",
                "datatype": float,
                "default": 0.1,
            },
            "use_gradients": {
                "description": "if direct gradient observations are available, use them",
                "datatype": bool,
                "default": True,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,  # type: ignore
            "eta_1": self.check_eta_1,
            "eta_2": self.check_eta_2,
            "gamma_1": self.check_gamma_1,
            "gamma_2": self.check_gamma_2,
            "lambda_min": self.check_lambda_min,
            "ps_sufficient_reduction": self.check_ps_sufficient_reduction,
        }

    def __init__(
        self, name: str = "ASTRODF", fixed_factors: dict | None = None
    ) -> None:
        """
        Initialize the ASTRO-DF solver.
        Arguments
        ---------
        name : str
            user-specified name for solver
        fixed_factors : dict
            fixed_factors of the solver
        """
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

        # Initialize instance variables
        self.iteration_count = 0
        self.delta_k = None
        self.delta_max = None
        self.expended_budget = 0
        self.recommended_solns = []
        self.intermediate_budgets = []
        self.incumbent_x = None
        self.incumbent_solution = None
        self.interpolation_solns = []
        self.visited_pts_list = []
        self.intermediate_budgets = []
        self.kappa = 0
        self.h_k = None

    def check_eta_1(self) -> None:
        if self.factors["eta_1"] <= 0:
            raise ValueError("Eta 1 must be greater than 0.")

    def check_eta_2(self) -> None:
        if self.factors["eta_2"] <= self.factors["eta_1"]:
            raise ValueError("Eta 2 must be greater than Eta 1.")

    def check_gamma_1(self) -> None:
        if self.factors["gamma_1"] <= 1:
            raise ValueError("Gamma 1 must be greater than 1.")

    def check_gamma_2(self) -> None:
        if self.factors["gamma_2"] >= 1 or self.factors["gamma_2"] <= 0:
            raise ValueError("Gamma 2 must be between 0 and 1.")

    def check_lambda_min(self) -> None:
        if self.factors["lambda_min"] <= 2:
            raise ValueError("The minimum sample size must be greater than 2.")

    def check_ps_sufficient_reduction(self) -> None:
        if self.factors["ps_sufficient_reduction"] < 0:
            raise ValueError(
                "ps_sufficient reduction must be greater than or equal to 0."
            )

    def get_coordinate_vector(self, size: int, v_no: int) -> np.ndarray:
        """
        Generate the coordinate vector corresponding to the variable number v_no.
        """
        arr = np.zeros(size)
        arr[v_no] = 1.0
        return arr

    def get_rotated_basis(
        self, first_basis: np.ndarray, rotate_index: np.ndarray
    ) -> np.ndarray:
        """
        Generate the basis (rotated coordinate) (the first vector comes from the visited design points (origin basis))
        """
        rotate_matrix = np.array(first_basis)
        rotation = np.zeros((2, 2), dtype=int)
        rotation[0][1] = -1
        rotation[1][0] = 1

        # rotate the coordinate basis based on the first basis vector (first_basis)
        # choose two dimensions which we use for the rotation (0,i)

        for i in range(1, len(rotate_index)):
            v1 = np.array(
                [[first_basis[rotate_index[0]]], [first_basis[rotate_index[i]]]]
            )
            v2 = np.dot(rotation, v1)
            rotated_basis = np.copy(first_basis)
            rotated_basis[rotate_index[0]] = v2[0][0]
            rotated_basis[rotate_index[i]] = v2[1][0]
            # stack the rotated vector
            rotate_matrix = np.vstack((rotate_matrix, rotated_basis))

        return rotate_matrix

    def evaluate_model(self, x_k: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Compute the local model value with a linear interpolation with a diagonal Hessian
        """
        xk_arr = np.array(x_k).flatten()
        x_val = np.hstack(([1], xk_arr, xk_arr**2))
        result = np.matmul(x_val, q)
        return result

    def get_stopping_time(
        self,
        pilot_run: int,
        sig2: float,
        delta: float,
        kappa: float,
    ) -> int:
        """
        Compute the sample size based on adaptive sampling stopping rule using the optimality gap
        """
        if kappa == 0:
            kappa = 1
        # lambda_k = max(
        #     self.factors["lambda_min"], 2 * log(dim + 0.5, 10)
        # ) * max(log(k + 0.1, 10) ** (1.01), 1)

        # compute sample size
        raw_sample_size = pilot_run * max(
            1, sig2 / (kappa**2 * delta**self.delta_power)
        )
        # Convert out of ndarray if it is
        if isinstance(raw_sample_size, np.ndarray):
            raw_sample_size = raw_sample_size[0]
        # round up to the nearest integer
        sample_size: int = ceil(raw_sample_size)
        return sample_size

    def select_interpolation_points(
        self, delta_k: float, f_index: int
    ) -> tuple[list, list]:
        # If it is the first iteration or there is no design point we can reuse within the trust region, use the coordinate basis
        if (
            self.iteration_count == 1
            or (
                norm(
                    np.array(self.incumbent_x)
                    - np.array(self.visited_pts_list[f_index].x)
                )
                == 0
            )
            or not self.reuse_points
        ):
            # Construct the interpolation set
            var_y = self.get_coordinate_basis_interpolation_points(
                self.incumbent_x, delta_k, self.problem
            )
            var_z = self.get_coordinate_basis_interpolation_points(
                tuple(np.zeros(self.problem.dim)), delta_k, self.problem
            )
        # Else if we will reuse one design point (k > 1)
        elif self.iteration_count > 1:
            visited_pts_array = np.array(self.visited_pts_list[f_index].x)
            diff_array = visited_pts_array - np.array(self.incumbent_x)
            first_basis = (diff_array) / norm(diff_array)
            # if first_basis has some non-zero components, use rotated basis for those dimensions
            rotate_list = np.nonzero(first_basis)[0]
            rotate_matrix = self.get_rotated_basis(first_basis, rotate_list)

            # if first_basis has some zero components, use coordinate basis for those dimensions
            for i in range(self.problem.dim):
                if first_basis[i] == 0:
                    coord_vector = self.get_coordinate_vector(
                        self.problem.dim, i
                    )
                    rotate_matrix = np.vstack(
                        (
                            rotate_matrix,
                            coord_vector,
                        )
                    )

            # construct the interpolation set
            var_y = self.get_rotated_basis_interpolation_points(
                np.array(self.incumbent_x),
                delta_k,
                self.problem,
                rotate_matrix,
                self.visited_pts_list[f_index].x,
            )
            var_z = self.get_rotated_basis_interpolation_points(
                np.zeros(self.problem.dim),
                delta_k,
                self.problem,
                rotate_matrix,
                np.array(self.visited_pts_list[f_index].x)
                - np.array(self.incumbent_x),
            )
        # Else
        # TODO: figure out what to do if the above conditions are not met
        else:
            var_y = []
            var_z = []

        return var_y, var_z

    def construct_model(
        self,
    ) -> tuple[
        list[float],
        list,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[Solution],
    ]:
        """
        construct the "qualified" local model for each iteration k with the center point x_k
        reconstruct with new points in a shrunk trust-region if the model fails the criticality condition
        the criticality condition keeps the model gradient norm and the trust-region size in lock-step
        """
        interpolation_solns = []

        ## inner loop parameters
        w = 0.85  # self.factors["w"]
        mu = 1000  # self.factors["mu"]
        beta = 10  # self.factors["beta"]
        # criticality_threshold = 0.1  # self.factors["criticality_threshold"]
        # skip_criticality = True  # self.factors["skip_criticality"]
        # Problem and solver factors

        lambda_max = self.budget - self.expended_budget
        # lambda_max = budget / (15 * sqrt(problem.dim))
        pilot_run = ceil(
            max(
                self.lambda_min * log(10 + self.iteration_count, 10) ** 1.1,
                min(0.5 * self.problem.dim, lambda_max),
            )
            - 1
        )

        delta = self.delta_k
        model_iterations: int = 0
        while True:
            delta_k = delta * w**model_iterations
            model_iterations += 1

            # Calculate the distance between the center point and other design points
            distance_array = []
            for point in self.visited_pts_list:
                dist_diff = np.array(point.x) - np.array(self.incumbent_x)
                distance = norm(dist_diff) - delta_k
                # If the design point is outside the trust region, we will not reuse it (distance = -big M)
                dist_to_append = -delta_k * 10000 if distance > 0 else distance
                distance_array.append(dist_to_append)

            # Find the index of visited design points list for reusing points
            # The reused point will be the farthest point from the center point among the design points within the trust region
            f_index = distance_array.index(max(distance_array))

            var_y, var_z = self.select_interpolation_points(delta_k, f_index)

            # Evaluate the function estimate for the interpolation points
            fval = []
            double_dim = 2 * self.problem.dim + 1
            for i in range(double_dim):
                # Special cases for the first two iterations
                is_first_dp_iteration = i == 0
                is_second_dp_iteration = i == 1
                if is_first_dp_iteration:
                    # for anthing other than x_0, we need to simulate the new solution
                    if self.iteration_count > 1:
                        # reuse the replications for x_k (center point, i.e., the incumbent solution)
                        sample_size = self.incumbent_solution.n_reps
                        sig2 = self.incumbent_solution.objectives_var[0]
                        # adaptive sampling
                        while True:
                            stopping = self.get_stopping_time(
                                pilot_run,
                                sig2,
                                delta_k,
                                self.kappa,
                            )
                            if (
                                sample_size >= min(stopping, lambda_max)
                                or self.expended_budget >= self.budget
                            ):
                                break
                            self.problem.simulate(self.incumbent_solution, 1)
                            self.expended_budget += 1
                            sample_size += 1
                            sig2 = self.incumbent_solution.objectives_var[0]
                    fval.append(
                        -1
                        * self.problem.minmax[0]
                        * self.incumbent_solution.objectives_mean
                    )
                    interpolation_solns.append(self.incumbent_solution)
                # else if reuse one design point, reuse the replications
                elif (
                    is_second_dp_iteration
                    and self.reuse_points
                    and norm(
                        np.array(self.incumbent_x)
                        - np.array(self.visited_pts_list[f_index].x)
                    )
                    != 0
                ):
                    sample_size = self.visited_pts_list[f_index].n_reps
                    sig2 = self.visited_pts_list[f_index].objectives_var[0]
                    # adaptive sampling
                    while True:
                        stopping = self.get_stopping_time(
                            pilot_run,
                            sig2,
                            delta_k,
                            self.kappa,
                        )
                        if (
                            sample_size >= min(stopping, lambda_max)
                            or self.expended_budget >= self.budget
                        ):
                            break
                        self.problem.simulate(self.visited_pts_list[f_index], 1)
                        self.expended_budget += 1
                        sample_size += 1
                        sig2 = self.visited_pts_list[f_index].objectives_var[0]
                    fval.append(
                        -1
                        * self.problem.minmax[0]
                        * self.visited_pts_list[f_index].objectives_mean
                    )
                    interpolation_solns.append(self.visited_pts_list[f_index])
                # for new points, run the simulation with pilot run
                else:
                    decision_vars = tuple(var_y[i][0])
                    new_solution = self.create_new_solution(
                        decision_vars, self.problem
                    )
                    self.visited_pts_list.append(new_solution)
                    self.problem.simulate(new_solution, pilot_run)
                    self.expended_budget += pilot_run
                    sample_size = pilot_run

                    # adaptive sampling
                    while True:
                        sig2 = new_solution.objectives_var[0]
                        stopping = self.get_stopping_time(
                            pilot_run,
                            sig2,
                            delta_k,
                            self.kappa,
                        )
                        if (
                            sample_size >= min(stopping, lambda_max)
                            or self.expended_budget >= self.budget
                        ):
                            break
                        self.problem.simulate(new_solution, 1)
                        self.expended_budget += 1
                        sample_size += 1

                    fval.append(
                        -1
                        * self.problem.minmax[0]
                        * new_solution.objectives_mean
                    )
                    interpolation_solns.append(new_solution)

            # construct the model and obtain the model coefficients
            q, grad, hessian = self.get_model_coefficients(
                var_z, fval, self.problem
            )

            if delta_k <= mu * norm(grad):
                break

            # If a model gradient norm is zero, there is a possibility that the code stuck in this while loop
            if norm(grad) == 0:
                break
            # TODO: investigate if this can be implemented instead of checking norm(grad) == 0
            # MAX_ITER = 100
            # if model_iterations > MAX_ITER:
            #     break

        beta_n_grad = float(beta * norm(grad))
        self.delta_k = min(max(beta_n_grad, delta_k), delta)

        return (
            fval,
            var_y,
            q,
            grad,
            hessian,
            interpolation_solns,
        )

    def get_model_coefficients(
        self, y_var: list, fval: list, problem: Problem
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the model coefficients using (2d+1) design points and their function estimates."""
        num_design_points = 2 * problem.dim + 1

        # Construct the matrix with ones, linear terms, and squared terms
        m_var = np.array(
            [
                np.hstack(([1], np.ravel(y_var[i]), np.ravel(y_var[i]) ** 2))
                for i in range(num_design_points)
            ]
        )

        # Compute the inverse or pseudoinverse of the matrix
        try:
            matrix_inverse = inv(m_var)
        except LinAlgError:
            matrix_inverse = pinv(m_var)

        inverse_mult = np.matmul(matrix_inverse, fval)

        # Extract gradient and Hessian from the result
        decision_var_idx = problem.dim + 1
        grad = inverse_mult[1:decision_var_idx].reshape(problem.dim)
        hessian = inverse_mult[decision_var_idx:num_design_points].reshape(
            problem.dim
        )

        return inverse_mult, grad, hessian

    def get_coordinate_basis_interpolation_points(
        self, x_k: tuple[int | float, ...], delta: float, problem: Problem
    ) -> list:
        """
        Compute the interpolation points (2d+1) using the coordinate basis.
        """
        y_var = [[x_k]]
        is_block_constraint = sum(x_k) != 0
        num_decision_vars = problem.dim

        lower_bounds = problem.lower_bounds
        upper_bounds = problem.upper_bounds

        for var_idx in range(num_decision_vars):
            coord_vector = self.get_coordinate_vector(
                num_decision_vars, var_idx
            )
            coord_diff = delta * coord_vector

            minus = [x - d for x, d in zip(x_k, coord_diff)]
            plus = [x + d for x, d in zip(x_k, coord_diff)]

            if is_block_constraint:
                minus = [
                    clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                    for j, val in enumerate(minus)
                ]
                plus = [
                    clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                    for j, val in enumerate(plus)
                ]

            y_var.append([plus])
            y_var.append([minus])

        return y_var

    def get_rotated_basis_interpolation_points(
        self,
        x_k: np.ndarray,
        delta: float,
        problem: Problem,
        rotate_matrix: np.ndarray,
        reused_x: np.ndarray,
    ) -> list[list[np.ndarray]]:
        """
        Compute the interpolation points (2d+1) using the rotated coordinate basis (reuse one design point)
        """
        y_var = [[x_k]]
        is_block_constraint = np.sum(x_k) != 0
        num_decision_vars = problem.dim

        lower_bounds = np.array(problem.lower_bounds)
        upper_bounds = np.array(problem.upper_bounds)

        for i in range(num_decision_vars):
            rotate_matrix_delta = delta * rotate_matrix[i]

            if i == 0:
                plus = np.array(reused_x)
            else:
                plus = x_k + rotate_matrix_delta

            minus = x_k - rotate_matrix_delta

            if is_block_constraint:
                minus = [
                    clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                    for j, val in enumerate(minus)
                ]
                plus = [
                    clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                    for j, val in enumerate(plus)
                ]

            y_var.append([plus])
            y_var.append([minus])

        return y_var

    def iterate(self) -> None:
        """
        Run one iteration of trust-region algorithm by bulding and solving a local model and updating the current incumbent and trust-region radius, and saving the data
        """
        self.iteration_count += 1

        # determine power of delta in adaptive sampling rule
        pilot_run = ceil(
            max(
                self.lambda_min * log(10 + self.iteration_count, 10) ** 1.1,
                min(0.5 * self.problem.dim, self.lambda_max),
            )
            - 1
        )
        if self.iteration_count == 1:
            self.incumbent_solution = self.create_new_solution(
                self.incumbent_x, self.problem
            )
            if len(self.visited_pts_list) == 0:
                self.visited_pts_list.append(self.incumbent_solution)

            # pilot run
            self.problem.simulate(self.incumbent_solution, pilot_run)
            self.expended_budget += pilot_run
            sample_size = pilot_run

            # adaptive sampling
            while True:
                if self.enable_gradient:
                    rhs_for_kappa = norm(
                        self.incumbent_solution.objectives_gradients_mean[0]
                    )
                else:
                    rhs_for_kappa = self.incumbent_solution.objectives_mean
                sig2 = self.incumbent_solution.objectives_var[0]
                if self.delta_power == 0:
                    sol_trace = np.trace(
                        self.incumbent_solution.objectives_gradients_var
                    )
                    sig2 = max(sig2, sol_trace)
                kappa = (
                    rhs_for_kappa
                    * np.sqrt(pilot_run)
                    / (self.delta_k ** (self.delta_power / 2))
                )
                stopping = self.get_stopping_time(
                    pilot_run,
                    sig2,
                    self.delta_k,
                    kappa,
                )
                if (
                    sample_size >= min(stopping, self.lambda_max)
                    or self.expended_budget >= self.budget
                ):
                    # calculate kappa
                    self.kappa = (
                        rhs_for_kappa
                        * np.sqrt(pilot_run)
                        / (self.delta_k ** (self.delta_power / 2))
                    )
                    # logging.debug("kappa "+str(kappa))
                    break
                self.problem.simulate(self.incumbent_solution, 1)
                self.expended_budget += 1
                sample_size += 1

            self.recommended_solns.append(self.incumbent_solution)
            self.intermediate_budgets.append(self.expended_budget)
        # since incument was only evaluated with the sample size of previous incumbent, here we compute its adaptive sample size
        elif self.factors["crn_across_solns"]:
            sample_size = self.incumbent_solution.n_reps
            # adaptive sampling
            while True:
                sig2 = self.incumbent_solution.objectives_var[0]
                if self.delta_power == 0:
                    sig2 = max(
                        sig2,
                        np.trace(
                            self.incumbent_solution.objectives_gradients_var
                        ),
                    )
                stopping = self.get_stopping_time(
                    pilot_run,
                    sig2,
                    self.delta_k,
                    self.kappa,
                )
                if (
                    sample_size >= min(stopping, self.lambda_max)
                    or self.expended_budget >= self.budget
                ):
                    break
                else:
                    self.problem.simulate(self.incumbent_solution, 1)
                    self.expended_budget += 1
                    sample_size += 1

        # use Taylor expansion if gradient available
        if self.enable_gradient:
            fval = (
                np.ones(2 * self.problem.dim + 1)
                * -1
                * self.problem.minmax[0]
                * self.incumbent_solution.objectives_mean
            )
            grad = (
                -1
                * self.problem.minmax[0]
                * self.incumbent_solution.objectives_gradients_mean[0]
            )
            hessian = self.h_k
        else:
            # build the local model with interpolation (subproblem)
            (
                fval,
                y_var,
                q,
                grad,
                hessian,
                interpolation_solns,
            ) = self.construct_model()

        # solve the local model (subproblem)
        if self.easy_solve:
            # Cauchy reduction
            # TODO: why do we need this? Check model reduction calculation too.
            # logging.debug("np.dot(np.multiply(grad, Hessian), grad) "+str(np.dot(np.multiply(grad, hessian), grad)))
            # logging.debug("np.dot(np.dot(grad, hessian), grad) "+str(np.dot(np.dot(grad, hessian), grad)))
            if self.enable_gradient:
                dot_a = np.dot(grad, hessian)
            else:
                dot_a = grad * hessian

            check_positive_definite: float = np.dot(dot_a, grad)

            if check_positive_definite <= 0:
                tau = 1.0
            else:
                norm_ratio = norm(grad) ** 3 / (
                    self.delta_k * check_positive_definite
                )
                tau = min(1.0, norm_ratio)

            grad: np.ndarray = np.reshape(grad, (1, self.problem.dim))[0]
            grad_norm = norm(grad)
            # Make sure we don't divide by 0
            if grad_norm == 0:
                candidate_x = self.incumbent_x
            else:
                product = tau * self.delta_k * grad
                adjustment = product / grad_norm
                candidate_x = self.incumbent_x - adjustment
            # if norm(incumbent_x - candidate_x) > 0:
            #     logging.debug("incumbent_x " + str(incumbent_x))
            #     logging.debug("candidate_x " + str(candidate_x))

        else:
            # Search engine - solve subproblem
            def subproblem(s: np.ndarray) -> float:
                s_grad_dot: np.ndarray = np.dot(s, grad)
                s_hessian_dot: np.ndarray = np.dot(np.multiply(s, hessian), s)
                result = fval[0] + s_grad_dot + s_hessian_dot
                return float(result[0])

            def con_f(s: np.ndarray) -> float:
                return float(norm(s))

            nlc = NonlinearConstraint(con_f, 0, self.delta_k)
            solve_subproblem = minimize(
                subproblem,
                np.zeros(self.problem.dim),
                method="trust-constr",
                constraints=nlc,
            )
            candidate_x = self.incumbent_x + solve_subproblem.x

        # logging.debug("problem.lower_bounds "+str(problem.lower_bounds))
        # handle the box constraints
        new_candidate_list = []
        for i in range(self.problem.dim):
            candidate = float(candidate_x[i])
            # Correct candidate if it violates the box constraints
            lower_bound = self.problem.lower_bounds[i]
            upper_bound = self.problem.upper_bounds[i]
            epsilon = 0.01
            if candidate <= lower_bound:
                candidate = lower_bound + epsilon
            elif candidate >= upper_bound:
                candidate = upper_bound - epsilon
            # Append the corrected candidate to the new candidate list
            new_candidate_list.append(candidate)
        candidate_x = tuple(new_candidate_list)

        # store the solution (and function estimate at it) to the subproblem as a candidate for the next iterate
        candidate_solution = self.create_new_solution(candidate_x, self.problem)
        self.visited_pts_list.append(candidate_solution)

        # if we use crn, then the candidate solution has the same sample size as the incumbent solution
        if self.factors["crn_across_solns"]:
            num_sims = self.incumbent_solution.n_reps
            self.problem.simulate(candidate_solution, num_sims)
            self.expended_budget += num_sims
        else:
            # pilot run and adaptive sampling
            self.problem.simulate(candidate_solution, pilot_run)
            self.expended_budget += pilot_run
            sample_size = pilot_run
            while True:
                # if enable_gradient:
                #     # logging.debug("incumbent_solution.objectives_gradients_var[0] "+str(candidate_solution.objectives_gradients_var[0]))
                #     while norm(candidate_solution.objectives_gradients_var[0]) == 0 and candidate_solution.n_reps < max(pilot_run, lambda_max/100):
                #         problem.simulate(candidate_solution, 1)
                #         expended_budget += 1
                #         sample_size += 1
                sig2 = candidate_solution.objectives_var[0]
                if self.delta_power == 0:
                    sig2 = max(
                        sig2,
                        np.trace(candidate_solution.objectives_gradients_var),
                    )
                stopping = self.get_stopping_time(
                    pilot_run,
                    sig2,
                    self.delta_k,
                    self.kappa,
                )
                if (
                    sample_size >= min(stopping, self.lambda_max)
                    or self.expended_budget >= self.budget
                ):
                    break
                else:
                    self.problem.simulate(candidate_solution, 1)
                    self.expended_budget += 1
                    sample_size += 1

        # TODO: make sure the solution whose estimated objevtive is abrupted bc of budget is not added to the list of recommended solutions, unless the error is negligible ...
        # if (expended_budget >= budget_limit) and (sample_size < stopping):
        #     final_ob = fval[0]
        # else:
        # calculate success ratio
        fval_tilde = (
            -1 * self.problem.minmax[0] * candidate_solution.objectives_mean
        )
        # replace the candidate x if the interpolation set has lower objective function value and with sufficient reduction (pattern search)
        # also if the candidate solution's variance is high that could be caused by stopping early due to exhausting budget
        # logging.debug("cv "+str(candidate_solution.objectives_var/(candidate_solution.n_reps * candidate_solution.objectives_mean ** 2)))
        # logging.debug("fval[0] - min(fval) "+str(fval[0] - min(fval)))

        if not self.enable_gradient:
            min_fval = min(fval)
            sufficient_reduction = (fval[0] - min_fval) >= self.factors[
                "ps_sufficient_reduction"
            ] * self.delta_k**2

            condition_met = min_fval < fval_tilde and sufficient_reduction

            high_variance = False
            if not condition_met:
                # Treat variance as low if mean is zero to avoid division by
                # zero (zero mean typically indicates negligible uncertainty)
                if candidate_solution.objectives_mean[0] == 0:
                    logging.debug(
                        "Candidate solution objectives_mean is zero, skipping variance check."
                    )
                else:
                    high_variance = (
                        candidate_solution.objectives_var[0]
                        / (
                            candidate_solution.n_reps
                            * candidate_solution.objectives_mean[0] ** 2
                        )
                    ) > 0.75

            if condition_met or high_variance:
                fval_tilde = min_fval
                min_idx = fval.index(min_fval)
                candidate_x = y_var[min_idx][0]
                candidate_solution = interpolation_solns[min_idx]

        # compute the success ratio rho
        candidate_x_arr = np.array(candidate_x)
        incumbent_x_arr = np.array(self.incumbent_x)
        s = np.subtract(candidate_x_arr, incumbent_x_arr)
        if self.enable_gradient:
            model_reduction = -np.dot(s, grad) - 0.5 * np.dot(
                np.dot(s, hessian), s
            )
        else:
            model_reduction = self.evaluate_model(
                np.zeros(self.problem.dim),
                q,  # type: ignore
            ) - self.evaluate_model(s, q)  # type: ignore
        if model_reduction <= 0:
            rho = 0
        else:
            rho = (fval[0] - fval_tilde) / model_reduction

        epsilon = 1e-15
        successful = rho >= self.eta_1
        # successful: accept
        if successful:
            self.incumbent_x = candidate_x
            self.incumbent_solution = candidate_solution
            self.recommended_solns.append(candidate_solution)
            self.intermediate_budgets.append(self.expended_budget)
            self.delta_k = min(self.delta_k, self.delta_max)

            # very successful: expand
            if rho >= self.eta_2:
                self.delta_k = min(self.gamma_1 * self.delta_k, self.delta_max)

            if self.enable_gradient:
                candidate_grad = (
                    -1
                    * self.problem.minmax[0]
                    * candidate_solution.objectives_gradients_mean[0]
                )
                y_k = candidate_grad - grad
                y_ks = y_k @ s

                if np.isclose(y_ks, 0, atol=epsilon):
                    warning_msg = "y_ks near zero; skipping Hessian update to avoid numerical instability."
                    logging.warning(warning_msg)
                    r_k = 0
                else:
                    r_k = 1.0 / y_ks

                h_s_k = self.h_k @ s
                s_h_s_k = s @ h_s_k

                if np.isclose(s_h_s_k, 0, atol=epsilon):
                    warning_msg = "s_h_s_k near zero; skipping Hessian update to avoid numerical instability."
                    logging.warning(warning_msg)
                else:
                    self.h_k += (
                        np.outer(y_k, y_k) * r_k
                        - np.outer(h_s_k, h_s_k) / s_h_s_k
                    )

        elif not successful:
            self.delta_k = min(self.gamma_2 * self.delta_k, self.delta_max)

        # TODO: unified TR management
        # delta_k = min(kappa * norm(grad), self.delta_max)
        # logging.debug("norm of grad "+str(norm(grad)))

    @property
    def remaining_budget(self) -> int:
        """Return the remaining budget."""
        return self.budget - self.expended_budget

    def _initialize_solving(self) -> None:
        """Setup the solver for the first iteration."""
        self.budget: int = self.problem.factors["budget"]
        self.eta_1: float = self.factors["eta_1"]
        self.eta_2: float = self.factors["eta_2"]
        self.gamma_1: float = self.factors["gamma_1"]
        self.gamma_2: float = self.factors["gamma_2"]
        self.easy_solve: bool = self.factors["easy_solve"]
        self.reuse_points: bool = self.factors["reuse_points"]
        self.lambda_min: int = self.factors["lambda_min"]
        self.lambda_max = self.budget

        # Designate random number generator for random sampling
        rng = self.rng_list[1]

        # Generate dummy solutions to estimate a reasonable maximum radius
        dummy_solns = [
            self.problem.get_random_solution(rng)
            for _ in range(1000 * self.problem.dim)
        ]

        # Range for each dimension is calculated and compared with box constraints range if given
        # TODO: just use box constraints range if given
        # self.delta_max = min(self.factors["delta_max"], problem.upper_bounds[0] - problem.lower_bounds[0])
        delta_max_candidates: list[float | int] = []
        for i in range(self.problem.dim):
            sol_values = [sol[i] for sol in dummy_solns]
            min_soln, max_soln = min(sol_values), max(sol_values)
            bound_range = (
                self.problem.upper_bounds[i] - self.problem.lower_bounds[i]
            )
            delta_max_candidates.append(min(max_soln - min_soln, bound_range))

        # TODO: update this so that it could be used for problems with decision variables at varying scales!
        self.delta_max = max(delta_max_candidates)
        # logging.debug("delta_max  " + str(self.delta_max))

        # Initialize trust-region radius
        self.delta_k = 10 ** (
            ceil(log(self.delta_max * 2, 10) - 1) / self.problem.dim
        )
        # logging.debug("initial delta " + str(self.delta_k))

        if "initial_solution" in self.problem.factors:
            self.incumbent_x = tuple(self.problem.factors["initial_solution"])
        else:
            self.incumbent_x = tuple(self.problem.get_random_solution(rng))

        self.incumbent_solution = self.create_new_solution(
            self.incumbent_x, self.problem
        )
        self.h_k = np.identity(self.problem.dim).tolist()

        self.enable_gradient = (
            self.problem.gradient_available and self.factors["use_gradients"]
        )

        if self.factors["crn_across_solns"]:
            self.delta_power = 0 if self.enable_gradient else 2
        else:
            self.delta_power = 4

        # Reset iteration count and data storage
        self.iteration_count = 0
        self.expended_budget = 0
        self.recommended_solns = []
        self.intermediate_budgets = []
        self.visited_pts_list = []

    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """
        Run a single macroreplication of a solver on a problem.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve

        Returns
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes
        """
        self.problem = problem
        self._initialize_solving()

        while self.expended_budget < self.budget:
            self.iterate()

        return self.recommended_solns, self.intermediate_budgets


def clamp_with_epsilon(
    val: float, lower_bound: float, upper_bound: float, epsilon: float = 0.01
) -> float:
    """Apply epsilon adjustment to keep values within bounds while avoiding exact boundary values."""
    if val <= lower_bound:
        return lower_bound + epsilon
    elif val >= upper_bound:
        return upper_bound - epsilon
    return val
