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

import sys
from math import ceil, log
from typing import Callable

import numpy as np
from numpy.linalg import norm, pinv
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

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.SINGLE

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BOX

    @property
    def variable_type(self) -> VariableType:
        return VariableType.CONTINUOUS

    @property
    def gradient_needed(self) -> bool:
        return False

    @property
    def specifications(self) -> dict[str, dict]:
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
        Generate the basis (rotated coordinate) (the first vector comes from the visited design points (origin basis)
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
        x_val = [1]
        x_val = np.append(x_val, np.array(x_k))
        x_val = np.append(x_val, np.array(x_k) ** 2)
        return np.matmul(x_val, q)

    def get_stopping_time(
        self,
        pilot_run: int,
        sig2: float,
        delta: float,
        kappa: float,
        dim: int,
        delta_power: int,
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
            1, sig2 / (kappa**2 * delta**delta_power)
        )
        # Convert out of ndarray if it is
        if isinstance(raw_sample_size, np.ndarray):
            raw_sample_size = raw_sample_size[0]
        # round up to the nearest integer
        sample_size: int = ceil(raw_sample_size)
        return sample_size

    def construct_model(
        self,
        x_k: tuple[int | float, ...],
        delta: float,
        k: int,
        problem: Problem,
        expended_budget: int,
        kappa: float,
        incumbent_solution: Solution,
        visited_pts_list: list,
        delta_power: int,
    ) -> tuple[
        list[float],
        list,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        int,
        list[Solution],
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
        j = 0
        # Problem and solver factors
        reuse_points: bool = self.factors["reuse_points"]
        lambda_min: int = self.factors["lambda_min"]
        budget: int = problem.factors["budget"]

        lambda_max = budget - expended_budget
        # lambda_max = budget / (15 * sqrt(problem.dim))
        pilot_run = ceil(
            max(
                lambda_min * log(10 + k, 10) ** 1.1,
                min(0.5 * problem.dim, lambda_max),
            )
            - 1
        )
        while True:
            fval = []
            j = j + 1
            delta_k = delta * w ** (j - 1)

            # Calculate the distance between the center point and other design points
            distance_array = []
            for i in range(len(visited_pts_list)):
                distance_array.append(
                    norm(np.array(visited_pts_list[i].x) - np.array(x_k))
                    - delta_k
                )
                # If the design point is outside the trust region, we will not reuse it (distance = -big M)
                if distance_array[i] > 0:
                    distance_array[i] = -delta_k * 10000

            # Find the index of visited design points list for reusing points
            # The reused point will be the farthest point from the center point among the design points within the trust region
            f_index = distance_array.index(max(distance_array))

            # If it is the first iteration or there is no design point we can reuse within the trust region, use the coordinate basis

            if (
                (k == 1)
                or (
                    norm(np.array(x_k) - np.array(visited_pts_list[f_index].x))
                    == 0
                )
                or not reuse_points
            ):
                # Construct the interpolation set
                var_y = self.get_coordinate_basis_interpolation_points(
                    x_k, delta_k, problem
                )
                var_z = self.get_coordinate_basis_interpolation_points(
                    tuple(np.zeros(problem.dim)), delta_k, problem
                )
            # Else if we will reuse one design point
            elif k > 1:
                visited_pts_array = np.array(visited_pts_list[f_index].x)
                diff_array = visited_pts_array - np.array(x_k)
                first_basis = (diff_array) / norm(diff_array)
                # if first_basis has some non-zero components, use rotated basis for those dimensions
                rotate_list = np.nonzero(first_basis)[0]
                rotate_matrix = self.get_rotated_basis(first_basis, rotate_list)

                # if first_basis has some zero components, use coordinate basis for those dimensions
                for i in range(problem.dim):
                    if first_basis[i] == 0:
                        coord_vector = self.get_coordinate_vector(
                            problem.dim, i
                        )
                        rotate_matrix = np.vstack(
                            (
                                rotate_matrix,
                                coord_vector,
                            )
                        )

                # construct the interpolation set
                var_y = self.get_rotated_basis_interpolation_points(
                    np.array(x_k),
                    delta_k,
                    problem,
                    rotate_matrix,
                    visited_pts_list[f_index].x,
                )
                var_z = self.get_rotated_basis_interpolation_points(
                    np.zeros(problem.dim),
                    delta_k,
                    problem,
                    rotate_matrix,
                    np.array(visited_pts_list[f_index].x) - np.array(x_k),
                )
            # Else
            # TODO: figure out what to do if the above conditions are not met
            else:
                error_msg = "Error in constructing the interpolation set"
                raise ValueError(error_msg)
            # Evaluate the function estimate for the interpolation points
            for i in range(2 * problem.dim + 1):
                # for x_0, we don't need to simulate the new solution
                if (k == 1) and (i == 0):
                    fval.append(
                        -1
                        * problem.minmax[0]
                        * incumbent_solution.objectives_mean
                    )
                    interpolation_solns.append(incumbent_solution)
                # reuse the replications for x_k (center point, i.e., the incumbent solution)
                elif i == 0:
                    sample_size = incumbent_solution.n_reps
                    sig2 = incumbent_solution.objectives_var[0]
                    # adaptive sampling
                    while True:
                        stopping = self.get_stopping_time(
                            pilot_run,
                            sig2,
                            delta_k,
                            kappa,
                            problem.dim,
                            delta_power,
                        )
                        if (
                            sample_size >= min(stopping, lambda_max)
                            or expended_budget >= budget
                        ):
                            break
                        problem.simulate(incumbent_solution, 1)
                        expended_budget += 1
                        sample_size += 1
                        sig2 = incumbent_solution.objectives_var[0]
                    fval.append(
                        -1
                        * problem.minmax[0]
                        * incumbent_solution.objectives_mean
                    )
                    interpolation_solns.append(incumbent_solution)
                # else if reuse one design point, reuse the replications
                elif (
                    (i == 1)
                    and (
                        norm(
                            np.array(x_k)
                            - np.array(visited_pts_list[f_index].x)
                        )
                        != 0
                    )
                    and reuse_points
                ):
                    sample_size = visited_pts_list[f_index].n_reps
                    sig2 = visited_pts_list[f_index].objectives_var[0]
                    # adaptive sampling
                    while True:
                        stopping = self.get_stopping_time(
                            pilot_run,
                            sig2,
                            delta_k,
                            kappa,
                            problem.dim,
                            delta_power,
                        )
                        if (
                            sample_size >= min(stopping, lambda_max)
                            or expended_budget >= budget
                        ):
                            break
                        problem.simulate(visited_pts_list[f_index], 1)
                        expended_budget += 1
                        sample_size += 1
                        sig2 = visited_pts_list[f_index].objectives_var[0]
                    fval.append(
                        -1
                        * problem.minmax[0]
                        * visited_pts_list[f_index].objectives_mean
                    )
                    interpolation_solns.append(visited_pts_list[f_index])
                # for new points, run the simulation with pilot run
                else:
                    decision_vars = tuple(var_y[i][0])
                    new_solution = self.create_new_solution(
                        decision_vars, problem
                    )
                    visited_pts_list.append(new_solution)
                    problem.simulate(new_solution, pilot_run)
                    expended_budget += pilot_run
                    sample_size = pilot_run

                    # adaptive sampling
                    while True:
                        problem.simulate(new_solution, 1)
                        expended_budget += 1
                        sample_size += 1
                        sig2 = new_solution.objectives_var[0]
                        stopping = self.get_stopping_time(
                            pilot_run,
                            sig2,
                            delta_k,
                            kappa,
                            problem.dim,
                            delta_power,
                        )
                        if (
                            sample_size >= min(stopping, lambda_max)
                            or expended_budget >= budget
                        ):
                            break
                    fval.append(
                        -1 * problem.minmax[0] * new_solution.objectives_mean
                    )
                    interpolation_solns.append(new_solution)

            # construct the model and obtain the model coefficients
            q, grad, hessian = self.get_model_coefficients(var_z, fval, problem)

            if delta_k <= mu * norm(grad):
                break

            # If a model gradient norm is zero, there is a possibility that the code stuck in this while loop
            if norm(grad) == 0:
                break

        beta_n_grad = float(beta * norm(grad))
        delta_k = min(max(beta_n_grad, delta_k), delta)

        return (
            fval,
            var_y,
            q,
            grad,
            hessian,
            delta_k,
            expended_budget,
            interpolation_solns,
            visited_pts_list,
        )

    def get_model_coefficients(
        self, y_var: list, fval: list, problem: Problem
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the model coefficients using (2d+1) design points and their function estimates
        """
        m_var = []
        for i in range(0, 2 * problem.dim + 1):
            m_var.append(1)
            m_var[i] = np.append(m_var[i], np.array(y_var[i]))
            m_var[i] = np.append(m_var[i], np.array(y_var[i]) ** 2)

        q: np.ndarray = np.matmul(
            pinv(m_var), fval
        )  # pinv returns the inverse of your matrix when it is available and the pseudo inverse when it isn't.
        grad = q[1 : problem.dim + 1]
        grad = np.reshape(grad, problem.dim)
        hessian = q[problem.dim + 1 : 2 * problem.dim + 1]
        hessian = np.reshape(hessian, problem.dim)
        return q, grad, hessian

    def get_coordinate_basis_interpolation_points(
        self, x_k: tuple[int | float, ...], delta: float, problem: Problem
    ) -> list:
        """
        Compute the interpolation points (2d+1) using the coordinate basis
        """
        y_var = [[x_k]]
        epsilon = 0.01
        num_decision_vars = problem.dim
        is_block_constraint = sum(x_k) != 0

        for var_idx in range(num_decision_vars):
            coord_vector = self.get_coordinate_vector(
                num_decision_vars, var_idx
            )
            coord_diff = delta * coord_vector
            minus = y_var[0] - coord_diff
            plus = y_var[0] + coord_diff

            if is_block_constraint:
                # block constraints
                if minus[0][var_idx] <= problem.lower_bounds[var_idx]:
                    minus[0][var_idx] = problem.lower_bounds[var_idx] + epsilon
                if plus[0][var_idx] >= problem.upper_bounds[var_idx]:
                    plus[0][var_idx] = problem.upper_bounds[var_idx] - epsilon

            y_var.append(list(plus))
            y_var.append(list(minus))
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
        epsilon = 0.01
        is_block_constraint = sum(x_k) != 0
        num_decision_vars = problem.dim

        for i in range(num_decision_vars):
            rotate_matrix_delta: np.ndarray = delta * rotate_matrix[i]

            if i == 0:
                plus = tuple([np.array(reused_x)])
            else:
                plus = y_var[0] + rotate_matrix_delta
            minus = y_var[0] - rotate_matrix_delta

            if is_block_constraint:
                # block constraints
                for j in range(num_decision_vars):
                    lower_bound = problem.lower_bounds[j]
                    upper_bound = problem.upper_bounds[j]
                    if minus[0][j] <= lower_bound:
                        minus[0][j] = lower_bound + epsilon
                    elif minus[0][j] >= upper_bound:
                        minus[0][j] = upper_bound - epsilon
                    if plus[0][j] <= lower_bound:
                        plus[0][j] = lower_bound + epsilon
                    elif plus[0][j] >= upper_bound:
                        plus[0][j] = upper_bound - epsilon

            y_var.append(list(plus))
            y_var.append(list(minus))
        return y_var

    def iterate(
        self,
        k: int,
        delta_k: float,
        delta_max: float | int,
        problem: Problem,
        visited_pts_list: list,
        incumbent_x: tuple[int | float, ...],
        expended_budget: int,
        budget_limit: int,
        recommended_solns: list,
        intermediate_budgets: list,
        kappa: float,
        incumbent_solution: Solution,
        h_k: list,
    ) -> tuple[
        float,
        float,
        list[Solution],
        list[int],
        int,
        tuple[int | float, ...],
        float,
        Solution,
        list[Solution],
        list[list],
    ]:
        """
        Run one iteration of trust-region algorithm by bulding and solving a local model and updating the current incumbent and trust-region radius, and saving the data
        """
        # default values
        eta_1: float = self.factors["eta_1"]
        eta_2: float = self.factors["eta_2"]
        gamma_1: float = self.factors["gamma_1"]
        gamma_2: float = self.factors["gamma_2"]
        easy_solve: bool = self.factors["easy_solve"]
        lambda_min: int = self.factors["lambda_min"]
        lambda_max = budget_limit - expended_budget
        # lambda_max = budget_limit / (15 * sqrt(problem.dim))
        gradient_availability = problem.gradient_available
        # uncomment the next line to avoid Hessian updating
        # h_k = np.identity(problem.dim)
        # determine power of delta in adaptive sampling rule
        if self.factors["crn_across_solns"]:
            if gradient_availability:
                delta_power = 0
            else:
                delta_power = 2
        else:
            delta_power = 4

        pilot_run = ceil(
            max(
                lambda_min * log(10 + k, 10) ** 1.1,
                min(0.5 * problem.dim, lambda_max),
            )
            - 1
        )
        if k == 1:
            incumbent_solution = self.create_new_solution(incumbent_x, problem)
            if len(visited_pts_list) == 0:
                visited_pts_list.append(incumbent_solution)

            # pilot run
            problem.simulate(incumbent_solution, pilot_run)
            expended_budget += pilot_run
            sample_size = pilot_run

            # adaptive sampling
            while True:
                if gradient_availability:
                    rhs_for_kappa = norm(
                        incumbent_solution.objectives_gradients_mean[0]
                    )
                else:
                    rhs_for_kappa = incumbent_solution.objectives_mean
                sig2 = incumbent_solution.objectives_var[0]
                if delta_power == 0:
                    sig2 = max(
                        sig2,
                        np.trace(incumbent_solution.objectives_gradients_var),
                    )
                stopping = self.get_stopping_time(
                    pilot_run,
                    sig2,
                    delta_k,
                    rhs_for_kappa
                    * np.sqrt(pilot_run)
                    / (delta_k ** (delta_power / 2)),
                    problem.dim,
                    delta_power,
                )
                if (
                    sample_size >= min(stopping, lambda_max)
                    or expended_budget >= budget_limit
                ):
                    # calculate kappa
                    kappa = (
                        rhs_for_kappa
                        * np.sqrt(pilot_run)
                        / (delta_k ** (delta_power / 2))
                    )
                    # print("kappa "+str(kappa))
                    break
                else:
                    problem.simulate(incumbent_solution, 1)
                    expended_budget += 1
                    sample_size += 1

            recommended_solns.append(incumbent_solution)
            intermediate_budgets.append(expended_budget)
        elif self.factors[
            "crn_across_solns"
        ]:  # since incument was only evaluated with the sample size of previous incumbent, here we compute its adaptive sample size
            sample_size = incumbent_solution.n_reps
            # adaptive sampling
            while True:
                sig2 = incumbent_solution.objectives_var[0]
                if delta_power == 0:
                    sig2 = max(
                        sig2,
                        np.trace(incumbent_solution.objectives_gradients_var),
                    )
                stopping = self.get_stopping_time(
                    pilot_run, sig2, delta_k, kappa, problem.dim, delta_power
                )
                if (
                    sample_size >= min(stopping, lambda_max)
                    or expended_budget >= budget_limit
                ):
                    break
                else:
                    problem.simulate(incumbent_solution, 1)
                    expended_budget += 1
                    sample_size += 1

        # use Taylor expansion if gradient available
        if gradient_availability:
            fval = (
                np.ones(2 * problem.dim + 1)
                * -1
                * problem.minmax[0]
                * incumbent_solution.objectives_mean
            )
            grad = (
                -1
                * problem.minmax[0]
                * incumbent_solution.objectives_gradients_mean[0]
            )
            hessian = h_k
        else:
            # build the local model with interpolation (subproblem)
            (
                fval,
                y_var,
                q,
                grad,
                hessian,
                delta_k,
                expended_budget,
                interpolation_solns,
                visited_pts_list,
            ) = self.construct_model(
                incumbent_x,
                delta_k,
                k,
                problem,
                expended_budget,
                kappa,
                incumbent_solution,
                visited_pts_list,
                delta_power,
            )

        # solve the local model (subproblem)
        if easy_solve:
            # Cauchy reduction
            # TODO: why do we need this? Check model reduction calculation too.
            # print("np.dot(np.multiply(grad, Hessian), grad) "+str(np.dot(np.multiply(grad, hessian), grad)))
            # print("np.dot(np.dot(grad, hessian), grad) "+str(np.dot(np.dot(grad, hessian), grad)))
            if gradient_availability:
                # print("hessian " + str(hessian))
                check_positive_definite = np.dot(np.dot(grad, hessian), grad)
            else:
                check_positive_definite = np.dot(
                    np.multiply(grad, hessian), grad
                )
            if check_positive_definite <= 0:
                tau = 1
            else:
                tau = min(
                    1, norm(grad) ** 3 / (delta_k * check_positive_definite)
                )
                # print("tau "+str(tau))
            grad = np.reshape(grad, (1, problem.dim))[0]
            grad_norm = norm(grad)
            # Make sure we don't divide by 0
            if grad_norm == 0:
                candidate_x = incumbent_x
            else:
                product = tau * delta_k * grad
                adjustment = product / grad_norm
                candidate_x = incumbent_x - adjustment
            # if norm(incumbent_x - candidate_x) > 0:
            #     print("incumbent_x " + str(incumbent_x))
            #     print("candidate_x " + str(candidate_x))

        else:
            # Search engine - solve subproblem
            def subproblem(s: np.ndarray) -> float:
                s_grad_dot: np.ndarray = np.dot(s, grad)
                s_hessian_dot: np.ndarray = np.dot(np.multiply(s, hessian), s)
                result = fval[0] + s_grad_dot + s_hessian_dot
                return float(result[0])

            def con_f(s: np.ndarray) -> float:
                return float(norm(s))

            nlc = NonlinearConstraint(con_f, 0, delta_k)
            solve_subproblem = minimize(
                subproblem,
                np.zeros(problem.dim),
                method="trust-constr",
                constraints=nlc,
            )
            candidate_x = incumbent_x + solve_subproblem.x

        # print("problem.lower_bounds "+str(problem.lower_bounds))
        # handle the box constraints
        new_candidate_list = []
        for i in range(problem.dim):
            if candidate_x[i] <= problem.lower_bounds[i]:
                new_candidate_list.append(problem.lower_bounds[i] + 0.01)
            elif candidate_x[i] >= problem.upper_bounds[i]:
                new_candidate_list.append(problem.upper_bounds[i] - 0.01)
            else:
                new_candidate_list.append(candidate_x[i])
        candidate_x = tuple(new_candidate_list)

        # store the solution (and function estimate at it) to the subproblem as a candidate for the next iterate
        candidate_solution = self.create_new_solution(candidate_x, problem)
        visited_pts_list.append(candidate_solution)

        # if we use crn, then the candidate solution has the same sample size as the incumbent solution
        if self.factors["crn_across_solns"]:
            problem.simulate(candidate_solution, incumbent_solution.n_reps)
            expended_budget += incumbent_solution.n_reps
        else:
            # pilot run and adaptive sampling
            problem.simulate(candidate_solution, pilot_run)
            expended_budget += pilot_run
            sample_size = pilot_run
            while True:
                # if gradient_availability:
                #     # print("incumbent_solution.objectives_gradients_var[0] "+str(candidate_solution.objectives_gradients_var[0]))
                #     while norm(candidate_solution.objectives_gradients_var[0]) == 0 and candidate_solution.n_reps < max(pilot_run, lambda_max/100):
                #         problem.simulate(candidate_solution, 1)
                #         expended_budget += 1
                #         sample_size += 1
                sig2 = candidate_solution.objectives_var[0]
                if delta_power == 0:
                    sig2 = max(
                        sig2,
                        np.trace(candidate_solution.objectives_gradients_var),
                    )
                stopping = self.get_stopping_time(
                    pilot_run, sig2, delta_k, kappa, problem.dim, delta_power
                )
                if (
                    sample_size >= min(stopping, lambda_max)
                    or expended_budget >= budget_limit
                ):
                    break
                else:
                    problem.simulate(candidate_solution, 1)
                    expended_budget += 1
                    sample_size += 1

        # TODO: make sure the solution whose estimated objevtive is abrupted bc of budget is not added to the list of recommended solutions, unless the error is negligible ...
        # if (expended_budget >= budget_limit) and (sample_size < stopping):
        #     final_ob = fval[0]
        # else:
        # calculate success ratio
        fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
        # replace the candidate x if the interpolation set has lower objective function value and with sufficient reduction (pattern search)
        # also if the candidate solution's variance is high that could be caused by stopping early due to exhausting budget
        # print("cv "+str(candidate_solution.objectives_var/(candidate_solution.n_reps * candidate_solution.objectives_mean ** 2)))
        # print("fval[0] - min(fval) "+str(fval[0] - min(fval)))
        if not gradient_availability and (
            (
                (min(fval) < fval_tilde)
                and (
                    (fval[0] - min(fval))
                    >= self.factors["ps_sufficient_reduction"] * delta_k**2
                )
            )
            or (
                (
                    candidate_solution.objectives_var
                    / (
                        candidate_solution.n_reps
                        * candidate_solution.objectives_mean**2
                    )
                )[0]
                > 0.75
            )
        ):
            fval_tilde = min(fval)
            candidate_x = y_var[fval.index(min(fval))][0]  # type: ignore
            candidate_solution = interpolation_solns[fval.index(min(fval))]  # type: ignore

        # compute the success ratio rho
        candidate_x_arr = np.array(candidate_x)
        incumbent_x_arr = np.array(incumbent_x)
        s = np.subtract(candidate_x_arr, incumbent_x_arr)
        if gradient_availability:
            model_reduction = -np.dot(s, grad) - 0.5 * np.dot(
                np.dot(s, hessian), s
            )
        else:
            model_reduction = self.evaluate_model(
                np.zeros(problem.dim),
                q,  # type: ignore
            ) - self.evaluate_model(s, q)  # type: ignore
        if model_reduction <= 0:
            rho = 0
        else:
            rho = (fval[0] - fval_tilde) / model_reduction
        # successful: accept
        if rho >= eta_1:
            incumbent_x = candidate_x
            incumbent_solution = candidate_solution
            final_ob = candidate_solution.objectives_mean
            recommended_solns.append(candidate_solution)
            intermediate_budgets.append(expended_budget)
            delta_k = min(delta_k, delta_max)
            # very successful: expand
            if rho >= eta_2:
                delta_k = min(gamma_1 * delta_k, delta_max)
            if gradient_availability:
                candidate_grad = (
                    -1
                    * problem.minmax[0]
                    * candidate_solution.objectives_gradients_mean[0]
                )
                y_k = np.array(
                    candidate_grad - grad
                )  # np.clip(candidate_grad - grad, 1e-5, np.inf)
                # Compute the intermediate terms for the SMW update
                y_ks = y_k @ s
                if y_ks == 0:
                    warning_msg = (
                        "Warning: Division by 0 in ASTRO-DF solver (y_ks == 0)."
                    )
                    print(warning_msg, file=sys.stderr)
                    r_k = 0
                else:
                    r_k = 1.0 / (y_k @ s)
                h_s_k = h_k @ s
                h_k = (
                    h_k
                    + np.outer(y_k, y_k) * r_k
                    - np.outer(h_s_k, h_s_k) / (s @ h_s_k)
                ) # type: ignore
        # unsuccessful: shrink and reject
        else:
            delta_k = min(gamma_2 * delta_k, delta_max)
            final_ob = fval[0]

        # TODO: unified TR management
        # delta_k = min(kappa * norm(grad), delta_max)
        # print("norm of grad "+str(norm(grad)))
        return (
            final_ob,
            delta_k,
            recommended_solns,
            intermediate_budgets,
            expended_budget,
            incumbent_x,
            kappa,
            incumbent_solution,
            visited_pts_list,
            h_k,
        )

    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """
        Run a single macroreplication of a solver on a problem.
        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        crn_across_solns : bool
            indicates if CRN are used when simulating different solutions
        Returns
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes
        """

        budget = problem.factors["budget"]

        # Designate random number generator for random sampling
        find_next_soln_rng = self.rng_list[1]

        # Generate many dummy solutions without replication only to find a reasonable maximum radius
        dummy_solns: list[tuple[int, ...]] = []
        for _ in range(1000 * problem.dim):
            random_soln = problem.get_random_solution(find_next_soln_rng)
            dummy_solns.append(random_soln)
        # Range for each dimension is calculated and compared with box constraints range if given
        # TODO: just use box constraints range if given
        # delta_max = min(self.factors["delta_max"], problem.upper_bounds[0] - problem.lower_bounds[0])
        delta_max_arr: list[float | int] = []
        for i in range(problem.dim):
            delta_max_arr += [
                min(
                    max([sol[i] for sol in dummy_solns])
                    - min([sol[i] for sol in dummy_solns]),
                    problem.upper_bounds[0] - problem.lower_bounds[0],
                )
            ]
        # TODO: update this so that it could be used for problems with decision variables at varying scales!
        delta_max = max(delta_max_arr)
        # print("delta_max  " + str(delta_max))
        # Reset iteration and data storage arrays
        visited_pts_list = []
        k = 0
        delta_k = 10 ** (ceil(log(delta_max * 2, 10) - 1) / problem.dim)
        # print("initial delta " + str(delta_k))
        incumbent_x: tuple[int | float, ...] = problem.factors[
            "initial_solution"
        ]
        expended_budget, kappa = 0, 0
        recommended_solns, intermediate_budgets = [], []
        incumbent_solution = self.create_new_solution(
            tuple(incumbent_x), problem
        )
        h_k = np.identity(problem.dim).tolist()

        while expended_budget < budget:
            k += 1
            (
                final_ob,
                delta_k,
                recommended_solns,
                intermediate_budgets,
                expended_budget,
                incumbent_x,
                kappa,
                incumbent_solution,
                visited_pts_list,
                h_k,
            ) = self.iterate(
                k,
                delta_k,
                delta_max,
                problem,
                visited_pts_list,
                incumbent_x,
                expended_budget,
                budget,
                recommended_solns,
                intermediate_budgets,
                kappa,
                incumbent_solution,
                h_k, # type: ignore
            )

        return recommended_solns, intermediate_budgets
