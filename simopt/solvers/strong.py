"""STRONG solver for simulation-optimization problems.

STRONG: A trust-region-based algorithm that fits first- or second-order models through function evaluations taken within
a neighborhood of the incumbent solution.
A detailed description of the solver can be found
`here <https://simopt.readthedocs.io/en/latest/strong.html>`__.
"""

from __future__ import annotations

from numpy.linalg import norm
import numpy as np
import math
from simopt.base import Solver, Problem, Solution


class STRONG(Solver):
    """A trust-region-based algorithm that fits first- or second-order models through function evaluations taken within a neighborhood of the incumbent solution.

    Attributes:
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

    Arguments:
    ---------
    name : str
        user-specified name for solver
    fixed_factors : dict
        fixed_factors of the solver

    See Also:
    --------
    base.Solver

    """

    def __init__(
        self, name: str = "STRONG", fixed_factors: dict | None = None
    ) -> None:
        """Initialize STRONG solver with default values for factors."""
        if fixed_factors is None:
            fixed_factors = {}

        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "use CRN across solutions?",
                "datatype": bool,
                "default": True,
            },
            "n0": {
                "description": "initial sample size",
                "datatype": int,
                "default": 10,
            },
            "n_r": {
                "description": "number of replications taken at each solution",
                "datatype": int,
                "default": 10,
            },
            "sensitivity": {
                "description": "shrinking scale for VarBds",
                "datatype": float,
                "default": 10 ** (-7),
            },
            "delta_threshold": {
                "description": "maximum value of the radius",
                "datatype": float,
                "default": 1.2,
            },
            "delta_T": {
                "description": "initial size of trust region",
                "datatype": float,
                "default": 2,
            },
            "eta_0": {
                "description": "constant for accepting",
                "datatype": float,
                "default": 0.01,
            },
            "eta_1": {
                "description": "constant for more confident accepting",
                "datatype": float,
                "default": 0.3,
            },
            "gamma_1": {
                "description": "constant for shrinking the trust region",
                "datatype": float,
                "default": 0.9,
            },
            "gamma_2": {
                "description": "constant for expanding the trust region",
                "datatype": float,
                "default": 1.11,
            },
            "lambda": {
                "description": "magnifying factor for n_r inside the finite difference function",
                "datatype": int,
                "default": 2,
            },
            "lambda_2": {
                "description": "magnifying factor for n_r in stage I and stage II",
                "datatype": float,
                "default": 1.01,
            },
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "n_r": self.check_n_r,
            "sensitivity": self.check_sensitivity,
            "delta_threshold": self.check_delta_threshold,
            "delta_T": self.check_delta_t,
            "eta_0": self.check_eta_0,
            "eta_1": self.check_eta_1,
            "gamma_1": self.check_gamma_1,
            "gamma_2": self.check_gamma_2,
            "lambda": self.check_lambda,
        }
        super().__init__(fixed_factors)

    def check_n_r(self) -> bool:
        """Check if n_r is a positive integer."""
        return self.factors["n_r"] > 0

    def check_sensitivity(self) -> bool:
        """Check if sensitivity is a positive float."""
        return self.factors["sensitivity"] > 0

    def check_delta_threshold(self) -> bool:
        """Check if delta_threshold is a positive float."""
        return self.factors["delta_threshold"] > 0

    def check_delta_t(self) -> bool:
        """Check if delta_T is a positive float."""
        return self.factors["delta_T"] > self.factors["delta_threshold"]

    def check_eta_0(self) -> bool:
        """Check if eta_0 is a float between 0 and 1."""
        return self.factors["eta_0"] > 0 and self.factors["eta_0"] < 1

    def check_eta_1(self) -> bool:
        """Check if eta_1 is a float between 0 and 1."""
        return (
            self.factors["eta_1"] < 1
            and self.factors["eta_1"] > self.factors["eta_0"]
        )

    def check_gamma_1(self) -> bool:
        """Check if gamma_1 is a float between 0 and 1."""
        return self.factors["gamma_1"] > 0 and self.factors["gamma_1"] < 1

    def check_gamma_2(self) -> bool:
        """Check if gamma_2 is a float greater than 1."""
        return self.factors["gamma_2"] > 1

    def check_lambda(self) -> bool:
        """Check if lambda is an integer greater than 1."""
        return self.factors["lambda"] > 1

    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """Run a single macroreplication of a solver on a problem.

        Arguments:
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        crn_across_solns : bool
            indicates if CRN are used when simulating different solutions

        Returns:
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes

        """
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0

        # Default values.
        n0 = self.factors["n0"]
        n_r = self.factors["n_r"]
        delta_threshold = self.factors["delta_threshold"]
        delta_t = self.factors["delta_T"]
        eta_0 = self.factors["eta_0"]
        eta_1 = self.factors["eta_1"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        lam = self.factors["lambda"]

        # Upper bound and lower bound.
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        # Start with the initial solution.
        new_solution = self.create_new_solution(
            problem.factors["initial_solution"], problem
        )
        problem.simulate(new_solution, n0)
        expended_budget += n0
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        while expended_budget < problem.factors["budget"]:
            new_x = new_solution.x
            # Check variable bounds.
            forward = np.isclose(
                new_x, lower_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            backward = np.isclose(
                new_x, upper_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            # BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
            bds_check = np.subtract(forward, backward)

            # Stage I.
            if delta_t > delta_threshold:
                # Step 1: Build the linear model.
                num_evals = 2 * problem.dim - np.sum(bds_check != 0)
                grad, hessian = self.finite_diff(
                    new_solution, bds_check, 1, problem, n_r
                )
                expended_budget += num_evals * n_r
                # A while loop to prevent zero gradient
                while norm(grad) == 0:
                    if expended_budget > problem.factors["budget"]:
                        break
                    grad, hessian = self.finite_diff(
                        new_solution, bds_check, 1, problem, n_r
                    )
                    expended_budget += num_evals * n_r
                    # Update n_r and counter after each loop.
                    n_r = int(lam * n_r)

                # Step 2: Solve the subproblem.
                # Cauchy reduction.
                candidate_x = self.cauchy_point(grad, hessian, new_x, problem)
                candidate_solution = self.create_new_solution(
                    tuple(candidate_x), problem
                )

                # Step 3: Compute the ratio.
                # Use n_r simulated observations to estimate g_new.
                problem.simulate(candidate_solution, n_r)
                expended_budget += n_r
                # Find the old objective value and the new objective value.
                g_old = -1 * problem.minmax[0] * new_solution.objectives_mean
                g_new = (
                    -1 * problem.minmax[0] * candidate_solution.objectives_mean
                )
                # Construct the polynomial.
                r_old = g_old
                r_new = (
                    g_old
                    + np.matmul(np.subtract(candidate_x, new_x), grad)
                    + 0.5
                    * np.matmul(
                        np.matmul(np.subtract(candidate_x, new_x), hessian),
                        np.subtract(candidate_x, new_x),
                    )
                )
                rho = (g_old - g_new) / (r_old - r_new)

                # Step 4: Update the trust region size and determine to accept or reject the solution.
                if (
                    (rho < eta_0)
                    | ((g_old - g_new) <= 0)
                    | ((r_old - r_new) <= 0)
                ):
                    # The solution fails either the RC or SR test, the center point reamins and the trust region shrinks.
                    delta_t = gamma_1 * delta_t
                elif (eta_0 <= rho) & (rho < eta_1):
                    # The center point moves to the new solution and the trust region remains.
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                else:
                    # The center point moves to the new solution and the trust region enlarges.
                    delta_t = gamma_2 * delta_t
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                n_r = int(np.ceil(self.factors["lambda_2"] * n_r))

            # Stage II.
            # When trust region size is very small, use the quadratic design.
            else:
                n_onbound = np.sum(bds_check != 0)
                if n_onbound <= 1:
                    num_evals = problem.dim**2
                else:
                    num_evals = (
                        problem.dim**2
                        + problem.dim
                        - math.factorial(n_onbound)
                        / (math.factorial(2), math.factorial(n_onbound - 2))
                    )
                # Step1: Build the quadratic model.
                grad, hessian = self.finite_diff(
                    new_solution, bds_check, 2, problem, n_r
                )
                expended_budget += num_evals * n_r
                # A while loop to prevent zero gradient
                while norm(grad) == 0:
                    if expended_budget > problem.factors["budget"]:
                        break
                    grad, hessian = self.finite_diff(
                        new_solution, bds_check, 2, problem, n_r
                    )
                    expended_budget += num_evals * n_r
                    # Update n_r and counter after each loop.
                    n_r = int(lam * n_r)
                # Step2: Solve the subproblem.
                # Cauchy reduction.
                candidate_x = self.cauchy_point(
                    grad,
                    hessian,
                    new_x,
                    problem,
                )
                candidate_solution = self.create_new_solution(
                    tuple(candidate_x), problem
                )
                # Step 3: Compute the ratio.
                # Use r simulated observations to estimate g(x_start\).
                problem.simulate(candidate_solution, n_r)
                expended_budget += n_r
                # Find the old objective value and the new objective value.
                g_old = -1 * problem.minmax[0] * new_solution.objectives_mean
                g_new = (
                    -1 * problem.minmax[0] * candidate_solution.objectives_mean
                )
                # Construct the polynomial.
                r_old = g_old
                r_new = (
                    g_old
                    + np.matmul(np.subtract(candidate_x, new_x), grad)
                    + 0.5
                    * np.matmul(
                        np.matmul(np.subtract(candidate_x, new_x), hessian),
                        np.subtract(candidate_x, new_x),
                    )
                )
                rho = (g_old - g_new) / (r_old - r_new)
                # Step4: Update the trust region size and determine to accept or reject the solution.
                if (
                    (rho < eta_0)
                    | ((g_old - g_new) <= 0)
                    | ((r_old - r_new) <= 0)
                ):
                    # Inner Loop.
                    rr_old = r_old
                    g_b_old = rr_old
                    sub_counter = 1
                    result_solution = new_solution
                    result_x = new_x

                    while np.sum(result_x != new_x) == 0:
                        if expended_budget > problem.factors["budget"]:
                            break
                        # Step1: Build the quadratic model.
                        g, h = self.finite_diff(
                            new_solution,
                            bds_check,
                            2,
                            problem,
                            (sub_counter + 1) * n_r,
                        )
                        expended_budget += num_evals * (sub_counter + 1) * n_r
                        # A while loop to prevent zero gradient
                        while norm(g) == 0:
                            if expended_budget > problem.factors["budget"]:
                                break
                            g, h = self.finite_diff(
                                new_solution,
                                bds_check,
                                2,
                                problem,
                                (sub_counter + 1) * n_r,
                            )
                            expended_budget += (
                                num_evals * (sub_counter + 1) * n_r
                            )
                            # Update n_r and counter after each loop.
                            n_r = int(lam * n_r)

                        # Step2: determine the new inner solution based on the accumulated design matrix X.
                        try_x = self.cauchy_point(g, h, new_x, problem)
                        try_solution = self.create_new_solution(
                            tuple(try_x), problem
                        )

                        # Step 3.
                        problem.simulate(
                            try_solution,
                            int(
                                n_r
                                + np.ceil(
                                    sub_counter ** self.factors["lambda_2"]
                                )
                            ),
                        )
                        expended_budget += int(
                            n_r
                            + np.ceil(sub_counter ** self.factors["lambda_2"])
                        )
                        g_b_new = (
                            -1
                            * problem.minmax[0]
                            * try_solution.objectives_mean
                        )
                        dummy_solution = new_solution
                        problem.simulate(
                            dummy_solution,
                            int(
                                np.ceil(sub_counter ** self.factors["lambda_2"])
                                - np.ceil(
                                    (sub_counter - 1)
                                    ** self.factors["lambda_2"]
                                )
                            ),
                        )
                        expended_budget += int(
                            np.ceil(sub_counter ** self.factors["lambda_2"])
                            - np.ceil(
                                (sub_counter - 1) ** self.factors["lambda_2"]
                            )
                        )
                        dummy = (
                            -1
                            * problem.minmax[0]
                            * dummy_solution.objectives_mean
                        )
                        # Update g_old.
                        g_b_old = (
                            g_b_old
                            * (
                                n_r
                                + np.ceil(
                                    (sub_counter - 1)
                                    ** self.factors["lambda_2"]
                                )
                            )
                            + (
                                np.ceil(sub_counter ** self.factors["lambda_2"])
                                - np.ceil(
                                    (sub_counter - 1)
                                    ** self.factors["lambda_2"]
                                )
                            )
                            * dummy
                        ) / (
                            n_r
                            + np.ceil(sub_counter ** self.factors["lambda_2"])
                        )
                        rr_new = (
                            g_b_old
                            + np.matmul(np.subtract(try_x, new_x), g)
                            + 0.5
                            * np.matmul(
                                np.matmul(np.subtract(try_x, new_x), h),
                                np.subtract(try_x, new_x),
                            )
                        )
                        rr_old = g_b_old
                        rrho = (g_b_old - g_b_new) / (rr_old - rr_new)
                        if (
                            (rrho < eta_0)
                            | ((g_b_old - g_b_new) <= 0)
                            | ((rr_old - rr_new) <= 0)
                        ):
                            delta_t = gamma_1 * delta_t
                            result_solution = new_solution
                            result_x = new_x

                        elif (eta_0 <= rrho) & (rrho < eta_1):
                            # Accept the solution and remains the size of trust region.
                            result_solution = try_solution
                            result_x = try_x
                            rr_old = g_b_new
                        else:
                            # Accept the solution and expand the size of trust region.
                            delta_t = gamma_2 * delta_t
                            result_solution = try_solution
                            result_x = try_x
                            rr_old = g_b_new
                        sub_counter = sub_counter + 1
                    new_solution = result_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                elif (eta_0 <= rho) & (rho < eta_1):
                    # The center point moves to the new solution and the trust region remains.
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                else:
                    # The center point moves to the new solution and the trust region enlarges.
                    delta_t = gamma_2 * delta_t
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                n_r = int(np.ceil(self.factors["lambda_2"] * n_r))
        return recommended_solns, intermediate_budgets

    def cauchy_point(self, grad: np.array, hessian: np.array, new_x: np.array, problem: Problem) -> np.array:
        """Find the Cauchy point based on the gradient and Hessian matrix.

        Arguments:
        ---------
        grad : np.array
            gradient of the objective function
        hessian : np.array
            Hessian matrix of the objective function
        new_x : np.array
            current solution
        problem : Problem object
            simulation-optimization problem to solve

        Returns:
        -------
        Cauchy_x : np.array
            Cauchy point

        """
        delta_t = self.factors["delta_T"]
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        if np.dot(np.matmul(grad, hessian), grad) <= 0:
            tau = 1
        else:
            tau = min(
                1,
                norm(grad) ** 3
                / (delta_t * np.dot(np.matmul(grad, hessian), grad)),
            )
        grad = np.reshape(grad, (1, problem.dim))[0]
        candidate_x = new_x - tau * delta_t * grad / norm(grad)
        cauchy_x = self.check_cons(candidate_x, new_x, lower_bound, upper_bound)
        return cauchy_x

    def check_cons(self, candidate_x: np.array, new_x: np.array, lower_bound: np.array, upper_bound: np.array) -> np.array:
        """Check the feasibility of the Cauchy point and update the point accordingly.

        Arguments:
        ---------
        candidate_x : np.array
            candidate solution
        new_x : np.array
            current solution
        lower_bound : np.array
            lower bound of the variables
        upper_bound : np.array
            upper bound of the variables

        Returns:
        -------
        modified_x : np.array
            modified Cauchy point

        """
        # The current step.
        step_v = np.subtract(candidate_x, new_x)
        # Form a matrix to determine the possible stepsize.
        tmax_v = np.ones((2, len(candidate_x)))
        for i in range(0, len(candidate_x)):
            if step_v[i] > 0:
                tmax_v[0, i] = (upper_bound[i] - new_x[i]) / step_v[i]
            elif step_v[i] < 0:
                tmax_v[1, i] = (lower_bound[i] - new_x[i]) / step_v[i]
        # Find the minimum stepsize.
        t2 = tmax_v.min()
        # Calculate the modified x.
        modified_x = new_x + t2 * step_v
        return modified_x

    def finite_diff(self, new_solution: Solution, bds_check: np.array, stage: int, problem: Problem, n_r: int) -> tuple[np.array, np.array]:
        """Finite difference for calculating gradients and BFGS for calculating Hessian matrix.

        Arguments:
        ---------
        new_solution : Solution object
            current solution
        bds_check : np.array
            check variable bounds
        stage : int
            stage of the algorithm
        problem : Problem object
            simulation-optimization problem to solve
        n_r : int
            number of replications taken at each solution

        Returns:
        -------
        grad : np.array
            gradient of the objective function
        Hessian : np.array
            Hessian matrix of the objective function

        """
        delta_t = self.factors["delta_T"]
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -1 * problem.minmax[0] * new_solution.objectives_mean
        new_x = new_solution.x
        # Store values for each dimension.
        fn_plus_minus = np.zeros((problem.dim, 3))
        grad = np.zeros(problem.dim)
        hessian = np.zeros((problem.dim, problem.dim))

        for i in range(problem.dim):
            # Initialization.
            x1 = list(new_x)
            x2 = list(new_x)
            # Forward stepsize.
            steph1 = delta_t
            # Backward stepsize.
            steph2 = delta_t

            # Check variable bounds.
            if x1[i] + steph1 > upper_bound[i]:
                steph1 = np.abs(upper_bound[i] - x1[i])
            if x2[i] - steph2 < lower_bound[i]:
                steph2 = np.abs(x2[i] - lower_bound[i])

            # Decide stepsize.
            # Central diff.
            if bds_check[i] == 0:
                fn_plus_minus[i, 2] = min(steph1, steph2)
                x1[i] = x1[i] + fn_plus_minus[i, 2]
                x2[i] = x2[i] - fn_plus_minus[i, 2]
            # Forward diff.
            elif bds_check[i] == 1:
                fn_plus_minus[i, 2] = steph1
                x1[i] = x1[i] + fn_plus_minus[i, 2]
            # Backward diff
            else:
                fn_plus_minus[i, 2] = steph2
                x2[i] = x2[i] - fn_plus_minus[i, 2]
            x1_solution = self.create_new_solution(tuple(x1), problem)
            if bds_check[i] != -1:
                problem.simulate_up_to([x1_solution], n_r)
                fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
                # First column is f(x+h,y).
                fn_plus_minus[i, 0] = fn1
            x2_solution = self.create_new_solution(tuple(x2), problem)
            if bds_check[i] != 1:
                problem.simulate_up_to([x2_solution], n_r)
                fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
                # Second column is f(x-h,y).
                fn_plus_minus[i, 1] = fn2

            # Calculate gradient.
            if bds_check[i] == 0:
                grad[i] = (fn1 - fn2) / (2 * fn_plus_minus[i, 2])
            elif bds_check[i] == 1:
                grad[i] = (fn1 - fn) / fn_plus_minus[i, 2]
            elif bds_check[i] == -1:
                grad[i] = (fn - fn2) / fn_plus_minus[i, 2]

        if stage == 2:
            # Diagonal in Hessian.
            for i in range(problem.dim):
                if bds_check[i] == 0:
                    hessian[i, i] = (
                        fn_plus_minus[i, 0] - 2 * fn + fn_plus_minus[i, 1]
                    ) / (fn_plus_minus[i, 2] ** 2)
                elif bds_check[i] == 1:
                    x3 = list(new_x)
                    x3[i] = x3[i] + fn_plus_minus[i, 2] / 2
                    x3_solution = self.create_new_solution(tuple(x3), problem)
                    # Check budget.
                    problem.simulate_up_to([x3_solution], n_r)
                    fn3 = -1 * problem.minmax[0] * x3_solution.objectives_mean
                    hessian[i, i] = (
                        4
                        * (fn_plus_minus[i, 1] - 2 * fn3 + fn)
                        / (fn_plus_minus[i, 2] ** 2)
                    )
                elif bds_check[i] == -1:
                    x4 = list(new_x)
                    x4[i] = x4[i] - fn_plus_minus[i, 2] / 2
                    x4_solution = self.create_new_solution(tuple(x4), problem)
                    # Check budget.
                    problem.simulate_up_to([x4_solution], n_r)
                    fn4 = -1 * problem.minmax[0] * x4_solution.objectives_mean
                    hessian[i, i] = (
                        4
                        * (fn - 2 * fn4 + fn_plus_minus[i, 1])
                        / (fn_plus_minus[i, 2] ** 2)
                    )

                # Upper triangle in Hessian
                for j in range(i + 1, problem.dim):
                    # Neither x nor y on boundary.
                    if bds_check[i] ** 2 + bds_check[j] ** 2 == 0:
                        # Represent f(x+h,y+k).
                        x5 = list(new_x)
                        x5[i] = x5[i] + fn_plus_minus[i, 2]
                        x5[j] = x5[j] + fn_plus_minus[j, 2]
                        x5_solution = self.create_new_solution(
                            tuple(x5), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x5_solution], n_r)
                        fn5 = (
                            -1 * problem.minmax[0] * x5_solution.objectives_mean
                        )
                        # Represent f(x-h,y-k).
                        x6 = list(new_x)
                        x6[i] = x6[i] - fn_plus_minus[i, 2]
                        x6[j] = x6[j] - fn_plus_minus[j, 2]
                        x6_solution = self.create_new_solution(
                            tuple(x5), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x6_solution], n_r)
                        fn6 = (
                            -1 * problem.minmax[0] * x6_solution.objectives_mean
                        )
                        # Compute second order gradient.
                        hessian[i, j] = (
                            fn5
                            - fn_plus_minus[i, 0]
                            - fn_plus_minus[j, 0]
                            + 2 * fn
                            - fn_plus_minus[i, 1]
                            - fn_plus_minus[j, 1]
                            + fn6
                        ) / (2 * fn_plus_minus[i, 2] * fn_plus_minus[j, 2])
                        hessian[j, i] = hessian[i, j]
                    # When x on boundary, y not.
                    elif bds_check[j] == 0:
                        # Represent f(x+/-h,y+k).
                        x5 = list(new_x)
                        x5[i] = x5[i] + bds_check[i] * fn_plus_minus[i, 2]
                        x5[j] = x5[j] + fn_plus_minus[j, 2]
                        x5_solution = self.create_new_solution(
                            tuple(x5), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x5_solution], n_r)
                        fn5 = (
                            -1 * problem.minmax[0] * x5_solution.objectives_mean
                        )
                        # Represent f(x+/-h,y-k).
                        x6 = list(new_x)
                        x6[i] = x6[i] + bds_check[i] * fn_plus_minus[i, 2]
                        x6[j] = x6[j] - fn_plus_minus[j, 2]
                        x6_solution = self.create_new_solution(
                            tuple(x6), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x6_solution], n_r)
                        fn6 = (
                            -1 * problem.minmax[0] * x6_solution.objectives_mean
                        )
                        # Compute second order gradient.
                        hessian[i, j] = (
                            fn5 - fn_plus_minus[j, 0] - fn6 + fn_plus_minus[j, 1]
                        ) / (
                            2
                            * fn_plus_minus[i, 2]
                            * fn_plus_minus[j, 2]
                            * bds_check[i]
                        )
                        hessian[j, i] = hessian[i, j]
                    # When y on boundary, x not.
                    elif bds_check[i] == 0:
                        # Represent f(x+h,y+/-k).
                        x5 = list(new_x)
                        x5[i] = x5[i] + fn_plus_minus[i, 2]
                        x5[j] = x5[j] + bds_check[j] * fn_plus_minus[j, 2]
                        x5_solution = self.create_new_solution(
                            tuple(x5), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x5_solution], n_r)
                        fn5 = (
                            -1 * problem.minmax[0] * x5_solution.objectives_mean
                        )
                        # Represent f(x-h,y+/-k).
                        x6 = list(new_x)
                        x6[i] = x6[i] + fn_plus_minus[i, 2]
                        x6[j] = x6[j] + bds_check[j] * fn_plus_minus[j, 2]
                        x6_solution = self.create_new_solution(
                            tuple(x6), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x6_solution], n_r)
                        fn6 = (
                            -1 * problem.minmax[0] * x6_solution.objectives_mean
                        )
                        # Compute second order gradient.
                        hessian[i, j] = (
                            fn5 - fn_plus_minus[i, 0] - fn6 + fn_plus_minus[i, 1]
                        ) / (
                            2
                            * fn_plus_minus[i, 2]
                            * fn_plus_minus[j, 2]
                            * bds_check[j]
                        )
                        hessian[j, i] = hessian[i, j]
                    elif bds_check[i] == 1:
                        if bds_check[j] == 1:
                            # Represent f(x+h,y+k).
                            x5 = list(new_x)
                            x5[i] = x5[i] + fn_plus_minus[i, 2]
                            x5[j] = x5[j] + fn_plus_minus[j, 2]
                            x5_solution = self.create_new_solution(
                                tuple(x5), problem
                            )
                            # Check budget.
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = (
                                -1
                                * problem.minmax[0]
                                * x5_solution.objectives_mean
                            )
                            # Compute second order gradient.
                            hessian[i, j] = (
                                fn5 - fn_plus_minus[i, 0] - fn_plus_minus[j, 0] + fn
                            ) / (fn_plus_minus[i, 2] * fn_plus_minus[j, 2])
                            hessian[j, i] = hessian[i, j]
                        else:
                            # Represent f(x+h,y-k).
                            x5 = list(new_x)
                            x5[i] = x5[i] + fn_plus_minus[i, 2]
                            x5[j] = x5[j] - fn_plus_minus[j, 2]
                            x5_solution = self.create_new_solution(
                                tuple(x5), problem
                            )
                            # Check budget.
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = (
                                -1
                                * problem.minmax[0]
                                * x5_solution.objectives_mean
                            )
                            # Compute second order gradient.
                            hessian[i, j] = (
                                fn_plus_minus[i, 0] - fn5 - fn + fn_plus_minus[j, 1]
                            ) / (fn_plus_minus[i, 2] * fn_plus_minus[j, 2])
                            hessian[j, i] = hessian[i, j]
                    elif bds_check[i] == -1:
                        if bds_check[j] == 1:
                            # Represent f(x-h,y+k).
                            x5 = list(new_x)
                            x5[i] = x5[i] - fn_plus_minus[i, 2]
                            x5[j] = x5[j] + fn_plus_minus[j, 2]
                            x5_solution = self.create_new_solution(
                                tuple(x5), problem
                            )
                            # Check budget
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = (
                                -1
                                * problem.minmax[0]
                                * x5_solution.objectives_mean
                            )
                            # Compute second order gradient.
                            hessian[i, j] = (
                                fn_plus_minus[j, 0] - fn - fn5 + fn_plus_minus[i, 1]
                            ) / (fn_plus_minus[i, 2] * fn_plus_minus[j, 2])
                            hessian[j, i] = hessian[i, j]
                        else:
                            # Represent f(x-h,y-k).
                            x5 = list(new_x)
                            x5[i] = x5[i] - fn_plus_minus[i, 2]
                            x5[j] = x5[j] - fn_plus_minus[j, 2]
                            x5_solution = self.create_new_solution(
                                tuple(x5), problem
                            )
                            # Check budget.
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = (
                                -1
                                * problem.minmax[0]
                                * x5_solution.objectives_mean
                            )
                            # Compute second order gradient.
                            hessian[i, j] = (
                                fn - fn_plus_minus[j, 1] - fn_plus_minus[i, 1] + fn5
                            ) / (fn_plus_minus[i, 2] * fn_plus_minus[j, 2])
                            hessian[j, i] = hessian[i, j]
        return grad, hessian
