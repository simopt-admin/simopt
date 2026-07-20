"""ASTRO-DF Solver.

The ASTRO-DF solver progressively builds local models (quadratic with diagonal Hessian)
using interpolation on a set of points on the coordinate bases of the best (incumbent)
solution. Solving the local models within a trust region (closed ball around the
incumbent solution) at each iteration suggests a candidate solution for the next
iteration. If the candidate solution is worse than the best interpolation point, it is
replaced with the latter (a.k.a. direct search). The solver then decides whether to
accept the candidate solution and expand the trust-region or reject it and shrink the
trust-region based on a success ratio test. The sample size at each visited point is
determined adaptively and based on closeness to optimality. A detailed description of
the solver can be found `here <https://simopt.readthedocs.io/en/latest/astrodf.html>`__.

This version does not require a delta_max, instead it estimates the maximum step size
using get_random_solution(). Parameter tuning on delta_max is therefore not needed and
removed from this version as well.
- Delta_max is so longer a factor, instead the maximum step size is estimated using get_random_solution().
- Parameter tuning on delta_max is therefore not needed and removed from this version as well.
- No upper bound on sample size may be better - testing
- It seems for SAN we always use pattern search - why? because the problem is convex and model may be misleading at the beginning
- Added sufficient reduction for the pattern search
"""  # noqa: E501

# TODO: check if bullet points can be indented and ignore tag removed

from __future__ import annotations

import logging
from math import ceil, log
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.linalg import LinAlgError, inv, norm, pinv
from pydantic import Field, model_validator
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize, LinearConstraint, lsq_linear
from scipy.linalg import null_space

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    SolverConfig,
    VariableType,
)
class InterpolationPoint:
    """One point in the interpolation set: its position (relative to the
    InterpolationSet's base_x) and the raw Solution object simulated there."""
    def __init__(self, x_dist: np.ndarray, solution: Solution):
        self.x_dist = x_dist      # position, relative to InterpolationSet.base_x
        self.solution = solution    # raw Solution — no sign convention applied

class InterpolationSet:
    """
    Persistent interpolation point set for ASTRO-DF's diagonal-Hessian local model.
    Points are stored as displacements from `base_x` and survive across iterations;
    only points flagged for replacement (Piece 4) are re-simulated.

    Stores raw Solution objects — callers apply neg_minmax when reading
    objective values, consistent with the rest of the solver.
    """
    def __init__(self, dim: int, base_x: np.ndarray, base_solution: Solution):
        self.dim = dim
        self.num_pts = 2 * dim + 1
        self.base_x = base_x.copy()
        # points[0] is always the current center point at construction time
        self.points: list[InterpolationPoint] = [InterpolationPoint(np.zeros(dim), base_solution)]
        self.kopt = 0  # index of best point (by raw objective, sign applied by caller)

    def size(self) -> int:
        return len(self.points)

    def is_complete(self) -> bool:
        return self.size() == self.num_pts

    def xopt_dist(self) -> np.ndarray:
        """Best point's displacement from base_x."""
        return self.points[self.kopt].x_dist

    def xopt_abs(self) -> np.ndarray:
        """Best point's coordinates, in absolute (problem) space."""
        return self.base_x + self.points[self.kopt].x_dist

    def solution_at(self, k: int) -> Solution:
        return self.points[k].solution

    def raw_obj_mean(self, k: int) -> float:
        """Raw (unsigned) objective mean at point k. Caller applies neg_minmax."""
        return float(np.asarray(self.points[k].solution.objectives_mean).reshape(-1)[0])

    def shift_base(self, new_base_x: np.ndarray) -> None:
        """Recompute all stored displacements relative to a new base point."""
        shift = new_base_x - self.base_x
        for pt in self.points:
            pt.x_dist = pt.x_dist - shift
        self.base_x = new_base_x.copy()
    
    def reduce_to_trust_region(self, delta_k: float) -> None:
        """Remove points whose displacement from base_x falls outside the trust region.
            Only called when recenter_mode == 'reduce'."""
        self.points = [pt for pt in self.points if norm(pt.x_dist) <= delta_k]        

    def add_point(
        self, x_dist: np.ndarray, solution: Solution, victim_idx: int | None = None
    ) -> None:
        """Insert a point. If below capacity, appends. If at capacity, evicts
        a victim first. If victim_idx is supplied, evicts that index directly.
        Otherwise falls back to the default distance-based _choose_exiting."""
        if self.size() >= self.num_pts:
            idx = victim_idx if victim_idx is not None else self._choose_exiting_by_dist()
            del self.points[idx]
        self.points.append(InterpolationPoint(x_dist, solution))    
        
    def _choose_exiting_by_dist(self) -> int:
        ''' Chose exiting point by distance to centerpoint only'''
        dists = [norm(pt.x_dist) for pt in self.points]
        return int(np.argmax(dists))
    
    

    def recompute_kopt(self, neg_minmax: int) -> None:
        """Caller supplies the sign convention of optimization problem"""
        values = [neg_minmax * self.raw_obj_mean(k) for k in range(self.size())]
        self.kopt = int(np.argmin(values))
        
    def recenter(
        self,
        new_base_x: np.ndarray,
        new_incumbent_solution: Solution,
        delta_k: float,
        neg_minmax: int,
        recenter_mode: str,
        victim_idx: int | None = None
    ) -> None:
        """Recenter the set on a new incumbent: shift coordinates, optionally
        drop out-of-radius points, insert the new incumbent, and refresh kopt.
        This is the single entry point the solver calls on a successful step --
        it exists so shift_base/reduce_to_trust_region/add_point/recompute_kopt
        can't be sequenced inconsistently across call sites."""
        self.shift_base(new_base_x)
    
        if recenter_mode == "reduce":
            self.reduce_to_trust_region(delta_k)
        elif recenter_mode != "shift":
            raise ValueError(f"Unknown recenter_mode: {recenter_mode!r}")
    
        # New incumbent sits at displacement 0 from itself
        self.add_point(np.zeros(self.dim), new_incumbent_solution, victim_idx = victim_idx)
        self.recompute_kopt(neg_minmax) # kopt currently not used for anything, we use x_k for interopolation update
        
    def _coordinate_offset(self, slot_idx: int, delta_k: float) -> np.ndarray:
        """Generate the displacement for interpolation slot `slot_idx` using plain
        coordinate directions: slot 0 is the center (handled separately), odd slots
        are +delta_k along axis i, even slots are -delta_k along axis i."""
        axis = (slot_idx - 1) // 2
        sign = 1.0 if (slot_idx - 1) % 2 == 0 else -1.0
        offset = np.zeros(self.dim)
        offset[axis] = sign * delta_k
        return offset
    
    def missing_index_offsets(self, delta_k: float) -> list[np.ndarray]:
        """Return displacement vectors for however many points are needed to
        reach num_pts, using fixed coordinate-axis placement at the trust
        region boundary. This is the fallback/init strategy for filling gaps --
        distinct from future Lagrange-polynomial-based placement, which will
        depend on the fitted model rather than a fixed slot pattern."""
        num_missing = self.num_pts - self.size()
        offsets = []
        slot_idx = 1  # slot 0 is reserved for the center point, never generated here
        while len(offsets) < num_missing:
            offsets.append(self._coordinate_offset( slot_idx, delta_k))
            slot_idx += 1
        return offsets
    
    def design_matrix_inputs(self, neg_minmax: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_var, fval) in the same shape get_model_coefficients expects:
        y_var[i] = displacement of point i from base_x, fval[i] = signed objective
        mean at point i. Caller applies neg_minmax since the set stores raw Solutions."""
        if not self.is_complete():
            raise ValueError(
                f"Cannot fit model: set has {self.size()} points, needs {self.num_pts}"
            )
        y_var = np.array([pt.x_dist for pt in self.points])
        fval = np.array([neg_minmax * self.raw_obj_mean(k) for k in range(self.size())])
        return y_var, fval
        
class SQPASTRODFConfig(SolverConfig):
    """Configuration for ASTRO-DF solver."""

    eta_1: Annotated[
        float,
        Field(default=0.1, gt=0, description="threshold for a successful iteration"),
    ]
    eta_2: Annotated[
        float,
        Field(
            default=0.8,
            description="threshold for a very successful iteration",
        ),
    ]
    gamma_1: Annotated[
        float,
        Field(
            default=2.5,
            gt=1,
            description="trust-region radius increase rate after successful iteration",
        ),
    ]
    gamma_2: Annotated[
        float,
        Field(
            default=0.5,
            gt=0,
            lt=1,
            description="trust-region radius decrease rate after failed iteration",
        ),
    ]
    lambda_min: Annotated[
        int, Field(default=5, gt=2, description="minimum sample size")
    ]
    easy_solve: Annotated[
        bool,
        Field(
            default=True,
            description="solve the subproblem approximately with Cauchy point",
        ),
    ]
    reuse_points: Annotated[
        bool, Field(default=True, description="reuse the previously visited points")
    ]
    ps_sufficient_reduction: Annotated[
        float,
        Field(
            default=0.1,
            ge=0,
            description=(
                "use pattern search if with sufficient reduction, "
                "0 always allows it, large value never does"
            ),
        ),
    ]
    use_gradients: Annotated[
        bool,
        Field(
            default=True,
            description="if direct gradient observations are available, use them",
        ),
    
    
    ]
    mu: Annotated[
        float,
        Field(
            default=0.1,
            description="criticality measure weight factor",
        ),
    
    
    ]
    a_normal: Annotated[
        float,
        Field(
            default=0.5,
            description="portion of the trust-region dedicated to the normal step",
        ),
    
    
    ]
    a_tangent: Annotated[
        float,
        Field(
            default=0.5,
            description="portion of the trust-region dedicated to the tangent step",
        ),
    
    
    ]
    nu: Annotated[
        float,
        Field(
            default=10e-4,
            description="used to ensure model improvement when normal step results in worsening objectives",
        ),
    
    
    ]
    tau_1: Annotated[
        float,
        Field(
            default=2.0,
            description="controls increase of pentalty parameter",
        ),
    
    
    ]
    tau_2: Annotated[
        float,
        Field(
            default=1.0,
            description="also controls increase of pentalty parameter, must have (tau_1 -1)*tau_2) > 0",
        ),
    
    
    ]
    sigma_min: Annotated[
        float,
        Field(
            default=10.0,
            description="initial penalty parameter",
        ),
    
    
    ]
    sigma_b_max: Annotated[
        float,
        Field(
            default=1e8,
            description="upper bound on simga_b",
        ),
    
    
    ]
    sigma_b_increase: Annotated[
        float,
        Field(
            default=0.01,
            description="multiple to increase sigma_b by each iteration, sigma_b_k = sigma_min + sigma_b_incrase*k",
        ),
    
    
    ]
    feas_tol: Annotated[
        float,
        Field(
            default=1e-8,
            description="Solve to specified feasibility tolerance",
        ),
    
    
    ]
    delta_0: Annotated[
        float,
        Field(
            default=1,
            description="Initial trust-region radius",
        ),
    ]
    delta_max: Annotated[
        float,
        Field(
            default=100,
            description="Maximum allowed trust-region radius",
        ),
    ]
    sampling_method: Annotated[
        str,
        Field(
            default="adaptive",
            description="Maximum allowed trust-region radius",
        ),
    ]
    epsilon: Annotated[
        float,
        Field(
            default=0.5,
            description="fraction to boundary measure for barrier problem",
        ),
    ]
    reuse_interpolation_set: Annotated[
    bool,
    Field(default=False,
          description="maintain a persistent interpolation set across iterations, reusing all valid points instead of rebuilding from scratch each time"),
]
    recenter_mode: Annotated[
    str,
    Field(default="shift", 
          description="how to recenter the persistent interpolation set on a new incumbent: 'shift' keeps all points (cheapest), 'reduce' discards points outside the new trust region (costlier, tighter geometry)"),
]
    use_lagrange_geometry: Annotated[
    bool,
    Field(
        default=True,
        description="after fitting, check point-set poisedness via Lagrange polynomials and replace the worst-poised point if it exceeds a threshold; only used when reuse_interpolation_set is True",
    ),
]
    lagrange_poisedness_threshold: Annotated[
        float,
        Field(
            default=10.0,
            description="trigger a Lagrange-based point replacement when max |L_k(x)| exceeds this value (py-bobyqa-style default; a well-poised set keeps this near 1)",
        ),
]
    theta_decrease: Annotated[
        float,
        Field(
            default=0.8,
            description="trigger a Lagrange-based point replacement when max |L_k(x)| exceeds this value (py-bobyqa-style default; a well-poised set keeps this near 1)",
        ),
]
    
    

    @model_validator(mode="after")
    def _validate_eta_2_greater_than_eta_1(self) -> Self:
        if self.eta_2 < self.eta_1:
            raise ValueError("Eta 2 must be greater than or equal to Eta 1.")
        return self


class SQPASTRODF(Solver):
    """The ASTRO-DF solver."""

    name: str = "SQPASTRODF"
    config_class: ClassVar[type[SolverConfig]] = SQPASTRODFConfig
    class_name_abbr: ClassVar[str] = "SQPASTRODF"
    class_name: ClassVar[str] = "SQP-ASTRO-DF"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    @property
    def iteration_count(self) -> int:
        """Get the current iteration count."""
        return self._iteration_count

    @iteration_count.setter
    def iteration_count(self, value: int) -> None:
        """Set the current iteration count."""
        self._iteration_count = value

    @property
    def delta_k(self) -> float:
        """Get the current delta_k value."""
        return self._delta_k

    @delta_k.setter
    def delta_k(self, value: float) -> None:
        """Set the current delta_k value."""
        self._delta_k = value

    @property
    def delta_max(self) -> float:
        """Get the current delta_max value."""
        return self._delta_max

    @delta_max.setter
    def delta_max(self, value: float) -> None:
        """Set the current delta_max value."""
        self._delta_max = value

    @property
    def incumbent_x(self) -> tuple[float, ...]:
        """Get the incumbent solution."""
        return self._incumbent_x

    @incumbent_x.setter
    def incumbent_x(self, value: tuple[float, ...]) -> None:
        """Set the incumbent solution."""
        self._incumbent_x = value

    @property
    def incumbent_solution(self) -> Solution:
        """Get the incumbent solution."""
        return self._incumbent_solution

    @incumbent_solution.setter
    def incumbent_solution(self, value: Solution) -> None:
        """Set the incumbent solution."""
        self._incumbent_solution = value

    @property
    def h_k(self) -> np.ndarray:
        """Get the Hessian approximation."""
        return self._h_k

    @h_k.setter
    def h_k(self, value: np.ndarray) -> None:
        """Set the Hessian approximation."""
        self._h_k = value

    def get_coordinate_vector(self, size: int, v_no: int) -> np.ndarray:
        """Generate the coordinate vector corresponding to the variable number v_no."""
        arr = np.zeros(size)
        arr[v_no] = 1.0
        return arr

    def _simulate_missing_indexes(self, delta_k: float, pilot_run: int) -> None:
        """Generate, simulate, and insert points for any empty slots in
        self.interp_set, using the fallback coordinate-offset strategy."""
        offsets = self.interp_set.missing_index_offsets(delta_k)
        for offset in offsets:
            raw_x = self.interp_set.base_x + offset
            # make sure x statisfies bound constraints
            x = tuple(
                clamp_with_epsilon(float(raw_x[i]), self.problem.lower_bounds[i], self.problem.upper_bounds[i])
                for i in range(self.problem.dim)
            )
            # simulate solution at interpolation point
            new_solution = self.create_new_solution(x, self.problem)
            self.budget.request(pilot_run)
            self.problem.simulate(new_solution, pilot_run)
            self.perform_adaptive_sampling(new_solution, pilot_run, delta_k)
    
            actual_dist = np.array(x) - self.interp_set.base_x  # may differ from `offset` due to clamping
            self.interp_set.add_point(actual_dist, new_solution)
    

    def get_rotated_basis(
        self, first_basis: np.ndarray, rotate_index: np.ndarray
    ) -> np.ndarray:
        """Generate the basis (rotated coordinate).

        The first vector comes from the visited design points (origin basis)
        """
        rotate_matrix = np.array(first_basis)
        rotation = np.zeros((2, 2), dtype=int)
        rotation[0][1] = -1
        rotation[1][0] = 1

        # rotate the coordinate basis based on the first basis vector (first_basis)
        # choose two dimensions which we use for the rotation (0,i)

        for i in range(1, len(rotate_index)):
            v1 = np.array(
                [
                    [first_basis[rotate_index[0]]],
                    [first_basis[rotate_index[i]]],
                ]
            )
            v2 = np.dot(rotation, v1)
            rotated_basis = np.copy(first_basis)
            rotated_basis[rotate_index[0]] = v2[0][0]
            rotated_basis[rotate_index[i]] = v2[1][0]
            # stack the rotated vector
            rotate_matrix = np.vstack((rotate_matrix, rotated_basis))

        return rotate_matrix

    def evaluate_model(self, x_k: np.ndarray, q: np.ndarray) -> float:
        """Evaluate a local quadratic model using linear interpolation and a diagonal Hessian.

        Args:
            x_k (np.ndarray): The point at which to evaluate the model
                (decision variables).
            q (np.ndarray): Coefficient vector defining the local quadratic model.

        Returns:
            np.ndarray: The evaluated model value as a NumPy array.
        """  # noqa: E501
        xk_arr = np.array(x_k).flatten()
        x_val = np.hstack(([1], xk_arr, xk_arr**2))
        return np.matmul(x_val, q).item()

    def get_stopping_time(
        self,
        pilot_run: int,
        sig2: float,
        delta: float,
        kappa: float,
    ) -> int:
        """Compute the sample size using adaptive stopping based on the optimality gap.

        Args:
            pilot_run (int): Number of initial samples used in the pilot run.
            sig2 (float): Estimated variance of the solution.
            delta (float): Optimality gap threshold.
            kappa (float): Constant in the stopping time denominator.
                If 0, it defaults to 1.

        Returns:
            int: The computed sample size, rounded up to the nearest integer.
        """
        if kappa == 0:
            kappa = 1

        # compute sample size
        raw_sample_size = pilot_run * max(
            1.0, max(sig2,1) / (kappa**2 * delta**self.delta_power)
        )
        return ceil(raw_sample_size)

    def select_interpolation_points(
        self, delta_k: float, f_index: int
    ) -> tuple[list, list]:
        """Select interpolation points for the local model.

        Args:
            delta_k (float): The current trust-region radius.
            f_index (int): The index of the farthest design point.

        Returns:
            tuple[list, list]: A tuple containing:
                - var_y (list): The interpolation points.
                - var_z (list): The reused design point.
        """
        if self.incumbent_x is None:
            raise ValueError("incumbent_x should be initialized before use")

        # If it is the first iteration or there is no design point we can reuse within
        # the trust region, use the coordinate basis
        if (
            not self.reuse_points
            or (
                norm(
                    np.array(self.incumbent_x)
                    - np.array(self.visited_pts_list[f_index].x)
                )
                == 0
            )
            or self.iteration_count == 1
        ):
            # Construct the interpolation set
            var_y = self.get_coordinate_basis_interpolation_points(
                self.incumbent_x, delta_k, self.problem
            )
            var_z = self.get_coordinate_basis_interpolation_points(
                tuple(np.zeros(self.problem.dim)), delta_k, self.problem
            )
        # Else if we will reuse one design point (k > 1)
        else:
            visited_pts_array = np.array(self.visited_pts_list[f_index].x)
            diff_array = visited_pts_array - np.array(self.incumbent_x)
            first_basis = (diff_array) / norm(diff_array)
            # if first_basis has some non-zero components, use rotated basis for those
            # dimensions
            rotate_list = np.nonzero(first_basis)[0]
            rotate_matrix = self.get_rotated_basis(first_basis, rotate_list)

            # if first_basis has some zero components, use coordinate basis for those
            # dimensions
            for i in range(self.problem.dim):
                if first_basis[i] == 0:
                    coord_vector = self.get_coordinate_vector(self.problem.dim, i)
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
                np.array(rotate_matrix),
                self.visited_pts_list[f_index].x,
            )
            var_z = self.get_rotated_basis_interpolation_points(
                np.zeros(self.problem.dim),
                delta_k,
                self.problem,
                np.array(rotate_matrix),
                np.array(self.visited_pts_list[f_index].x) - np.array(self.incumbent_x),
            )

        return var_y, var_z

    def _choose_exiting_at_step(self, matrix_inverse: np.ndarray, d: np.ndarray, delta_k: float) -> int:
        """After adding x_k, choose point to remove from interpolation set"""
        best_idx = None
        best_scaden = None #scaled denominator
        for k in range(self.interp_set.size()):
            coeffs = self.lagrange_polynomial_coeffs(matrix_inverse, k)
            den = self.evaluate_model(d, coeffs)
    
            point_dist = self.interp_set.points[k].x_dist
            distsq = float(norm(point_dist) ** 2)
            temp = max(1.0, (distsq / delta_k**2) ** 2)
    
            scaden = temp * abs(den)
            if best_scaden is None or scaden > best_scaden:
                best_scaden = scaden
                best_idx = k
        return best_idx

    def lagrange_polynomial_coeffs(self, matrix_inverse: np.ndarray, k: int) -> np.ndarray:
        """Returns kth lagrange polynomial coefficient vector of current interpolation set"""
        # slice uscaled matrix
        return matrix_inverse[:, k]    

    def _maximize_abs_lagrange_old(self, coeffs: np.ndarray, delta_k: float) -> tuple[np.ndarray, float]:
        """Find the displacement x (within the trust-region ball and box constraints)
        that maximizes |L(x)| for a Lagrange polynomial given by `coeffs - note this version has been replaced
        with an approximate version that has much smaller runtime"""
        dim = self.problem.dim
    
        def make_objective(sign: float):
            # run lagrange coeffs through model at given x since lagrange poly matches structure of model 
            def objective(x: np.ndarray) -> float:
                return sign * self.evaluate_model(x, coeffs)
            return objective
    
        def norm_x(x: np.ndarray) -> float:
            return float(norm(x))
    
        tr_constraint = NonlinearConstraint(norm_x, 0, delta_k)
        
        # respect bound constraints to avoid breaking simulations
        box_bounds = [
            (
                self.problem.lower_bounds[i] - self.interp_set.base_x[i],
                self.problem.upper_bounds[i] - self.interp_set.base_x[i],
            )
            for i in range(dim)
        ]
    
        best_x = None
        best_abs_val = -np.inf
        # find min and max of objective and take largest absolute value to avoid maximizing a norm
        for sign in (1.0, -1.0):
            result: OptimizeResult = minimize(
                make_objective(sign),
                np.zeros(dim),
                bounds=box_bounds,
                constraints=[tr_constraint],
            )
            abs_val = abs(self.evaluate_model(result.x, coeffs))
            if abs_val > best_abs_val:
                best_abs_val = abs_val
                best_x = result.x
    
        return best_x, best_abs_val
    
    def _maximize_abs_lagrange(self, coeffs: np.ndarray, delta_k: float) -> tuple[np.ndarray, float]:
        """Approximate maximizer of |L(x)| within the trust-region ball and box
        constraints"""
        dim = self.problem.dim
        c = coeffs[0]
        g = coeffs[1 : dim + 1]
        h = coeffs[dim + 1 :]
    
        lower = self.problem.lower_bounds - self.interp_set.base_x
        upper = self.problem.upper_bounds - self.interp_set.base_x
        box_lo = np.maximum(lower, -delta_k)
        box_hi = np.minimum(upper, delta_k)
    
        def solve_min(sign: float) -> np.ndarray:
            gs = sign * g
            hs = sign * h
            x = np.zeros(dim)
            for i in range(dim):
                lo, hi = box_lo[i], box_hi[i]
                if hs[i] > 0:
                    t_star = -gs[i] / hs[i]
                    x[i] = min(max(t_star, lo), hi)
                elif hs[i] < 0:
                    f_lo = gs[i] * lo + 0.5 * hs[i] * lo**2
                    f_hi = gs[i] * hi + 0.5 * hs[i] * hi**2
                    x[i] = lo if f_lo <= f_hi else hi
                else:
                    x[i] = lo if gs[i] > 0 else (hi if gs[i] < 0 else 0.0)
    
            # Bounding-box relaxation can violate the true ball -- project back.
            # Safe for box feasibility too: every box interval contains 0, so
            # scaling any feasible point toward 0 stays inside the (convex) box.
            norm_x = norm(x)
            if norm_x > delta_k and norm_x > 0:
                x = x * (delta_k / norm_x)
            return x

        best_x = None
        best_abs_val = -np.inf
        for sign in (1.0, -1.0):
            x_candidate = solve_min(sign)
            abs_val = abs(self.evaluate_model(x_candidate, coeffs))
            if abs_val > best_abs_val:
                best_abs_val = abs_val
                best_x = x_candidate

        return best_x, best_abs_val
    
    def _improve_geometry(self, matrix_inverse: np.ndarray, delta_k: float, pilot_run: int) -> None:
        """If distance to farthest point exceeds threshold, replace with point intended to impove geometry"""
        victim_idx = self.interp_set._choose_exiting_by_dist()
        victim_dist = self.interp_set.points[victim_idx].x_dist
        distsq = float(norm(victim_dist) ** 2)
    
        distsq_thresh = self.lagrange_poisedness_threshold * delta_k**2
        if distsq <= distsq_thresh:
            return False # farthest point still close enough -- nothing to do
    
        coeffs = self.lagrange_polynomial_coeffs(matrix_inverse, victim_idx)
        replacement_dist, _ = self._maximize_abs_lagrange(coeffs, delta_k)
    
        self._replace_worst_point(victim_idx, replacement_dist, delta_k, pilot_run)
        
        return True

    def _choose_exiting_lagrange(
        self, matrix_inverse: np.ndarray, delta_k: float
    ) -> tuple[int, np.ndarray, float]:
        """Lagrange-poisedness-based eviction: for each point k currently in the
        set, find the displacement that maximizes |L_k(x)| in the trust region.
        The point with the largest such value is the worst-poised  -- return its index along with the
        maximizing displacement and value"""
        worst_idx = None
        worst_x = None
        worst_val = -np.inf
        for k in range(self.interp_set.size()):
            coeffs = self.lagrange_polynomial_coeffs(matrix_inverse, k)
            x_max, max_abs_val = self._maximize_abs_lagrange(coeffs, delta_k)
            if max_abs_val > worst_val:
                worst_val = max_abs_val
                worst_x = x_max
                worst_idx = k
            if worst_idx is None:
                logging.warning(
                    "Lagrange eviction scan found no valid candidate "
                    "(all points returned nan) -- skipping geometry improvement this pass"
                )
                return None, None, -np.inf
        return worst_idx, worst_x, worst_val
    
    def _replace_worst_point(
            self,
            victim_idx: int,
            replacement_dist: np.ndarray,
            delta_k: float,
            pilot_run: int,) -> None:
        """Lagrange-poisedness-based geometry improvement: find the worst-poised
        point in the current set, simulate a replacement at the location that
        most improves that point's poisedness, and swap it in."""
    
        raw_x = self.interp_set.base_x + replacement_dist
        x = tuple(
            clamp_with_epsilon(float(raw_x[i]), self.problem.lower_bounds[i], self.problem.upper_bounds[i])
            for i in range(self.problem.dim)
        )
        
        #simulate new point entering interpolationset
        new_solution = self.create_new_solution(x, self.problem)
        self.budget.request(pilot_run)
        self.problem.simulate(new_solution, pilot_run)
        self.perform_adaptive_sampling(new_solution, pilot_run, delta_k)
    
        actual_dist = np.array(x) - self.interp_set.base_x  # may differ from replacement_dist due to clamping
        self.interp_set.add_point(actual_dist, new_solution, victim_idx=victim_idx)
        
        # victim_idx may have been kopt so recompute
        neg_minmax = -self.problem.minmax[0]
        self.interp_set.recompute_kopt(neg_minmax)

    def _simulate_surviving_points(self, delta_k: float, pilot_run: int) -> None:
        """Re-run adaptive sampling on every point currently in the set."""
        for k in range(self.interp_set.size()):
            self.perform_adaptive_sampling(self.interp_set.solution_at(k), pilot_run, delta_k)


    def perform_adaptive_sampling(
        self,
        solution: Solution,
        pilot_run: int,
        delta_k: float,
        compute_kappa: bool = False,
    ) -> None:
        """Perform adaptive sampling on a solution until the stopping condition is met.

        Args:
            solution (Solution): The solution object being sampled.
            pilot_run (int): The number of initial pilot runs.
            delta_k (float): The current trust-region radius.
            compute_kappa (bool): Whether or not to compute kappa dynamically (needed in
                the first iteration).
        """
        sample_size = solution.n_reps if solution.n_reps > 0 else pilot_run
        lambda_max = self.budget.remaining
        
        
        # run a fixed number of replications at each interpolation point
        if self.sampling_method != "adaptive":
            if solution.n_reps < self.lambda_min:
                more_reps = self.lambda_min - solution.n_reps 
                self.budget.request(more_reps)
                self.problem.simulate(solution, more_reps)
        
        else:
            

            # Initial Simulation (only if needed)
            if solution.n_reps == 0:
                self.budget.request(pilot_run)
                self.problem.simulate(solution, pilot_run)
                sample_size = pilot_run
    
            while True:
                # Compute variance
                sig2 = solution.objectives_var[0]
                if self.delta_power == 0:
                    sig2 = max(sig2, np.trace(solution.objectives_gradients_var))
    
                # Compute stopping condition
                kappa: float | None = None
                if compute_kappa:
                    if self.enable_gradient:
                        rhs_for_kappa = norm(solution.objectives_gradients_mean[0])
                    else:
                        rhs_for_kappa = solution.objectives_mean
                    kappa = (
                        rhs_for_kappa
                        * np.sqrt(pilot_run)
                        / (delta_k ** (self.delta_power / 2))
                    ).item()
    
                # Set k to the right kappa
                if kappa is not None:
                    k = kappa
                elif self.kappa is not None:
                    k = self.kappa
                else:
                    # TODO: figure out if we need to raise an error instead
                    logging.warning("kappa is not set. Using default value of 0.")
                    k = 0
                # Compute stopping time
                stopping = self.get_stopping_time(pilot_run, sig2, delta_k, k)
    
                # Stop if conditions are met
                if sample_size >= min(stopping, lambda_max) or self.budget.remaining <= 0:
                    if compute_kappa:
                        self.kappa = kappa  # Update kappa only if needed
                    break
    
                # Perform additional simulation
                self.budget.request(1)
                self.problem.simulate(solution, 1)
                sample_size += 1

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
        """Construct the local model for the current iteration.

        Construct the "qualified" local model for each iteration k with the center
        point x_k reconstruct with new points in a shrunk trust-region if the model
        fails the criticality condition the criticality condition keeps the model
        gradient norm and the trust-region size in lock-step
        """
        # Make sure we have our global variables initialized
        if self.delta_k is None:
            raise ValueError("delta_k should be initialized before use")
        if self.incumbent_x is None:
            raise ValueError("incumbent_x should be initialized before use")
        if self.incumbent_solution is None:
            raise ValueError("incumbent_solution should be initialized before use")
            
        if self.reuse_interpolation_set: # new interpolation method that re-uses points
            return self.construct_model_persistent() 
        else: #old interpolation method

            interpolation_solns = []
    
            ## inner loop parameters
            w = 0.85  # self.factors["w"]
            mu = self.mu
            beta = 10  # self.factors["beta"]
            # criticality_threshold = 0.1  # self.factors["criticality_threshold"]
            # skip_criticality = True  # self.factors["skip_criticality"]
            # Problem and solver factors
    
            lambda_max = self.budget.remaining
            # lambda_max = budget / (15 * sqrt(problem.dim))
            if self.sampling_method == "adaptive":
                pilot_run = ceil(
                    max(
                        self.lambda_min * log(10 + self.iteration_count, 10) ** 1.1,
                        min(0.5 * self.problem.dim, lambda_max),
                    )
                    - 1
                )
            else:
                pilot_run = self.lambda_min
    
            delta = self.delta_k
            model_iterations: int = 0
            while True:
                delta_k = delta * w**model_iterations
                model_iterations += 1
    
                # Calculate the distance between the center point and other design points
                distance_array: list[float] = []
                for point in self.visited_pts_list:
                    dist_diff = np.array(point.x) - np.array(self.incumbent_x)
                    distance = norm(dist_diff) - delta_k
                    # If the design point is outside the trust region, we will not reuse it
                    # (distance = -big M)
                    dist_to_append = -delta_k * 10000 if distance > 0 else distance
                    distance_array.append(float(dist_to_append))
    
                # Find the index of visited design points list for reusing points
                # The reused point will be the farthest point from the center point among
                # the design points within the trust region
                f_index = distance_array.index(max(distance_array))
    
                var_y, var_z = self.select_interpolation_points(delta_k, f_index)
    
                # Evaluate the function estimate for the interpolation points
                fval = []
                double_dim = 2 * self.problem.dim + 1
                for i in range(double_dim):
                    # If first iteration, reuse the incumbent solution
                    if i == 0:
                        adapt_soln = self.incumbent_solution
                    # If the second iteration and we can reuse points, reuse the farthest
                    # point from the center point
                    elif (
                        i == 1
                        and self.reuse_points
                        and norm(
                            np.array(self.incumbent_x)
                            - np.array(self.visited_pts_list[f_index].x)
                        )
                        != 0
                    ):
                        adapt_soln = self.visited_pts_list[f_index]
                    # Otherwise, create/initialize a new solution and use that
                    else:
                        decision_vars = tuple(var_y[i][0])
                        new_solution = self.create_new_solution(decision_vars, self.problem)
                        self.visited_pts_list.append(new_solution)
                        self.budget.request(pilot_run)
                        self.problem.simulate(new_solution, pilot_run)
                        adapt_soln = new_solution
    
                    # Don't perform adaptive sampling on x_0
                    if not (i == 0 and self.iteration_count == 0):
                        self.perform_adaptive_sampling(adapt_soln, pilot_run, delta_k)
    
                    # Append the function estimate to the list
                    obj_mean = float(np.asarray(adapt_soln.objectives_mean).reshape(-1)[0])
                    fval.append(-self.problem.minmax[0] * obj_mean)
                    interpolation_solns.append(adapt_soln)
    
                # construct the model and obtain the model coefficients
                q, grad, hessian, matrix_inverse = self.get_model_coefficients(var_z, fval, self.problem)
     
                norm_grad = norm(grad)
                if delta_k <= mu * norm_grad or norm_grad == 0:
                    break

            # If a model gradient norm is zero, there is a possibility that the code
            # stuck in this while loop
            # TODO: investigate if this can be implemented instead of checking
            # norm(grad) == 0
            # MAX_ITER = 100
            # if model_iterations > MAX_ITER:
            #     break

        beta_n_grad = float(beta * norm_grad)
        self.delta_k = min(max(beta_n_grad, delta_k), delta)

        return (
            fval,
            var_y,
            q,
            grad,
            hessian,
            interpolation_solns,
            matrix_inverse
        )
    
    # used for updated interpolation method
    def construct_model_persistent(
        self,
    ) -> tuple[list[float], list, np.ndarray, np.ndarray, np.ndarray, list[Solution]]:
        """Persistent-interpolation-set analogue of construct_model. Reuses
        self.interp_set across iterations rather than rebuilding from scratch;
        only refills/reduces points as needed for the current delta_k."""
        if self.delta_k is None:
            raise ValueError("delta_k should be initialized before use")
        if self.incumbent_x is None:
            raise ValueError("incumbent_x should be initialized before use")
    
        w = 0.85 #hard coded for now
        beta = 10 #hard coded for now
        mu = self.mu
    
        lambda_max = self.budget.remaining
        if self.sampling_method == "adaptive":
            pilot_run = ceil(
                max(
                    self.lambda_min * log(10 + self.iteration_count, 10) ** 1.1,
                    min(0.5 * self.problem.dim, lambda_max),
                )
                - 1
            )
        else:
            pilot_run = self.lambda_min
    
        delta = self.delta_k
        model_iterations = 0
        while True:
            delta_k = delta * self.gamma_2**model_iterations
            model_iterations += 1
    
            # Model-criticality check on the *previous* fit's geometry happens
            # after fitting below -- this pass first ensures the set matches delta_k.
            if self.recenter_mode == "reduce":
                self.interp_set.reduce_to_trust_region(delta_k)
            
            self._simulate_missing_indexes(delta_k, pilot_run)
            #self._simulate_surviving_points(delta_k, pilot_run)
    
            inverse_mult, grad, hessian, matrix_inverse = self.get_model_coefficients_persistent(delta_k)
            # update diag hessian with lagrange and compute lagrange multipliers
            H, self.lam = self.create_lagrange_hessian(hessian, grad) 
            # build problem matrices
            self.build_problem(grad, H)
            # update interpolation set if threshold for max lagrage poly is reached
            if self.use_lagrange_geometry:
                set_updated = self._improve_geometry(matrix_inverse, delta_k, pilot_run)
                # re-fit model coefficients if set has been updated
                if set_updated:
                    inverse_mult, grad, hessian, matrix_inverse = self.get_model_coefficients_persistent(delta_k)
            # check criticality loop, if not satisfied build model again with smaller delta
            if self.crit_measure >= self.mu*delta_k or self.crit_measure ==0:
                break

        beta_n_grad = float(beta * self.crit_measure)
        self.delta_k = min(max(beta_n_grad, delta_k), delta)
    
        # Build fval/y_var in the shapes the rest of iterate() expects
        neg_minmax = -self.problem.minmax[0]
        y_var, fval_arr = self.interp_set.design_matrix_inputs(neg_minmax)
        fval = list(fval_arr)
        interpolation_solns = [self.interp_set.solution_at(k) for k in range(self.interp_set.size())]
    
        return fval, list(y_var), inverse_mult, grad, hessian, interpolation_solns, matrix_inverse
    
    # this method is for old trust region interpolation construction
    def get_model_coefficients(
        self, y_var: list, fval: list, problem: Problem
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute model coefficients using 2d+1 design points and function values.

        This method fits a quadratic model with a diagonal Hessian by evaluating
        `2 * dim + 1` points centered at the solution.

        Args:
            y_var (list): List of sampled decision vectors (design points).
            fval (list): Corresponding function values for each design point.
            problem (Problem): Problem instance providing dimension and structure.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - q (np.ndarray): Coefficients of the fitted local quadratic model.
                - y_mean (np.ndarray): Mean of the y_var design points.
                - fval_mean (np.ndarray): Mean of the function values.
        """
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
        hessian = inverse_mult[decision_var_idx:num_design_points].reshape(problem.dim)

        return inverse_mult, grad, hessian, matrix_inverse
    
    # this method is for new interpolation handling
    def get_model_coefficients_persistent(self, delta_k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit the diagonal-Hessian model from self.interp_set, mirroring
        get_model_coefficients's math -- but conditioned by delta_k to avoid
        the design matrix becoming ill-scaled as delta_k shrinks in the
        criticality loop. Also returns the unscaled matrix_inverse, since its
        columns give each point's Lagrange polynomial coefficients directly
        (matrix_inverse @ e_k), needed for the future poisedness/eviction logic."""
        neg_minmax = -self.problem.minmax[0]
        y_var, fval = self.interp_set.design_matrix_inputs(neg_minmax)
    
        num_design_points = self.interp_set.num_pts
        dim = self.problem.dim
    
        # Fit in rescaled coordinates s = y_var / delta_k for numerical conditioning
        s = y_var / delta_k
        m_var = np.array(
            [
                np.hstack(([1], np.ravel(s[i]), np.ravel(s[i]) ** 2))
                for i in range(num_design_points)
            ]
        )
        #print("m_var_cond",np.linalg.cond(m_var))
        try:
            matrix_inverse_scaled = inv(m_var)
        except LinAlgError:
            matrix_inverse_scaled = pinv(m_var)
    
        # Unscale matrix_inverse row-wise: row 0 untouched, next `dim` rows /delta_k,
        # remaining `dim` rows /delta_k**2 -- same transform Step 2 applies to `q`.
        unscale_vec = np.concatenate((
            [1.0],
            np.full(dim, 1.0 / delta_k),
            np.full(dim, 1.0 / delta_k**2),
        ))
        matrix_inverse = matrix_inverse_scaled * unscale_vec[:, np.newaxis]
    
        inverse_mult = matrix_inverse @ fval  # equivalent to unscaling q after the scaled fit
    
        decision_var_idx = dim + 1
        grad = inverse_mult[1:decision_var_idx].reshape(dim)
        hessian = inverse_mult[decision_var_idx:num_design_points].reshape(dim)
    
        return inverse_mult, grad, hessian, matrix_inverse

    def get_coordinate_basis_interpolation_points(
        self, x_k: tuple[int | float, ...], delta: float, problem: Problem
    ) -> list[list[list[int | float]]]:
        """Compute the interpolation points (2d+1) using the coordinate basis."""
        y_var = [[list(x_k)]]
        is_block_constraint = sum(x_k) != 0
        num_decision_vars = problem.dim

        lower_bounds = problem.lower_bounds
        upper_bounds = problem.upper_bounds

        for var_idx in range(num_decision_vars):
            coord_vector = self.get_coordinate_vector(num_decision_vars, var_idx)
            coord_diff = delta * coord_vector

            minus: list[float] = [x - d for x, d in zip(x_k, coord_diff, strict=False)]
            plus: list[float] = [x + d for x, d in zip(x_k, coord_diff, strict=False)]

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
        """Compute the interpolation points (2d+1) using the rotated coordinate basis.

        One design point is reused, which is the farthest point from the center point.
        """
        y_var = [[x_k]]
        is_block_constraint = np.sum(x_k) != 0
        num_decision_vars = problem.dim

        lower_bounds = np.array(problem.lower_bounds)
        upper_bounds = np.array(problem.upper_bounds)

        for i in range(num_decision_vars):
            rotate_matrix_delta: np.ndarray = delta * rotate_matrix[i]

            plus = reused_x if i == 0 else x_k + rotate_matrix_delta

            minus = x_k - rotate_matrix_delta

            if is_block_constraint:
                minus = np.array(
                    [
                        clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                        for j, val in enumerate(minus)
                    ]
                )
                plus = np.array(
                    [
                        clamp_with_epsilon(val, lower_bounds[j], upper_bounds[j])
                        for j, val in enumerate(plus)
                    ]
                )

            y_var.append([plus])
            y_var.append([minus])

        return y_var

    def update_hessian(
        self, candidate_solution: Solution, grad: np.ndarray, s: np.ndarray
    ) -> None:
        """Performs Hessian update if gradients are enabled."""
        epsilon = 1e-15
        if not hasattr(self, "hessian_skip_count"):
            self.hessian_skip_count = 0

        def handle_hessian_skip(variable: str, value: float | np.ndarray) -> None:
            """Handles skipping Hessian update if gradients are near zero."""
            self.hessian_skip_count += 1
            message = (
                f"{variable} near zero ({value}); "
                "skipping Hessian update to avoid numerical instability. "
                f"({self.hessian_skip_count} consecutive skips)"
            )
            logging.debug(message)
            if self.hessian_skip_count == 10:
                message = (
                    "Hessian update skipped 10 consecutive times. "
                    "Check optimization stability."
                )
                logging.info(message)
            # If Hessian updates fail too often, the current approximation may
            # be useless or unstable. Resetting can prevent further instability
            # elif self.hessian_skip_count == 50:
            #     message = (
            #         "Hessian update skipped 50 consecutive times. "
            #         "Resetting Hessian approximation."
            #     )
            #     logging.warning(message)
            #     self.h_k = np.identity(self.problem.dim)
            #     self.hessian_skip_count = 0

        candidate_grad = (
            -1
            * self.problem.minmax[0]
            * candidate_solution.objectives_gradients_mean[0]
        )
        y_k = candidate_grad - grad
        y_ks = y_k @ s

        if np.isclose(y_ks, 0, atol=epsilon):
            handle_hessian_skip("y_ks", y_ks)
            return

        r_k = 1.0 / y_ks
        h_s_k = self.h_k @ s
        s_h_s_k = s @ h_s_k

        if np.all(np.isclose(s_h_s_k, 0, atol=epsilon)):
            handle_hessian_skip("s_h_s_k", s_h_s_k)
            return
        self.h_k += np.outer(y_k, y_k) * r_k - np.outer(h_s_k, h_s_k) / s_h_s_k
        # Reset counter on successful update
        self.hessian_skip_count = 0
        
    def estimate_lagrange_mult(self, grad, A_eq = None, A_ineq =None, c_ineq = None) -> np.array():
        """
        Compute estimited lagrange multipliers for constraints.
        
        A _eq= jacobian of equality constraints
        A_ineq = jacobian of inequality constraints
        grad = objective gradient
        
        Returns: vector of Lagrange mulitpliers ordered with equality constraints first followed by inequality constraints

        """
        
        #build problem based on constraint type
        if self.problem_type == "eq_only":
            res = lsq_linear(A_eq.T, -1*grad)
            return res.x
        else: #inequaltiy constraints present
            A = []
            lb = []
            ub = []
            n_eq = 0
            # add equaltiy constraints if present
            if self.problem_type == "both": 
                A.append(A_eq.T)
                n_eq = A_eq.shape[0]
                lb.extend([-np.inf]*n_eq)
                ub.extend([np.inf]*n_eq)

            #determine if constraint is active
            active = c_ineq >= self.feas_tol
            A_active = A_ineq[active]
            
            n_active = A_active.shape[0] #get number of active inequality constraints
            #add jacobian of active constraints and bounds to problem
            if n_active > 0:
                A.append(A_active.T)
                lb.extend([0.0]*n_active)
                ub.extend([np.inf]*n_active)
            # if no constraints are active all multipliers are 0
            if self.problem_type == "ineq_only" and n_active ==0 :
                return np.zeros(A_ineq.shape[0])
            
            #convert A to matrix
            A_lsq = np.hstack(A)
            res = lsq_linear(A_lsq, -1*grad, bounds=(np.array(lb),np.array(ub)))
            
            #build lambda vector
            lam = np.zeros(n_eq + A_ineq.shape[0]) #initialize lamda vector
            lam[:n_eq] = res.x[:n_eq] # add equality result
            if n_active > 0:
                lam[n_eq:][active] = res.x[n_eq:] # add active inequality constraints leaving inactive 0
                
            return lam

    def _cauchy_point_tangent(self, s_normal: np.ndarray, delta_hat: float) -> np.ndarray:
        """Closed-form Cauchy-point fallback for the tangent step, used when second degree curvature is negligable"""
        g = self.grad_term + self.W @ s_normal
    
        # Project g onto null(R) so the step satisfies R @ t = 0 by construction
        gp = self.P @ g
    
        norm_gp = norm(gp)
        if norm_gp == 0:
            return np.zeros(len(g))
    
        curvature = gp @ self.W @ gp
        if curvature <= 0:
            tau = 1.0
        else:
            tau = min(1.0, norm_gp**3 / (delta_hat * curvature))
    
        t_c = -tau * (delta_hat / norm_gp) * gp
    
        # Respect fraction-to-boundary box on the slack block: t_s >= -eps*e - w_s.
        # Origin (t=0) always satisfies this (w_s already respects its own bound
        # from solve_normal_step), so radial scaling toward 0 stays feasible --
        # same principle as the ball projection used elsewhere in this solver.
        if self.problem_type != "eq_only":
            lower = -self.epsilon * np.ones(self.n_v) - s_normal[self.n_x:]
            t_c_s = t_c[self.n_x:]
            alpha = 1.0
            for i in range(self.n_v):
                if t_c_s[i] < lower[i]:
                    alpha = min(alpha, lower[i] / t_c_s[i])
            t_c = alpha * t_c
    
        return t_c
        
    def solve_normal_step(self):
        
        # only determine normal step if current solution is infeasible
        if norm(self.feas) <= self.feas_tol : 
            s_normal = np.zeros(self.dim)
            s_normal_rescale = s_normal
        else:
            # Determine normal step
            def normal_subproblem(s_normal: np.ndarray) -> float:
                res: np.array =  self.R @ s_normal + self.feas
                return float(norm(res))
            # get norm of normal step
            def norm_s_normal(s_normal: np.ndarray) -> float:
                return float(norm(s_normal))
            # trust-region constraint
            tr_normal = NonlinearConstraint(norm_s_normal, 0, self.delta_k*self.a_normal)
            # set up normal equation based on problem type
            if self.problem_type == "eq_only":
                const = [tr_normal]
            else: # problem has inequality constraints
                #fraction to boundary constraint for slack variables
                E_v = np.hstack([
                    np.zeros((self.n_v, self.n_x)),
                    np.eye(self.n_v),
                    ])
                ftb = LinearConstraint(E_v, -1*self.a_normal*self.epsilon*np.ones(self.n_v), np.inf)
                const = [tr_normal, ftb]
            # solve for normal step
            solve_normal_subproblem: OptimizeResult = minimize( 
                normal_subproblem,
                np.zeros(self.dim),
                #method="trust-constr",
                constraints=const, 
            )
            # rescale normal step if necessary
            if self.problem_type == "eq_only":
                s_normal = solve_normal_subproblem.x 
                s_normal_rescale = s_normal # no rescale needed for eq only
            else:
                s_normal_rescale = solve_normal_subproblem.x 
                s_normal_v = self.V @ s_normal_rescale[self.n_x:]
                s_normal = np.concatenate((s_normal_rescale[:self.n_x], s_normal_v))
        return s_normal, s_normal_rescale
               
    def solve_tangent_step(self, s_normal):
        # if normal step = 0, allow tangent full trust region
        if norm(s_normal) == 0:
            a_tangent = 1.0 
        else:
            a_tangent = self.a_tangent
            
        # check if there is relevent second degree curvature
        curve = np.max(np.abs(np.diag(self.W))) if self.W.shape[0] > 0 else 0.0
        check = True
        #check = curve < self.feas_tol
        if check: # no relevent curvature, peform Cauchy step
            delta_hat = a_tangent*self.delta_k
            s_tangent_rescale = self._cauchy_point_tangent(s_normal, delta_hat)
        else:
            print(">>> ENTERED SLSQP/trust-constr BRANCH — check was False <<<")
            # Determine tangent step
            def tangent_subproblem(s_tangent: np.ndarray) -> float:
                res = (self.grad_term + self.W @ s_normal).T @ s_tangent + 0.5*(s_tangent.T @ self.W @ s_tangent)
                return float(res) 
            # get norm of tangent step
            def norm_s_tangent(s_tangent: np.ndarray) -> float:
                return float(norm(s_tangent))
            def tangent_subproblem_jac(s_tangent: np.ndarray) -> np.ndarray:
                return self.grad_term + self.W @ s_normal + self.W @ s_tangent
            
            def tangent_subproblem_hess(s_tangent: np.ndarray) -> np.ndarray:
                return self.W
            # constrain step by trust region
            tr_tangent = NonlinearConstraint(norm_s_tangent, 0, self.delta_k*a_tangent)
            #Linear constraint
            linc = LinearConstraint(self.R, np.zeros(self.R.shape[0]), np.zeros(self.R.shape[0]))    
            if self.problem_type == "eq_only":
                const = [tr_tangent,linc] 
            else: # problem has inequaltiy constraints
                E_v = np.hstack([
                    np.zeros((self.n_v, self.n_x)),
                    np.eye(self.n_v),
                    ])
                ftb = LinearConstraint(E_v, -1*self.epsilon*np.ones(self.n_v) - s_normal[self.n_x:], np.inf)
                const = [tr_tangent,linc,ftb]
            # solve for tangent step
            solve_tangent_subproblem: OptimizeResult = minimize(  # pyrefly: ignore
                tangent_subproblem,
                np.zeros(self.dim),
                method="trust-constr",
                constraints=const,
                hess = tangent_subproblem_hess,
                jac = tangent_subproblem_jac
            )
            s_tangent_rescale = solve_tangent_subproblem.x 
        if self.problem_type == "eq_only":
            s_tangent = s_tangent_rescale #no rescale needed for eq only
        else:
            s_tangent_v = self.V @ s_tangent_rescale[self.n_x:]
            s_tangent = np.concatenate((s_tangent_rescale[:self.n_x], s_tangent_v))
        return s_tangent, s_tangent_rescale

    def get_constraint_info(self, x):
        """Return lhs value of equality and inequality constraints as well as Jacobians of x (tuple of decision values)"""
        c_eq =  np.atleast_1d(self.problem.get_deterministic_equality_constraints(x))
        c_ineq = np.atleast_1d(self.problem.get_deterministic_inequality_constraints(x))
        A_eq = np.atleast_1d(self.problem.get_deterministic_equality_constraints_gradients(x))
        A_ineq = self.problem.get_deterministic_inequality_constraints_gradients(x)
        
        return c_eq, c_ineq, A_eq, A_ineq


    def build_problem(self, grad, H):
        '''
        Build necessary matrices for trust region constrained subproblems 

        Returns
        -------
        None.

        '''
        
        self.H = H
        
        if self.problem_type == "eq_only":
            self.feas = np.array(self.c_eq)
            self.R = self.A_eq
            self.grad_term = grad
            self.W = H
            self.V = None
            self.crit_measure = norm(self.feas) + norm(null_space(self.R).T @ grad)
        else:
            self.V = np.diag(self.incumbent_v)
            v = np.array(self.incumbent_v)
            J = self.A_ineq
            # construct objective terms
            self.grad_term = np.concatenate((grad, -1*self.theta*np.ones(self.n_v)))
            # save W to be used later
            self.W = np.block([
                [H, np.zeros((H.shape[0], self.n_v))],
                [np.zeros((self.n_v, H.shape[1])), self.theta * np.eye(self.n_v)]
                ])
        
            if self.problem_type == 'both':
                # if equalities present add eq jacobian
                A = self.A_eq
                m = A.shape[0]
                R_top = np.hstack((A, np.zeros((m, self.n_v))))
                R_bottom = np.hstack((J, self.V))
                self.R = np.vstack((R_top, R_bottom)) # save R to be used in tangent step
                self.feas = np.hstack((np.array(self.c_eq), np.array(self.c_ineq) + v))
            else: # otherwise construct without equality jacobian
                self.R = np.hstack([J, self.V])
                self.feas = np.array(self.c_ineq) + v
            # construct crit measure for inequality problems
            self.P = np.eye(self.R.shape[1]) - np.linalg.pinv(self.R)@ self.R
            self.crit_measure = norm(self.P @ self.grad_term) + norm(self.feas)
            
        
        
    def create_lagrange_hessian(self, hessian, grad):
        '''
        

        Returns
        -------
        Hessian matrix w/ corresponding Lagrange corrections based on constraint type and vector of lagrange multipliers.

        '''
        # convert inf or nan t0 num
        #np.nan_to_num(hessian, copy = False) 
        # create hessian diagonal matrix
        H = 2*np.diag(hessian)
        c_hess = self.problem.get_deterministic_constraints_hessian(self.incumbent_x)
        if self.problem_type == "eq_only":
            # estimate hessian of the lagrangian for equality-only problem
            lam = self.estimate_lagrange_mult (grad, A_eq = self.A_eq)
            for i in range(len(lam)):
                H -= lam[i]*c_hess[i] 
                
            return H, lam
                
        else: #problem has inequality constraints
            # add slack variables to inequality constraints
            if self.problem_type == "ineq_only":
                # estimate hessian of the lagrangian for equality-only problem
                lam = self.estimate_lagrange_mult(grad, A_ineq = self.A_ineq, c_ineq = self.c_ineq)
            else: #problem has both inequality and equality constraints
                lam = self.estimate_lagrange_mult(grad,A_eq = self.A_eq, A_ineq = self.A_ineq, c_ineq = self.c_ineq)
            # subtract constraint component from hessian
            for i in range(len(lam)):
                H -= lam[i]*c_hess[i]
            return H, lam
    
        
        
    def iterate(self) -> None:
        """Run one iteration of the ASTRO-DF algorithm.

        Build and solve a local model, update the current incumbent and trust-region
        radius, and save the data
        """
        self.iteration_count += 1
        neg_minmax = -self.problem.minmax[0]

        # determine power of delta in adaptive sampling rule
        if self.sampling_method == "adaptive":
            pilot_run = ceil(
                max(
                    self.lambda_min * log(10 + self.iteration_count, 10) ** 1.1,
                    min(0.5 * self.problem.dim, self.budget.total),
                )
                - 1
            )
        else: 
            pilot_run = self.lambda_min
        if self.iteration_count == 1:
            
            self.incumbent_solution = self.create_new_solution(
                self.incumbent_x, self.problem
            )
            self.visited_pts_list.append(self.incumbent_solution)

            self.perform_adaptive_sampling(
                self.incumbent_solution,
                pilot_run,
                self.delta_k,
                compute_kappa=True,
            )
            self.recommended_solns.append(self.incumbent_solution)
            self.intermediate_budgets.append(self.budget.used)
            
            #initialize interpolation set
            if self.reuse_interpolation_set:
                self.interp_set = InterpolationSet(
                    dim=self.problem.dim,
                    base_x=np.array(self.incumbent_x, dtype=float),
                    base_solution=self.incumbent_solution,
                )
            
            # if first iteration set current sigma to min
            self.sigma = self.sigma_min
        
            # set barrier parameter (only used in inequality constrained problems)
            #self.theta = min(1e-2, self.delta_k)
            self.theta = self.delta_k #set theta to starting delta
            self.theta_0 = self.delta_k
            # determine optimization type
            if self.problem.get_deterministic_equality_constraints(self.incumbent_x) == None:
                if self.problem.get_deterministic_inequality_constraints(self.incumbent_x) == None:
                    self.problem_type = "unconstrained"
                else:
                    self.problem_type = "ineq_only"
            elif self.problem.get_deterministic_inequality_constraints(self.incumbent_x) == None:
                self.problem_type = "eq_only"
            else:
                self.problem_type = "both"

            if self.problem_type in ("ineq_only", "both"):
               
                # set problem dimensions
                self.n_x = len(self.incumbent_x)
                self.n_v = len(np.atleast_1d(self.problem.get_deterministic_inequality_constraints(self.incumbent_x)))
                self.dim = self.n_x + self.n_v
            else: 
                self.n_x = len(self.incumbent_x)
                self.n_v = None
                self.dim = self.n_x

            
        # Since incument was only evaluated with the sample size of previous incumbent,
        # here we compute its adaptive sample size
        elif self.factors["crn_across_solns"]:
            self.perform_adaptive_sampling(
                self.incumbent_solution, pilot_run, self.delta_k
            )
        
        # get constraint info for current solution
        self.c_eq, self.c_ineq, self.A_eq, self.A_ineq = self.get_constraint_info(self.incumbent_x)
        #intialize slack variables
        if self.iteration_count == 1: 
            if self.problem_type in ("ineq_only", "both"):
                # just use feas tol as small constant to avoid another parameter
                self.incumbent_v = np.maximum(-self.c_ineq, self.epsilon * np.ones(self.n_v))
            else: 
                self.incumbent_v = None
        
        # use Taylor expansion if gradient available
        if self.enable_gradient:
            fval = (
                np.ones(2 * self.problem.dim + 1)
                * neg_minmax
                * self.incumbent_solution.objectives_mean
            )
            grad = neg_minmax * self.incumbent_solution.objectives_gradients_mean[0]
            hessian = self.h_k
            # Set empty variables to get rid of typing warnings
            q = np.array([])
            y_var = [[]]
            interpolation_solns: list[Solution] = []
        else:
            # build the local model with interpolation (subproblem)
            (
                fval,
                y_var,
                q,
                grad,
                hessian,
                interpolation_solns,
                matrix_inverse
            ) = self.construct_model()
        
        
        # remove easy solve
        
        # # solve the local model (subproblem)
        # if self.easy_solve:
        #     print("running easy solve...")
        #     s_normal = np.zeros(self.problem.dim)
        #     # Cauchy reduction
        #     # TODO: why do we need this? Check model reduction calculation too.
        #     # logging.debug(
        #     #     "np.dot(np.multiply(grad, Hessian), grad) "
        #     #     + str(np.dot(np.multiply(grad, hessian), grad))
        #     # )
        #     # logging.debug(
        #     #     "np.dot(np.dot(grad, hessian), grad) "
        #     #     + str(np.dot(np.dot(grad, hessian), grad))
        #     # )
        #     dot_a = np.dot(grad, hessian) if self.enable_gradient else grad * hessian

        #     check_positive_definite: float = np.dot(dot_a, grad)

        #     if check_positive_definite <= 0:
        #         tau = 1.0
        #     else:
        #         norm_ratio = norm(grad) ** 3 / (self.delta_k * check_positive_definite)
        #         tau = min(1.0, float(norm_ratio))

        #     grad: np.ndarray = np.reshape(grad, (1, self.problem.dim))[0]
        #     grad_norm = norm(grad)
        #     # Make sure we don't divide by 0
        #     if grad_norm == 0:
        #         candidate_x = self.incumbent_x
        #     else:
        #         product = tau * self.delta_k * grad
        #         adjustment = product / float(grad_norm)
        #         candidate_x = self.incumbent_x - adjustment
        #     # if norm(incumbent_x - candidate_x) > 0:
        #     #     logging.debug("incumbent_x " + str(incumbent_x))
        #     #     logging.debug("candidate_x " + str(candidate_x))
        
        # get lhs and Jacobian of constraints for current solution
        # self.c_eq =  np.atleast_1d(self.problem.get_deterministic_equality_constraints(self.incumbent_x))
        # self.c_ineq = np.atleast_1d(self.problem.get_deterministic_inequality_constraints(self.incumbent_x))
        # self.A_eq = np.atleast_1d(self.problem.get_deterministic_equality_constraints_gradients(self.incumbent_x))
        # self.A_ineq = self.problem.get_deterministic_inequality_constraints_gradients(self.incumbent_x)
        
        # intialize slack variables for first iteration
        if self.iteration_count == 1: 
            if self.problem_type in ("ineq_only", "both"):
                # just use feas tol as small constant to avoid another parameter
                self.incumbent_v = np.maximum(-self.c_ineq, self.epsilon * np.ones(self.n_v))
            else: 
                self.incumbent_v = None

        # convert any inf or -inf to numbers
        np.nan_to_num(hessian, copy = False)      
        # get hessian matrix w/ lagrange updates and lagrange mulitpliers
        #H, lam = self.create_lagrange_hessian(hessian, grad) 
        #temp remove lagrange updates to hessian
        #H = 2*np.diag(hessian)
        # build problem matrices
        #self.build_problem(grad, H)
        # compute normal step
        s_normal, s_normal_rescale = self.solve_normal_step()
        # compute tangent step
        s_tangent, s_tangent_rescale = self.solve_tangent_step(s_normal_rescale)
        # set composite step
        s = s_normal + s_tangent
        # set rescaled composite step (used to compute model improvements)
        s_rescale = s_normal_rescale + s_tangent_rescale
        # extract x variables from s
        if self.problem_type == "eq_only":
            s_x = s
            candidate_v = None
        else: # problem has inequality constraints and slack variables
            s_x = s[:self.n_x]
            candidate_v = self.incumbent_v + s[self.n_x:]
        # set new candidate solution 
        candidate_x = np.asarray(self.incumbent_x, dtype=float) + s_x
        

        # logging.debug("problem.lower_bounds "+str(problem.lower_bounds))
        # handle the box constraints
        candidate_x = tuple(
            clamp_with_epsilon(
                float(candidate_x[i]),
                self.problem.lower_bounds[i],
                self.problem.upper_bounds[i],
            )
            for i in range(self.problem.dim)
        )

        # Store the solution (and function estimate at it) to the subproblem as a
        # candidate for the next iterate
        candidate_solution = self.create_new_solution(candidate_x, self.problem)
        self.visited_pts_list.append(candidate_solution)

        # if we use crn, then the candidate solution has the same sample size as the
        # incumbent solution
        if self.factors["crn_across_solns"]:
            num_sims = self.incumbent_solution.n_reps
            self.budget.request(num_sims)
            self.problem.simulate(candidate_solution, num_sims)
        else:
            self.perform_adaptive_sampling(candidate_solution, pilot_run, self.delta_k)

        # TODO: make sure the solution whose estimated objevtive is abrupted bc of
        # budget is not added to the list of recommended solutions, unless the error
        # is negligible ...
        # if (expended_budget >= budget_limit) and (sample_size < stopping):
        #     final_ob = fval[0]
        # else:
        # calculate success ratio
        fval_tilde = neg_minmax * candidate_solution.objectives_mean
        # replace the candidate x if the interpolation set has lower objective function
        # value and with sufficient reduction (pattern search)
        # also if the candidate solution's variance is high that could be caused by
        # stopping early due to exhausting budget
        # logging.debug(
        #     "cv "
        #     + str(
        #         candidate_solution.objectives_var
        #         / (candidate_solution.n_reps * candidate_solution.objectives_mean**2)
        #     )
        # )
        # logging.debug("fval[0] - min(fval) " + str(fval[0] - min(fval)))
        
        #Disable pattern search for now

        # if not self.enable_gradient:
        #     min_fval = min(fval)
        #     sufficient_reduction = (fval[0] - min_fval) >= self.factors[
        #         "ps_sufficient_reduction"
        #     ] * self.delta_k**2

        #     condition_met = min_fval < fval_tilde and sufficient_reduction

        #     high_variance = False
        #     if not condition_met:
        #         # Treat variance as low if mean is zero to avoid division by
        #         # zero (zero mean typically indicates negligible uncertainty)
        #         if candidate_solution.objectives_mean[0] == 0:
        #             logging.debug(
        #                 "Candidate solution objectives_mean is zero, "
        #                 "skipping variance check."
        #             )
        #         else:
        #             high_variance = (
        #                 candidate_solution.objectives_var[0]
        #                 / (
        #                     candidate_solution.n_reps
        #                     * candidate_solution.objectives_mean[0] ** 2
        #                 )
        #             ) > 0.75

        #     if condition_met or high_variance:
 
        #         fval_tilde = min_fval
        #         min_idx = np.argmin(fval)
        #         candidate_x = y_var[min_idx][0]
        #         candidate_solution = interpolation_solns[min_idx]
        
        # reduction in normal model
        m_n_reduction = norm(self.feas) -norm(self.R @ s_normal_rescale + self.feas)   
        # reduction in tangent model
        #m_t_reduction = -1*(grad + H @ s_normal) @ s_tangent - .5*(s_tangent @ H @ s_tangent)
        #reduction in objective model after normal step
        #q_n_reduction = -1*(grad @ s_normal) - .5*(s_normal @ H @ s_normal)
        if norm(s_normal) == 0: # no normal step taken so normal improvment is 0
            sig_c_ratio = 0 # can use any value of sigma
        else:
            # objective improvement after normal step
            s_normal_x = s_normal[:self.n_x]
            q_n_reduction = -1*grad @ s_normal_x - 0.5* s_normal_x @ self.H @ s_normal_x
            if self.problem_type == "eq_only":
                barrier_reduction = 0 
            else:
                s_normal_v_rescale = s_normal_rescale[self.n_x:]
                barrier_reduction = self.theta*(np.sum(s_normal_v_rescale) - 0.5* s_normal_v_rescale @ s_normal_v_rescale)
            sig_numerator =-1* (q_n_reduction + barrier_reduction) 
            sig_c_ratio = sig_numerator / ((1-self.nu)*m_n_reduction)
        # set sigma b to be norm of lagrange multipliers
        sigma_b = min(norm(self.lam), self.sigma_b_max)
        sigma_c = max(sigma_b, sig_c_ratio)
        
        # update penalty parameter based on solving method
        if self.sampling_method != "IBO":
            if sigma_c > self.sigma:
                self.sigma = max(sigma_c, self.tau_1*self.sigma, self.sigma+ self.tau_2)
        else:
            self.sigma = max(self.sigma, sigma_b)
        # compute the success ratio rho
        # need to update success ratio to include constraints
        # candidate_x_arr = np.array(candidate_x)
        # incumbent_x_arr = np.array(self.incumbent_x)
        # s = s_normal + s_tangent
        # do not enable gradients for now
        # if self.enable_gradient:
        #     model_reduction = -np.dot(s, grad) - 0.5 * np.dot(np.dot(s, H), s)
        # else:
        #     # compute objective model reduction 
        #     model_reduction = self.evaluate_model(
        #         np.zeros(self.problem.dim),
        #         q,
        #     ) - self.evaluate_model(s, q)
        #     # compute merit model reduction
            
        # predicted reduction in merit model
        pred_merit_reduction = -1*self.grad_term @ s_rescale - 0.5*s_rescale @ self.W @ s_rescale + self.sigma*(m_n_reduction)
        # actual reduction in objective
        obj_reduction = fval[0] - fval_tilde
        if self.problem_type == "eq_only":
            feas_tilde = np.atleast_1d(self.problem.get_deterministic_equality_constraints(candidate_x))
            actual_barrier_reduction = 0
        else: # problem has inequality constraints
            # extract candidate slack variables
            if candidate_v[0] <= 0:
                print("curve", np.max(np.abs(np.diag(self.W))) if self.W.shape[0] > 0 else 0.0)
                print("incum v", self.incumbent_v)
                print("candidate", candidate_v)
                print("n_rescale", s_normal_rescale)
                print("t_rescale", s_tangent_rescale)
            if norm(candidate_v) <= 1e-12: # slack variables essentially 0
                actual_barrier_reduction = 0
            else:
                actual_barrier_reduction = -1*self.theta* (np.sum(np.log(self.incumbent_v) - np.log(candidate_v)))          
            if self.problem_type == "ineq_only":
                c_ineq_tilde = np.array(self.problem.get_deterministic_inequality_constraints(candidate_x))
                # extract candidate slack variables
                feas_tilde =  c_ineq_tilde + candidate_v
            else: # problem has equality and inequality constraints
                c_eq_tilde = np.array(self.problem.get_deterministic_equality_constraints(candidate_x))
                c_ineq_tilde = np.array(self.problem.get_deterministic_inequality_constraints(candidate_x))
                # extract candidate slack variables
                feas_tilde = np.hstack((c_eq_tilde, c_ineq_tilde + candidate_v))
        # actual reduction in feasibility 
        feas_reduction = self.sigma*( norm(self.feas) - norm(feas_tilde))
        # actual merit reduction
        actual_merit_reduction = obj_reduction + actual_barrier_reduction + feas_reduction
            
            
            
            # merit_model_reduction = m_t_reduction + self.sigma*(m_n_reduction) + model_reduction
            # merit_reduction = fval[0] - fval_tilde + self.sigma*(norm(c)-norm(candidate_c))
        
        # get trust region ratio
        rho = 0 if pred_merit_reduction <= 0 else actual_merit_reduction / pred_merit_reduction
        
        # print("s_normal:", s_normal_rescale)
        # print("s_tangent:", s_tangent_rescale)
        # print("s:", s_rescale )
        # print("candidate:", candidate_x)

        # check crit measure
        # if self.sampling_method != "IBO":
        #     crit_ok = self.crit_measure >= self.mu*self.delta_k
        # else:
        #     crit_ok = True
        successful = rho >= self.eta_1
        #print("delta", self.delta_k)
        # successful: accept
        if successful:
            self.incumbent_x = candidate_x
            self.incumbent_v = candidate_v
            
            self.incumbent_solution = candidate_solution
            self.recommended_solns.append(candidate_solution)
            self.intermediate_budgets.append(self.budget.used)
            self.delta_k = min(self.delta_k, self.delta_max)
            

            # very successful: expand
            if rho >= self.eta_2:
                self.delta_k = min(self.gamma_1 * self.delta_k, self.delta_max)

            if self.enable_gradient:
                self.update_hessian(candidate_solution, grad, s)
               
            # update interpolation set on successful iterations
            if self.reuse_interpolation_set:
                neg_minmax = -self.problem.minmax[0]
                d = np.array(candidate_x, dtype=float) - self.interp_set.base_x
                victim_idx = self._choose_exiting_at_step(matrix_inverse, d, self.delta_k)
                self.interp_set.recenter(
                    new_base_x=np.array(candidate_x, dtype=float),
                    new_incumbent_solution=candidate_solution,
                    delta_k=self.delta_k,
                    neg_minmax=neg_minmax,
                    recenter_mode=self.recenter_mode,
                    victim_idx=victim_idx
                )

        elif not successful:
            self.delta_k = min(self.gamma_2 * self.delta_k, self.delta_max)
        
        #update theta (only used for inequaltiy constrained problem)
        #force theta decrease for now
        self.theta = self.theta_decrease*self.theta

        # # have theta decrease with delta
        # if self.delta_k < self.theta:
        #     self.theta = self.delta_k
        #print("successful:", successful)
        
        # TODO: unified TR management
        # delta_k = min(kappa * norm(grad), self.delta_max)
        # logging.debug("norm of grad "+str(norm(grad)))

    def _initialize_solving(self) -> None:
        """Setup the solver for the first iteration."""
        self.eta_1: float = self.factors["eta_1"]
        self.eta_2: float = self.factors["eta_2"]
        self.gamma_1: float = self.factors["gamma_1"]
        self.gamma_2: float = self.factors["gamma_2"]
        self.easy_solve: bool = self.factors["easy_solve"]
        self.reuse_points: bool = self.factors["reuse_points"]
        self.lambda_min: int = self.factors["lambda_min"]
        self.mu : int = self.factors["mu"]
        self.a_normal : int = self.factors["a_normal"]
        self.a_tangent : int = self.factors["a_tangent"]
        self.nu : int = self.factors["nu"]
        self.tau_1 : int = self.factors["tau_1"]
        self.tau_2 : int = self.factors["tau_2"]
        self.sigma_min : int = self.factors["sigma_min"]
        self.sigma_b_max : int = self.factors["sigma_b_max"]
        self.sigma_b_increase : float = self.factors["sigma_b_increase"]
        self.feas_tol : float = self.factors["feas_tol"]
        self.sampling_method : str = self.factors["sampling_method"]
        self.epsilon : str = self.factors["epsilon"]
        self.reuse_interpolation_set: bool = self.factors["reuse_interpolation_set"]
        self.recenter_mode: str = self.factors["recenter_mode"]
        self.use_lagrange_geometry: bool = self.factors["use_lagrange_geometry"]
        self.lagrange_poisedness_threshold: str = self.factors["lagrange_poisedness_threshold"]
        self.theta_decrease : float = self.factors["theta_decrease"]
        if self.factors["delta_0"] is not None:
            self.delta_k : float = self.factors["delta_0"]
        if self.factors["delta_max"] is not None:
            self.delta_max : float = self.factors["delta_max"]
        # self.lambda_max = self.budget

        # Designate random number generator for random sampling
        rng = self.rng_list[1]

        # Generate dummy solutions to estimate a reasonable maximum radius
        dummy_solns = [
            self.problem.get_random_solution(rng)
            for _ in range(1000 * self.problem.dim)
        ]

        # Range for each dimension is calculated and compared with box constraints
        # range if given
        # TODO: just use box constraints range if given
        # self.delta_max = min(
        #     self.factors["delta_max"],
        #     problem.upper_bounds[0] - problem.lower_bounds[0]
        # )
        if self.factors["delta_0"]is None:
            delta_max_candidates: list[float | int] = []
            for i in range(self.problem.dim):
                sol_values = [sol[i] for sol in dummy_solns]
                min_soln, max_soln = min(sol_values), max(sol_values)
                bound_range = self.problem.upper_bounds[i] - self.problem.lower_bounds[i]
                delta_max_candidates.append(min(max_soln - min_soln, bound_range))
    
            # TODO: update this so that it could be used for problems with decision
            # variables at varying scales!
            self.delta_max = max(delta_max_candidates)
            # logging.debug("delta_max  " + str(self.delta_max))
    
            # Initialize trust-region radius
            self.delta_k = 10 ** (ceil(log(self.delta_max * 2, 10) - 1) / self.problem.dim)
        # logging.debug("initial delta " + str(self.delta_k))

        if "initial_solution" in self.problem.factors:
            self.incumbent_x = tuple(self.problem.factors["initial_solution"])
        else:
            self.incumbent_x = tuple(self.problem.get_random_solution(rng))

        self.incumbent_solution = self.create_new_solution(
            self.incumbent_x, self.problem
        )
        self.h_k = np.identity(self.problem.dim)
        

        self.enable_gradient = (
            self.problem.gradient_available and self.factors["use_gradients"]
        )

        if self.factors["crn_across_solns"]:
            self.delta_power = 0 if self.enable_gradient else 2
        else:
            # FIXME: fix type check error
            self.delta_power = 4  # pyrefly: ignore

        # Reset iteration count and data storage
        self.iteration_count = 0
        self.recommended_solns = []
        self.intermediate_budgets = []
        self.visited_pts_list = []
        self.kappa = None

    def solve(self, problem: Problem) -> None:  # noqa: D102
        self.problem = problem
        self._initialize_solving()

        while self.budget.remaining > 0:
            self.iterate()


def clamp_with_epsilon(
    val: float, lower_bound: float, upper_bound: float, epsilon: float = 0.01
) -> float:
    """Clamp a value within bounds while avoiding exact boundary values.

    Adds a small epsilon to the lower bound or subtracts it from the upper bound
    if `val` lies outside the specified range.

    Args:
        val (float): The value to clamp.
        lower_bound (float): Minimum acceptable value.
        upper_bound (float): Maximum acceptable value.
        epsilon (float, optional): Small margin to avoid returning exact boundary
            values. Defaults to 0.01.

    Returns:
        float: The adjusted value, guaranteed to lie strictly within the bounds.
    """
    if val <= lower_bound:
        return lower_bound + epsilon
    if val >= upper_bound:
        return upper_bound - epsilon
    return val
