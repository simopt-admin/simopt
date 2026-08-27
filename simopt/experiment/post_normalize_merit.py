"""Merit-based post normalization (objective + penalty * feasibility violation)."""

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import Solution
from simopt.curve import Curve
from simopt.utils import make_nonzero
from simopt.deterministic_feasibility import (
    deterministic_feas_violation as _deterministic_feas_violation,
    merit_from_obj_and_feas as _merit_from_obj_and_feas,
)

from .single import ProblemSolver


def _best_merit_with_feasibility(
    experiments: list[ProblemSolver],
    ref_experiment: ProblemSolver,
    baseline_rngs: list[MRG32k3a],
    n_postreps_init_opt: int,
    obj_const: float,
    feas_tol_lower: float,
    feas_tol_upper: float,
) -> tuple[tuple, float, np.ndarray]:
    """Find the empirically best merit solution and re-simulate it at high precision.

    Mirrors `_best_with_feasibility` in post_normalize.py: rather than trusting the
    single noisy objective estimate that made a solution look best across every
    (experiment, mrep, budget) triple, re-simulate that winning solution with
    `n_postreps_init_opt` fresh replications to get a low-bias estimate of its true
    merit.
    """
    problem = ref_experiment.problem
    best_merit_so_far = np.inf
    best_x = None

    for experiment in experiments:
        for mrep in range(experiment.n_macroreps):
            objs = np.asarray(experiment.all_est_objectives[mrep])
            xs = experiment.all_recommended_xs[mrep]
            for obj, x in zip(objs, xs):
                feas = _deterministic_feas_violation(problem, x)
                m = _merit_from_obj_and_feas(
                    float(obj), feas, obj_const, feas_tol_lower, feas_tol_upper
                )
                if m < best_merit_so_far:
                    best_merit_so_far = m
                    best_x = x

    if best_x is None or not np.isfinite(best_merit_so_far):
        error_msg = (
            "No feasible-enough solutions found for which to estimate proxy m*."
        )
        raise RuntimeError(error_msg)

    # Re-simulate the winning solution at high precision, correcting the optimistic
    # bias from taking a min() over many noisy objective estimates.
    xstar_merit = best_x
    opt_soln = Solution(xstar_merit, ref_experiment.problem)
    opt_soln.attach_rngs(rng_list=baseline_rngs, copy=False)
    ref_experiment.problem.simulate(
        solution=opt_soln, num_macroreps=n_postreps_init_opt
    )
    xstar_merit_postreps = opt_soln.objectives[:, 0]

    refined_obj = float(np.mean(xstar_merit_postreps))
    refined_feas = _deterministic_feas_violation(problem, xstar_merit)
    refined_merit = _merit_from_obj_and_feas(
        refined_obj, refined_feas, obj_const, feas_tol_lower, feas_tol_upper
    )

    return xstar_merit, refined_merit, xstar_merit_postreps


def post_normalize_merit(
    experiments: list[ProblemSolver],
    obj_const: float = 1e6,
    feas_tol_upper: float = 1e-5,
    feas_tol_lower: float = 1e-8,
    proxy_init_merit: float | None = None,
    proxy_opt_merit: float | None = None,
) -> None:
    """Constructs merit and normalized merit-progress curves for a set of experiments
    on the same problem. Must be called AFTER `post_normalize`.

    Args:
        experiments: Problem-solver pairs for different solvers on the same problem.
        obj_const: Penalty multiplier on feasibility violation.
        feas_tol_upper: Violation above this is treated as fully infeasible (merit=inf).
        feas_tol_lower: Violation at or below this is treated as exactly feasible.
        proxy_init_merit: Known/override merit at x0.
        proxy_opt_merit: Known/override best merit m*. If provided, skips the
            empirical search and re-simulation step entirely.
    """
    ref_experiment = experiments[0]
    for experiment in experiments:
        if not getattr(experiment, "has_postnormalized", False):
            error_msg = (
                f"Run post_normalize on {experiment.solver.name}/"
                f"{experiment.problem.name} before post_normalize_merit."
            )
            raise RuntimeError(error_msg)

    problem = ref_experiment.problem
    n_postreps_init_opt = ref_experiment.n_postreps_init_opt

    # --- initial merit at x0 ---
    if proxy_init_merit is not None:
        initial_merit_val = proxy_init_merit
    else:
        x0_obj = float(np.mean(ref_experiment.x0_postreps))
        x0_feas = _deterministic_feas_violation(problem, ref_experiment.x0)
        initial_merit_val = _merit_from_obj_and_feas(
            x0_obj, x0_feas, obj_const, feas_tol_lower, feas_tol_upper
        )

    # --- best merit m*, refined via re-simulation ---
    xstar_merit = None
    xstar_merit_postreps = None
    if proxy_opt_merit is not None:
        best_merit_val = proxy_opt_merit
    else:
        # Stream 2: reserved for merit-optimal re-simulation (stream 0 is used by
        # post_normalize for x0/xstar, stream 1 by bootstrap_procedure).
        baseline_rngs = [
            MRG32k3a(s_ss_sss_index=[2, problem.model.n_rngs + rng_index, 0])
            for rng_index in range(problem.model.n_rngs)
        ]
        xstar_merit, best_merit_val, xstar_merit_postreps = (
            _best_merit_with_feasibility(
                experiments,
                ref_experiment,
                baseline_rngs,
                n_postreps_init_opt,
                obj_const,
                feas_tol_lower,
                feas_tol_upper,
            )
        )

    initial_merit_gap = make_nonzero(
        float(initial_merit_val - best_merit_val), "initial_merit_gap"
    )

    for experiment in experiments:
        experiment.best_merit = best_merit_val
        if xstar_merit is not None:
            experiment.xstar_merit = xstar_merit
            experiment.xstar_merit_postreps = xstar_merit_postreps

        experiment.merit_curves = []
        experiment.merit_progress_curves = []
        for mrep in range(experiment.n_macroreps):
            budgets = experiment.all_intermediate_budgets[mrep]
            xs = experiment.all_recommended_xs[mrep]
            obj_vals = experiment.objective_curves[mrep].y_vals

            raw_merit = []
            for obj, x in zip(obj_vals, xs):
                feas = _deterministic_feas_violation(problem, x)
                raw_merit.append(
                    _merit_from_obj_and_feas(
                        obj, feas, obj_const, feas_tol_lower, feas_tol_upper
                    )
                )
            # this part means merit must not increase after (no points for momentary blip of good value)
            # Convert to running-best-so-far, matching the implicit monotonicity assumption
            # that compute_crossing_time relies on for the objective-based progress curves.
            best_so_far = np.inf
            running_best_merit = []
            for m in raw_merit:
                best_so_far = min(best_so_far, m)
                running_best_merit.append(best_so_far)
            raw_merit = running_best_merit

            experiment.merit_curves.append(Curve(x_vals=budgets, y_vals=raw_merit))

            norm_merit = [
                max((m - best_merit_val) / initial_merit_gap, 0.0)
                if np.isfinite(m)
                else 1.0
                for m in raw_merit
            ]
            frac_budgets = [
                b / experiment.problem.factors["budget"] for b in budgets
            ]
            experiment.merit_progress_curves.append(
                Curve(x_vals=frac_budgets, y_vals=norm_merit)
            )

        experiment.has_merit_postnormalized = True