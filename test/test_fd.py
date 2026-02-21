from __future__ import annotations

from itertools import product

import numpy as np

from simopt.solvers.utils import fd, fd_old


def _make_x_for_modes(
    modes: tuple[int, ...], lower_bound: np.ndarray, upper_bound: np.ndarray
) -> np.ndarray:
    x = (lower_bound + upper_bound) / 2
    for i, mode in enumerate(modes):
        if mode == 1:
            x[i] = lower_bound[i]
        elif mode == -1:
            x[i] = upper_bound[i]
    return x


def test_fd_matches_fd2_all_modes_dim3() -> None:
    dim = 3
    lower_bound = np.array([-2.0, -1.0, -3.0])
    upper_bound = np.array([3.0, 2.0, 4.0])
    step = 0.5

    a_mat = np.array(
        [
            [2.0, -0.5, 0.25],
            [-0.5, 1.5, 0.1],
            [0.25, 0.1, 3.0],
        ]
    )
    w = np.array([0.5, -1.25, 2.0])
    b = -0.7

    def fn(z: np.ndarray) -> float:
        return float(z @ a_mat @ z + w @ z + b)

    for modes in product((-1, 0, 1), repeat=dim):
        bounds_check = np.array(modes, dtype=int)
        x = _make_x_for_modes(modes, lower_bound, upper_bound)
        fn_value = fn(x)

        grad_fd = fd_old(x, fn_value, bounds_check, lower_bound, upper_bound, step, fn)
        grad_fd2 = fd(fn, x, step, fn_value, bounds_check, lower_bound, upper_bound)

        np.testing.assert_allclose(grad_fd, grad_fd2, rtol=1e-12, atol=1e-12)
