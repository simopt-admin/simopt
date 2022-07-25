#!/usr/bin/env python
"""
Summary
-------
Useful matrix/modulus operations for mrg32k3a generator.
"""


def mat33_mat31_mult(A, b):
    """Multiply a 3 x 3 matrix with a 3 x 1 matrix.

    Parameters
    ----------
    A : list [list [float]]
        3 x 3 matrix.
    b : list [float]
        3 x 1 matrix.

    Returns
    -------
    res : list [float]
        3 x 1 matrix.
    """
    res = [0, 0, 0]
    r3 = range(3)
    for i in r3:
        res[i] = sum([A[i][j] * b[j] for j in r3])
    return res


def mat33_mat33_mult(A, B):
    """Multiply a 3x3 matrix with a 3x3 matrix.

    Parameters
    ----------
    A : list [list [float]]
        3 x 3 matrix.
    B : list [list [float]]
        3 x 3 matrix.

    Returns
    -------
    res : list [float]
        3 x 3 matrix.
    """
    res = [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]
           ]
    r3 = range(3)
    for i in r3:
        for j in r3:
            res[i][j] = sum([A[i][k] * B[k][j] for k in r3])
    return res


def mat31_mod(b, m):
    """Compute moduli of a 3 x 1 matrix.

    Parameters
    ----------
    b : list [float]
        3 x 1 matrix.
    m : float
        Modulus.

    Returns
    -------
    res : list [float]
        3 x 1 matrix.
    """
    res = [0, 0, 0]
    for i in range(3):
        res[i] = int(b[i] - int(b[i] / m) * m)
        # if negative, add back modulus m
        if res[i] < 0:
            res[i] += m
    return res


def mat33_mod(A, m):
    """Compute moduli of a 3 x 3 matrix.

    Parameters
    ----------
    A : list [float]
        3 x 3 matrix.
    m : float
        Modulus.

    Returns
    -------
    res : list [float]
        3 x 3 matrix.
    """
    res = [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]
           ]
    r3 = range(3)
    for i in r3:
        for j in r3:
            res[i][j] = int(A[i][j] - int(A[i][j] / m) * m)
            # if negative, add back modulus m
            if res[i][j] < 0:
                res[i][j] += m
    return res


def mat33_mat33_mod(A, B, m):
    """Compute moduli of a 3 x 3 matrix x 3 x 3 matrix product.

    Parameters
    ----------
    A : list [list [float]]
        3 x 3 matrix.
    B : list [list [float]]
        3 x 3 matrix.
    m : float
        Modulus.

    Returns
    -------
    res : list [list [float]]
        3 x 3 matrix.
    """
    C = mat33_mat33_mult(A, B)
    res = mat33_mod(C, m)
    return res


def mat33_power_mod(A, j, m):
    """Compute moduli of a 3 x 3 matrix power.
    Use divide-and-conquer algorithm described in L'Ecuyer (1990).

    Parameters
    ----------
    A : list [list [float]]
        3 x 3 matrix.
    j : int
        Exponent.
    m : float
        Modulus.

    Returns
    -------
    res : list [list [float]]
        3 x 3 matrix.
    """
    B = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]
         ]
    while j > 0:
        if (j % 2 == 1):
            B = mat33_mat33_mod(A, B, m)
        A = mat33_mat33_mod(A, A, m)
        j = int(j / 2)
    res = B
    return res
