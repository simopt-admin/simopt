import unittest

from rng.matmodops import *


A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]
     ]
Aneg = [[-1, -2, -3],
        [-4, -5, -6],
        [-7, -8, -9]
        ]
b = [1, 2, 3]
bneg = [-1, -2, -3]
m = 3


class TestMatModOps(unittest.TestCase):

    def test_mat33_mat31_mult(self):
        self.assertEqual(mat33_mat31_mult(A, b), [14, 32, 50])

    def test_mat33_mat33_mult(self):
        self.assertEqual(mat33_mat33_mult(A, A), [[30, 36, 42], [66, 81, 96], [102, 126, 150]])

    def test_mat31_mod(self):
        self.assertEqual(mat31_mod(b, m), [1, 2, 0])

    def test_mat31_mod_neg(self):
        self.assertEqual(mat31_mod(bneg, m), [2, 1, 0])

    def test_mat33_mod(self):
        self.assertEqual(mat33_mod(A, m), [[1, 2, 0], [1, 2, 0], [1, 2, 0]])

    def test_mat33_mod_neg(self):
        self.assertEqual(mat33_mod(Aneg, m), [[2, 1, 0], [2, 1, 0], [2, 1, 0]])

    def test_mat33_mat33_mod(self):
        self.assertEqual(mat33_mat33_mod(A, A, m), [[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_mat33_power_mod_power0(self):
        self.assertEqual(mat33_power_mod(A, 0, m), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_mat33_power_mod_power1(self):
        self.assertEqual(mat33_power_mod(A, 1, m), [[1, 2, 0], [1, 2, 0], [1, 2, 0]])

    def test_mat33_power_mod_power2(self):
        self.assertEqual(mat33_power_mod(A, 2, m), [[0, 0, 0], [0, 0, 0], [0, 0, 0]])

if __name__ == '__main__':
    unittest.main()