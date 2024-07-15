import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_HOTEL1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "HOTEL-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (65, 55, 76, 74, 82, 6, 76, 41, 24, 21, 2, 23, 84, 51, 43, 59, 0, 29, 40, 57, 15, 87, 46, 27, 99, 5, 91, 14, 54, 63, 3, 71, 54, 96, 55, 65, 59, 83, 18, 50, 21, 10, 74, 82, 67, 0, 39, 46, 87, 94, 44, 25, 46, 86, 88, 85), (85, 56, 19, 58, 62, 11, 80, 1, 36, 6, 2, 33, 49, 58, 78, 13, 26, 41, 4, 64, 94, 21, 25, 98, 97, 76, 46, 57, 2, 69, 22, 51, 97, 78, 62, 60, 24, 26, 35, 78, 42, 8, 27, 96, 97, 34, 62, 62, 67, 26, 35, 43, 46, 80, 28, 29), (20, 41, 41, 98, 93, 36, 61, 29, 38, 50, 57, 88, 60, 72, 47, 91, 72, 42, 88, 58, 92, 90, 36, 94, 36, 1, 57, 86, 74, 11, 39, 91, 29, 69, 68, 0, 100, 1, 55, 73, 27, 85, 32, 66, 32, 96, 7, 14, 65, 15, 25, 21, 33, 9, 78, 77), (80, 27, 51, 77, 62, 41, 46, 69, 95, 29, 87, 68, 78, 28, 81, 21, 82, 32, 96, 83, 11, 31, 26, 63, 46, 54, 3, 89, 56, 91, 83, 100, 36, 82, 86, 70, 17, 37, 64, 54, 8, 69, 75, 61, 33, 3, 34, 67, 65, 56, 39, 78, 15, 60, 32, 4), (80, 27, 51, 77, 62, 41, 46, 69, 95, 29, 87, 68, 78, 28, 81, 21, 82, 32, 96, 83, 11, 31, 26, 63, 46, 54, 3, 89, 56, 91, 83, 100, 36, 82, 86, 70, 17, 37, 64, 54, 8, 69, 75, 61, 33, 3, 34, 67, 65, 56, 39, 78, 15, 60, 32, 4)], [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (73, 24, 88, 73, 92, 51, 22, 45, 98, 8, 34, 78, 37, 10, 66, 28, 64, 78, 97, 12, 30, 4, 83, 43, 24, 95, 49, 74, 67, 41, 26, 44, 74, 59, 20, 6, 3, 44, 73, 71, 96, 75, 53, 98, 2, 15, 76, 63, 73, 36, 57, 19, 19, 67, 85, 73), (12, 0, 97, 17, 62, 27, 95, 95, 90, 12, 42, 30, 7, 52, 28, 68, 40, 53, 1, 94, 52, 78, 84, 43, 7, 91, 54, 32, 50, 95, 62, 86, 93, 51, 23, 56, 90, 70, 92, 16, 64, 69, 97, 43, 81, 96, 82, 89, 95, 4, 72, 100, 24, 40, 24, 48), (12, 0, 97, 17, 62, 27, 95, 95, 90, 12, 42, 30, 7, 52, 28, 68, 40, 53, 1, 94, 52, 78, 84, 43, 7, 91, 54, 32, 50, 95, 62, 86, 93, 51, 23, 56, 90, 70, 92, 16, 64, 69, 97, 43, 81, 96, 82, 89, 95, 4, 72, 100, 24, 40, 24, 48)], [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (39, 95, 3, 14, 12, 74, 91, 95, 28, 50, 16, 71, 51, 45, 15, 61, 12, 69, 96, 34, 85, 41, 61, 41, 56, 58, 83, 32, 9, 64, 13, 54, 91, 69, 13, 18, 82, 4, 88, 2, 49, 37, 100, 16, 93, 12, 69, 34, 48, 64, 10, 1, 50, 89, 46, 32), (98, 69, 51, 32, 70, 44, 71, 58, 49, 97, 64, 84, 39, 94, 38, 92, 66, 14, 29, 91, 21, 51, 13, 31, 65, 100, 100, 19, 6, 24, 68, 56, 71, 72, 92, 68, 78, 14, 100, 48, 41, 3, 22, 67, 83, 63, 24, 12, 63, 26, 42, 14, 77, 24, 46, 92), (43, 83, 45, 50, 59, 65, 49, 54, 37, 98, 71, 8, 78, 1, 79, 65, 84, 81, 86, 11, 98, 0, 83, 45, 24, 75, 4, 73, 86, 30, 87, 19, 47, 49, 76, 61, 40, 92, 49, 45, 85, 18, 16, 30, 68, 18, 51, 81, 39, 97, 20, 52, 87, 83, 15, 39), (43, 83, 45, 50, 59, 65, 49, 54, 37, 98, 71, 8, 78, 1, 79, 65, 84, 81, 86, 11, 98, 0, 83, 45, 24, 75, 4, 73, 86, 30, 87, 19, 47, 49, 76, 61, 40, 92, 49, 45, 85, 18, 16, 30, 68, 18, 51, 81, 39, 97, 20, 52, 87, 83, 15, 39)], [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (22, 21, 29, 52, 91, 100, 33, 20, 35, 100, 42, 17, 63, 31, 12, 35, 14, 57, 30, 29, 32, 2, 50, 10, 67, 1, 75, 2, 23, 78, 72, 46, 55, 28, 76, 51, 93, 40, 66, 5, 55, 65, 94, 11, 44, 82, 45, 65, 68, 97, 53, 61, 39, 13, 99, 46), (26, 32, 76, 29, 61, 5, 23, 33, 58, 1, 46, 68, 60, 19, 28, 1, 99, 36, 95, 17, 52, 55, 100, 84, 46, 98, 100, 63, 92, 64, 62, 3, 80, 81, 43, 30, 39, 39, 65, 27, 85, 49, 59, 41, 52, 68, 9, 8, 95, 57, 59, 18, 31, 93, 5, 42)], [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (92, 43, 10, 94, 95, 1, 41, 7, 20, 82, 73, 21, 15, 33, 92, 92, 83, 85, 25, 54, 98, 4, 6, 40, 14, 60, 94, 23, 7, 25, 4, 74, 1, 58, 32, 5, 36, 29, 9, 23, 30, 61, 78, 0, 15, 56, 99, 81, 69, 8, 34, 85, 54, 60, 14, 31), (88, 77, 63, 99, 70, 86, 93, 33, 52, 72, 83, 73, 17, 97, 79, 73, 71, 5, 68, 22, 68, 27, 55, 1, 5, 31, 90, 65, 47, 46, 100, 83, 89, 78, 71, 94, 16, 34, 40, 71, 19, 48, 20, 26, 41, 7, 49, 10, 88, 12, 8, 83, 67, 48, 90, 20), (88, 77, 63, 99, 70, 86, 93, 33, 52, 72, 83, 73, 17, 97, 79, 73, 71, 5, 68, 22, 68, 27, 55, 1, 5, 31, 90, 65, 47, 46, 100, 83, 89, 78, 71, 94, 16, 34, 40, 71, 19, 48, 20, 26, 41, 7, 49, 10, 88, 12, 8, 83, 67, 48, 90, 20)], [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (30, 60, 19, 96, 22, 15, 7, 52, 78, 76, 84, 92, 32, 23, 41, 30, 59, 24, 22, 24, 64, 4, 36, 43, 20, 41, 9, 30, 99, 28, 97, 58, 82, 85, 65, 85, 1, 37, 98, 14, 8, 14, 59, 36, 58, 9, 91, 59, 95, 57, 66, 37, 60, 98, 65, 2), (12, 17, 16, 41, 90, 88, 83, 18, 31, 69, 79, 45, 18, 48, 20, 10, 32, 47, 66, 68, 86, 4, 83, 95, 63, 9, 78, 12, 80, 28, 94, 76, 97, 62, 92, 55, 45, 7, 27, 85, 51, 96, 37, 39, 37, 13, 46, 49, 7, 21, 56, 75, 2, 36, 91, 64), (12, 17, 16, 41, 90, 88, 83, 18, 31, 69, 79, 45, 18, 48, 20, 10, 32, 47, 66, 68, 86, 4, 83, 95, 63, 9, 78, 12, 80, 28, 94, 76, 97, 62, 92, 55, 45, 7, 27, 85, 51, 96, 37, 39, 37, 13, 46, 49, 7, 21, 56, 75, 2, 36, 91, 64)], [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (8, 44, 75, 74, 60, 85, 94, 73, 8, 32, 68, 53, 32, 94, 77, 52, 10, 37, 39, 13, 5, 43, 87, 35, 56, 27, 47, 97, 4, 85, 53, 15, 38, 26, 82, 63, 3, 84, 93, 22, 10, 19, 70, 50, 36, 27, 92, 46, 30, 81, 92, 87, 91, 24, 39, 71), (44, 10, 28, 11, 40, 25, 3, 76, 77, 64, 20, 94, 98, 92, 62, 68, 14, 44, 54, 25, 92, 100, 38, 67, 35, 31, 70, 6, 23, 18, 57, 10, 58, 61, 68, 68, 3, 100, 46, 84, 5, 14, 16, 57, 78, 60, 71, 31, 53, 100, 2, 94, 56, 13, 80, 51), (45, 74, 23, 27, 28, 69, 29, 25, 99, 41, 41, 2, 83, 15, 17, 11, 49, 31, 94, 70, 90, 85, 61, 20, 19, 58, 34, 54, 99, 57, 60, 15, 88, 21, 98, 17, 14, 69, 48, 37, 83, 8, 17, 80, 33, 15, 47, 93, 96, 76, 77, 69, 61, 36, 16, 29), (45, 74, 23, 27, 28, 69, 29, 25, 99, 41, 41, 2, 83, 15, 17, 11, 49, 31, 94, 70, 90, 85, 61, 20, 19, 58, 34, 54, 99, 57, 60, 15, 88, 21, 98, 17, 14, 69, 48, 37, 83, 8, 17, 80, 33, 15, 47, 93, 96, 76, 77, 69, 61, 36, 16, 29)], [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (95, 36, 8, 36, 91, 58, 41, 66, 40, 85, 20, 34, 25, 85, 93, 71, 85, 36, 16, 30, 31, 86, 4, 33, 72, 81, 95, 96, 0, 69, 91, 53, 65, 27, 52, 80, 63, 3, 81, 61, 3, 3, 56, 8, 32, 50, 79, 70, 15, 31, 44, 95, 36, 47, 69, 77), (73, 68, 64, 16, 29, 94, 71, 50, 3, 90, 34, 64, 75, 4, 54, 100, 89, 12, 12, 8, 90, 11, 69, 16, 51, 67, 79, 22, 65, 32, 14, 44, 89, 61, 5, 87, 82, 76, 11, 21, 23, 16, 61, 30, 88, 61, 78, 94, 4, 80, 8, 66, 50, 63, 26, 20), (58, 2, 53, 63, 10, 9, 56, 43, 25, 35, 81, 67, 4, 53, 61, 99, 91, 69, 57, 0, 51, 16, 21, 32, 51, 25, 39, 71, 12, 80, 28, 62, 26, 40, 57, 19, 61, 40, 51, 43, 70, 35, 99, 41, 19, 54, 2, 70, 55, 40, 67, 4, 92, 100, 73, 34), (59, 17, 94, 92, 7, 91, 44, 48, 27, 41, 15, 31, 5, 44, 44, 74, 74, 75, 33, 97, 44, 25, 92, 75, 23, 44, 92, 96, 54, 23, 40, 12, 81, 4, 77, 35, 81, 99, 1, 95, 10, 4, 98, 84, 55, 73, 20, 57, 94, 89, 69, 98, 62, 49, 47, 3), (30, 93, 68, 40, 50, 14, 31, 57, 56, 71, 33, 2, 42, 43, 17, 98, 6, 95, 10, 48, 99, 20, 32, 43, 100, 55, 61, 93, 91, 4, 62, 32, 67, 77, 54, 97, 85, 80, 79, 3, 22, 2, 89, 87, 88, 18, 23, 89, 3, 70, 51, 9, 37, 52, 17, 23), (83, 45, 19, 55, 78, 66, 56, 67, 56, 56, 9, 86, 34, 69, 71, 53, 52, 9, 45, 58, 52, 38, 39, 40, 5, 47, 23, 34, 27, 37, 88, 16, 95, 3, 68, 99, 70, 64, 57, 30, 76, 79, 74, 10, 6, 97, 82, 95, 62, 25, 9, 1, 87, 59, 80, 39), (83, 45, 19, 55, 78, 66, 56, 67, 56, 56, 9, 86, 34, 69, 71, 53, 52, 9, 45, 58, 52, 38, 39, 40, 5, 47, 23, 34, 27, 37, 88, 16, 95, 3, 68, 99, 70, 64, 57, 30, 76, 79, 74, 10, 6, 97, 82, 95, 62, 25, 9, 1, 87, 59, 80, 39)], [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (42, 13, 14, 23, 44, 9, 29, 3, 78, 76, 67, 21, 71, 39, 93, 86, 18, 92, 0, 21, 45, 30, 20, 66, 95, 59, 69, 10, 59, 57, 55, 20, 15, 58, 25, 32, 38, 86, 22, 63, 61, 4, 77, 78, 27, 16, 8, 23, 11, 28, 14, 96, 3, 44, 71, 19), (2, 61, 77, 82, 18, 48, 32, 77, 36, 33, 100, 86, 11, 43, 50, 32, 42, 22, 93, 31, 78, 43, 4, 5, 37, 15, 31, 57, 68, 48, 32, 31, 19, 19, 12, 11, 55, 49, 100, 62, 66, 20, 5, 28, 95, 64, 60, 55, 74, 27, 26, 90, 69, 84, 8, 87), (44, 35, 95, 31, 57, 69, 38, 19, 61, 17, 79, 58, 73, 61, 30, 84, 32, 49, 97, 19, 7, 72, 42, 40, 5, 40, 98, 24, 14, 26, 87, 55, 2, 95, 55, 77, 99, 75, 78, 6, 15, 90, 30, 68, 33, 40, 17, 8, 84, 5, 0, 78, 77, 86, 4, 63), (25, 46, 48, 42, 47, 14, 72, 2, 74, 34, 62, 70, 46, 57, 40, 89, 47, 12, 71, 75, 45, 85, 64, 61, 75, 37, 7, 98, 33, 17, 10, 16, 96, 83, 81, 51, 97, 92, 55, 88, 80, 82, 2, 22, 78, 10, 66, 38, 22, 93, 26, 5, 76, 65, 5, 72), (46, 76, 28, 53, 24, 46, 59, 97, 40, 12, 54, 38, 77, 39, 70, 11, 82, 99, 97, 54, 82, 27, 86, 84, 4, 12, 39, 24, 37, 38, 53, 57, 80, 31, 71, 48, 39, 97, 7, 92, 52, 98, 71, 65, 12, 4, 24, 87, 73, 23, 45, 92, 43, 18, 56, 27), (46, 76, 28, 53, 24, 46, 59, 97, 40, 12, 54, 38, 77, 39, 70, 11, 82, 99, 97, 54, 82, 27, 86, 84, 4, 12, 39, 24, 37, 38, 53, 57, 80, 31, 71, 48, 39, 97, 7, 92, 52, 98, 71, 65, 12, 4, 24, 87, 73, 23, 45, 92, 43, 18, 56, 27)], [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (21, 22, 13, 14, 25, 10, 100, 23, 18, 15, 88, 0, 50, 28, 50, 74, 93, 9, 22, 93, 79, 86, 64, 21, 25, 68, 81, 39, 10, 78, 37, 92, 8, 77, 72, 86, 14, 39, 55, 56, 68, 87, 53, 71, 68, 80, 76, 28, 85, 14, 64, 46, 91, 72, 81, 22), (66, 21, 1, 39, 93, 8, 51, 27, 83, 64, 48, 46, 85, 92, 44, 11, 100, 96, 97, 91, 94, 60, 69, 68, 2, 80, 35, 56, 96, 22, 6, 58, 81, 7, 16, 1, 69, 99, 40, 25, 70, 2, 86, 91, 88, 54, 89, 68, 49, 50, 33, 14, 31, 86, 78, 44), (66, 21, 1, 39, 93, 8, 51, 27, 83, 64, 48, 46, 85, 92, 44, 11, 100, 96, 97, 91, 94, 60, 69, 68, 2, 80, 35, 56, 96, 22, 6, 58, 81, 7, 16, 1, 69, 99, 40, 25, 70, 2, 86, 91, 88, 54, 89, 68, 49, 50, 33, 14, 31, 86, 78, 44)]]"
        self.expected_all_intermediate_budgets = "[[0, 20, 30, 40, 70, 100], [0, 20, 60, 100], [0, 20, 40, 90, 100], [0, 20, 100], [0, 20, 30, 100], [0, 20, 70, 100], [0, 20, 30, 90, 100], [0, 20, 30, 40, 60, 70, 80, 100], [0, 20, 30, 40, 60, 70, 100], [0, 20, 60, 100]]"
        self.expected_all_est_objectives = "[[0.0, 36010.5, 34949.0, 37327.5, 39544.0, 39544.0], [0.0, 38333.0, 39598.5, 39598.5], [0.0, 36170.5, 37515.0, 39309.5, 39309.5], [0.0, 36749.5, 39220.0], [0.0, 31064.0, 38573.5, 38573.5], [0.0, 36713.5, 37499.0, 37499.0], [0.0, 33870.0, 34293.5, 39030.5, 39030.5], [0.0, 33470.0, 33121.5, 34671.5, 36066.0, 35533.5, 40357.5, 40357.5], [0.0, 30089.0, 33001.0, 34389.0, 37961.5, 38593.0, 38593.0], [0.0, 36859.0, 38692.0, 38692.0]]"
        self.expected_objective_curves = "[([0, 20, 30, 40, 70, 100], [0.0, 36010.5, 34949.0, 37327.5, 39544.0, 39544.0]), ([0, 20, 60, 100], [0.0, 38333.0, 39598.5, 39598.5]), ([0, 20, 40, 90, 100], [0.0, 36170.5, 37515.0, 39309.5, 39309.5]), ([0, 20, 100], [0.0, 36749.5, 39220.0]), ([0, 20, 30, 100], [0.0, 31064.0, 38573.5, 38573.5]), ([0, 20, 70, 100], [0.0, 36713.5, 37499.0, 37499.0]), ([0, 20, 30, 90, 100], [0.0, 33870.0, 34293.5, 39030.5, 39030.5]), ([0, 20, 30, 40, 60, 70, 80, 100], [0.0, 33470.0, 33121.5, 34671.5, 36066.0, 35533.5, 40335.0, 40335.0]), ([0, 20, 30, 40, 60, 70, 100], [0.0, 30089.0, 33001.0, 34389.0, 37961.5, 38593.0, 38593.0]), ([0, 20, 60, 100], [0.0, 36859.0, 38692.0, 38692.0])]"
        self.expected_progress_curves = "[([0.0, 0.2, 0.3, 0.4, 0.7, 1.0], [1.0, 0.10721457791000372, 0.13353167224494855, 0.07456303458534771, 0.019610759885955127, 0.019610759885955127]), ([0.0, 0.2, 0.6, 1.0], [1.0, 0.049634312631709435, 0.01825957605057642, 0.01825957605057642]), ([0.0, 0.2, 0.4, 0.9, 1.0], [1.0, 0.10324779967769927, 0.06991446634436593, 0.025424569232676334, 0.025424569232676334]), ([0.0, 0.2, 1.0], [1.0, 0.08889302094954754, 0.02764348580637164]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.22985000619809098, 0.04367174910127681, 0.04367174910127681]), ([0.0, 0.2, 0.7, 1.0], [1.0, 0.08978554605181605, 0.07031114416759639, 0.07031114416759639]), ([0.0, 0.2, 0.3, 0.9, 1.0], [1.0, 0.16028263294905168, 0.14978306681542086, 0.03234163877525722, 0.03234163877525722]), ([0.0, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 1.0], [1.0, 0.17019957852981282, 0.17883971736705095, 0.1404115532416016, 0.10583860171067311, 0.11904053551506136, -0.0, -0.0]), ([0.0, 0.2, 0.3, 0.4, 0.6, 0.7, 1.0], [1.0, 0.25402256105119625, 0.18182719722325524, 0.14741539605801413, 0.05884467583984133, 0.043188298004214705, 0.043188298004214705]), ([0.0, 0.2, 0.6, 1.0], [1.0, 0.08617825709681418, 0.040733853972976325, 0.040733853972976325])]"

        # Convert the expected values from string to their actual types
        self.expected_all_recommended_xs = eval(self.expected_all_recommended_xs, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_intermediate_budgets = eval(self.expected_all_intermediate_budgets, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_est_objectives = eval(self.expected_all_est_objectives, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_objective_curves = eval(self.expected_objective_curves, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_progress_curves = eval(self.expected_progress_curves, {'nan': float('nan'), 'inf': float('inf')})
        
        # Number of macro-replications and post-replications
        self.num_macroreps = 10
        self.num_postreps = 200

        # Setup the solver and experiment
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.assertEqual(self.myexperiment.solver.name, self.expected_solver_name, "Solver name does not match (expected: " + self.expected_solver_name + ", actual: " + self.myexperiment.solver.name + ")")
        self.assertEqual(self.myexperiment.problem.name, self.expected_problem_name, "Problem name does not match (expected: " + self.expected_problem_name + ", actual: " + self.myexperiment.problem.name + ")")

    def test_run(self):
        # Check actual run results against expected
        self.myexperiment.run(n_macroreps=self.num_macroreps)
        self.assertEqual(self.myexperiment.n_macroreps, self.num_macroreps, "Number of macro-replications for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_recommended_xs[mrep]), len(self.expected_all_recommended_xs[mrep]), "Length of recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list of recommended solutions
            for list in range(len(self.myexperiment.all_recommended_xs[mrep])):
                # Check to make sure the tuples are the same length
                self.assertEqual(len(self.myexperiment.all_recommended_xs[mrep][list]), len(self.expected_all_recommended_xs[mrep][list]), "Recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
                # For each tuple of recommended solutions
                for tuple in range(len(self.myexperiment.all_recommended_xs[mrep][list])):
                    self.assertAlmostEqual(self.myexperiment.all_recommended_xs[mrep][list][tuple], self.expected_all_recommended_xs[mrep][list][tuple], 5, "Recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + " and tuple " + str(tuple) + ".")
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_intermediate_budgets[mrep]), len(self.expected_all_intermediate_budgets[mrep]), "Length of intermediate budgets for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list of intermediate budgets
            for list in range(len(self.myexperiment.all_intermediate_budgets[mrep])):
                # Check the values in the list
                self.assertAlmostEqual(self.myexperiment.all_intermediate_budgets[mrep][list], self.expected_all_intermediate_budgets[mrep][list], 5, "Intermediate budgets for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
            
    def test_post_replicate(self):
        # Simulate results from the run method
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.myexperiment.n_macroreps = self.num_macroreps
        self.myexperiment.all_recommended_xs = self.expected_all_recommended_xs
        self.myexperiment.all_intermediate_budgets = self.expected_all_intermediate_budgets

        # Check actual post-replication results against expected
        self.myexperiment.post_replicate(n_postreps=self.num_postreps)
        self.assertEqual(self.myexperiment.n_postreps, self.num_postreps, "Number of post-replications for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_est_objectives[mrep]), len(self.expected_all_est_objectives[mrep]), "Estimated objectives for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list in the estimated objectives
            for list in range(len(self.myexperiment.all_est_objectives[mrep])):
                # Check the values in the list
                self.assertAlmostEqual(self.myexperiment.all_est_objectives[mrep][list], self.expected_all_est_objectives[mrep][list], 5, "Estimated objectives for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")

    def test_post_normalize(self):
        # Simulate results from the post_replicate method
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.myexperiment.n_macroreps = self.num_macroreps
        self.myexperiment.n_postreps = self.num_postreps
        self.myexperiment.all_recommended_xs = self.expected_all_recommended_xs
        self.myexperiment.all_intermediate_budgets = self.expected_all_intermediate_budgets
        self.myexperiment.all_est_objectives = self.expected_all_est_objectives

        # Check actual post-normalization results against expected
        post_normalize([self.myexperiment], n_postreps_init_opt=self.num_postreps)

        # Loop through each curve object and convert it into a tuple
        for i in range(len(self.myexperiment.objective_curves)):
            self.myexperiment.objective_curves[i] = (self.myexperiment.objective_curves[i].x_vals, self.myexperiment.objective_curves[i].y_vals)
        for i in range(len(self.myexperiment.progress_curves)):
            self.myexperiment.progress_curves[i] = (self.myexperiment.progress_curves[i].x_vals, self.myexperiment.progress_curves[i].y_vals)

        for mrep in range(self.num_macroreps):
            # Check to make sure the same number of objective curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(len(self.myexperiment.objective_curves[mrep]), len(self.expected_objective_curves[mrep]), "Number of objective curves for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
            # Make sure that curves are only checked if they exist
            if (len(self.myexperiment.objective_curves[mrep]) > 0):
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(len(self.myexperiment.objective_curves[mrep][0]), len(self.expected_objective_curves[mrep][0]), "Length of X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                self.assertEqual(len(self.myexperiment.objective_curves[mrep][1]), len(self.expected_objective_curves[mrep][1]), "Length of Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                # Check X (0) and Y (1) values
                for x_index in range(len(self.myexperiment.objective_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.objective_curves[mrep][0][x_index])):
                        self.assertTrue(math.isnan(self.expected_objective_curves[mrep][0][x_index]), "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.objective_curves[mrep][0][x_index], self.expected_objective_curves[mrep][0][x_index], 5, "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                for y_index in range(len(self.myexperiment.objective_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.objective_curves[mrep][1][y_index])):
                        self.assertTrue(math.isnan(self.expected_objective_curves[mrep][1][y_index]), "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.objective_curves[mrep][1][y_index], self.expected_objective_curves[mrep][1][y_index], 5, "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
            
            # Check to make sure the same number of progress curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(len(self.myexperiment.progress_curves[mrep]), len(self.expected_progress_curves[mrep]), "Number of progress curves for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
            # Make sure that curves are only checked if they exist
            if (len(self.myexperiment.progress_curves[mrep]) > 0):
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(len(self.myexperiment.progress_curves[mrep][0]), len(self.expected_progress_curves[mrep][0]), "Length of X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                self.assertEqual(len(self.myexperiment.progress_curves[mrep][1]), len(self.expected_progress_curves[mrep][1]), "Length of Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                # Check X (0) and Y (1) values
                for x_index in range(len(self.myexperiment.progress_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.progress_curves[mrep][0][x_index])):
                        self.assertTrue(math.isnan(self.expected_progress_curves[mrep][0][x_index]), "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.progress_curves[mrep][0][x_index], self.expected_progress_curves[mrep][0][x_index], 5, "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                for y_index in range(len(self.myexperiment.progress_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.progress_curves[mrep][1][y_index])):
                        self.assertTrue(math.isnan(self.expected_progress_curves[mrep][1][y_index]), "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.progress_curves[mrep][1][y_index], self.expected_progress_curves[mrep][1][y_index], 5, "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")      
