import numpy as np
import pandas as pd
from pydoe import lhs

# =================================================================================================
# Function for constructing a DataFrame from a matrix with floating point numbers between 0 and 1
# =================================================================================================


def construct_df_from_random_matrix(
    x: np.ndarray,
    factor_array: np.ndarray,
) -> pd.DataFrame:
    """
    This function constructs a DataFrame out of matrix x and factor_array,
    both of which are assumed to be numpy arrays.
    It projects the numbers in x (which is output of a design-of-experiment
    build) to the factor array ranges.
    Here factor_array is assumed to have only min and max ranges.
    Matrix x is assumed to have numbers ranging from 0 to 1 only.
    """

    row_num = x.shape[0]  # Number of rows in the matrix x
    col_num = x.shape[1]  # Number of columns in the matrix x

    empty = np.zeros((row_num, col_num))

    def simple_substitution(idx: float, factor_list: np.ndarray) -> float:
        alpha = np.abs(factor_list[1] - factor_list[0])
        beta = idx
        return factor_list[0] + (beta * alpha)

    for i in range(row_num):
        for j in range(col_num):
            empty[i, j] = simple_substitution(x[i, j], factor_array[j])

    return pd.DataFrame(data=empty)


# ====================================================================================
# Function for building simple Latin Hypercube from a dictionary of process variables
# ====================================================================================


def build_lhs(
    factor_level_ranges: dict[int | str, list[int | float]],
    num_samples: int | None = None,
    prob_distribution: str | None = None,
) -> pd.DataFrame:
    """
    Builds a Latin Hypercube design dataframe from a dictionary of factor/level ranges.
    Only min and max values of the range are required.
    Example of the dictionary which is needed as the input:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    num_samples: Number of samples to be generated
    prob_distribution: Analytical probability distribution to be applied over
        the randomized sampling.
        Takes strings like: 'Normal', 'Poisson', 'Exponential', 'Beta', 'Gamma'

        Latin hypercube sampling (LHS) is a form of stratified sampling that can
        be applied to multiple variables. The method commonly used to reduce the
        number or runs necessary for a Monte Carlo simulation to achieve a
        reasonably accurate random distribution. LHS can be incorporated into an
        existing Monte Carlo model fairly easily, and work with variables
        following any analytical probability distribution.
    """
    _ = prob_distribution

    for key in factor_level_ranges:
        if len(factor_level_ranges[key]) != 2:
            factor_level_ranges[key][1] = factor_level_ranges[key][-1]
            factor_level_ranges[key] = factor_level_ranges[key][:2]
            print(f"{key} had more than two levels. Assigning the end point to the high level.")

    factor_count = len(factor_level_ranges)
    factor_lists = []

    if num_samples is None:
        num_samples = factor_count

    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])

    x = lhs(n=factor_count, samples=num_samples)
    factor_lists = np.array(factor_lists)

    df = construct_df_from_random_matrix(x, factor_lists)
    return df.set_axis(list(factor_level_ranges.keys()), axis=1)
