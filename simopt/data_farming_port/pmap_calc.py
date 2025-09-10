"""Calculator for PMAP metrics.

PMAP = maximum absolute pairwise correlation (lower = better).

Based on the following paper:
https://doi.org/10.1109/WSC.2012.6465112
"""


def calc_nmap(n: int, k: int) -> float:
    """Calculate the NMAP metric."""
    n_23 = n ** (-2 / 3)
    k_13 = k ** (-1 / 3)
    return 0.0873 + 7.859 * n_23 - 0.109 * k_13 - 11.702 * n_23 * k_13


def main() -> None:
    """Main method stuff."""
    # var_options = [7, 11, 16, 22, 29, 37, 46, 56, 67, 79, 106, 121, 137, 154, 172]
    # for sample_exp in range(4, 10):
    #     num_samples = 2**sample_exp

    #     for num_variables in var_options:
    #         if num_variables >= num_samples:
    #             continue
    #         pmap = calc_nmap(num_samples, num_variables)
    #         print(num_samples, num_variables, pmap, sep=",")

    for sample_exp in range(1, 10):
        num_samples = 2**sample_exp

        for num_variables in range(1, sample_exp + 1):
            pmap = calc_nmap(num_samples, num_variables)
            print(num_samples, num_variables, pmap, sep=",")


if __name__ == "__main__":
    main()
