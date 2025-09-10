"""Testing with SciPy's LatinHypercube."""

import numpy as np
import scipy.stats.qmc as qmc


def main() -> None:
    """Test SciPy's LatinHypercube."""
    seed = 0
    # For strength=2, n must be a perfect square of a prime.
    prime = 5
    num_samples = prime**2
    # For strength=2, d <= p + 1.
    # num_variables = prime + 1
    num_variables = 5
    print(f"Samples: {num_samples}")
    print(f"Variables: {num_variables}")

    sampler_rng = np.random.default_rng(seed=seed)
    # The 'd' parameter is the number of dimensions/variables.
    # The 'n' parameter is the number of samples.
    sampler = qmc.LatinHypercube(d=num_variables, strength=2, rng=sampler_rng)
    sample = sampler.random(n=num_samples)
    # print("Sampled points:")
    # print(sample)
    print(f"Discrepancy: {qmc.discrepancy(sample)}")

    # print()

    # min_val = 0
    # max_val = 5
    # num_decimals = 1
    # scaled_sample = qmc.scale(sample, l_bounds=min_val, u_bounds=max_val)
    # rounded_sample = np.round(scaled_sample, decimals=num_decimals)
    # # Rounding and clipping the sample.
    # rounded_sample = np.round(scaled_sample, decimals=num_decimals)
    # rounded_sample = np.clip(rounded_sample, min_val, max_val)

    # print(f"Scaled and rounded to {num_decimals} decimal(s):")
    # print(rounded_sample)


if __name__ == "__main__":
    main()
