"""Testing with Ruby's LatinHypercube."""

import nolhs_designs as nolhs
import numpy as np
import scipy.stats.qmc as qmc


def main() -> None:
    """Test with the Ruby Datafarming LatinHypercube."""
    table_keys = nolhs.DESIGN_TABLE.keys()
    for key in table_keys:
        print(f"Samples: {key}")
        print(f"Variables: {len(nolhs.DESIGN_TABLE[key][0])}")
        sample = nolhs.DESIGN_TABLE[key]

        # Scale to unit hypercube
        scaled_sample = (np.array(sample) - 1) / int(key)
        discrepancy = qmc.discrepancy(scaled_sample)
        print(f"\tdiscrepancy: {discrepancy}")


if __name__ == "__main__":
    main()
