"""Demo model for experimenting with traffic signal decision vectors.

This script runs a series of experiments on a traffic signal model,
varying the decision vector multiplier to observe its impact on average wait times.
"""

# Import standard libraries
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import modules that might depend on the simopt package
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.models.trafficsignal import TrafficLight
from simopt.utils import print_table

# #####################################################################################
# USER CONFIGURATION:
# #####################################################################################

# Minimum scale factor for the decision vector.
DECISION_VECTOR_MIN = 21
# Maximum scale factor for the decision vector.
DECISION_VECTOR_MAX = 21
# Step size for the decision vector multiplier.
DECISION_VECTOR_STEP = 0.25
# Runtime for the traffic signal model in seconds.
RUNTIME = 1200
# Number of simulation runs (macroreplications) for the same factors.
# Each macroreplication uses a different random number stream.
# Must be a positive integer.
NUM_MACROREPS = 5
# Number of extra RNG advancements before starting the macroreplications.
# This can be useful to simulate later macroreplications without having to run
# the earlier ones since each mrep advances the RNG by 1.
# EG: If previously running 10 mreps and you want to rerun just 6-10, you can
# set num_macroreps to 5 and extra_rng_advancements to 5.
# Each mrep is independent, so not running the first macroreps will not
# impact the results of the later ones.
EXTRA_RNG_ADVANCEMENTS = 5
# Whether to print the responses from each macroreplication.
PRINT_RESPONSES = True

# #####################################################################################
# Main Script Execution:
# This section contains the core logic and should typically NOT be modified by users.
# All user-specific settings are handled in the configuration section above.
# #####################################################################################


class DecisionVectorExperiment:
    """Class to encapsulate the decision vector experiment."""

    @property
    def avg_wait(self) -> float:
        """Calculate the average wait time across all experiments."""
        return (
            sum(self._avg_waits.values()) / len(self._avg_waits)
            if self._avg_waits
            else 0.0
        )

    @property
    def avg_waits(self) -> dict[int, float]:
        """Calculate the average wait times across all experiments."""
        return self._avg_waits

    @avg_waits.setter
    def avg_waits(self, value: dict[int, float]) -> None:
        """Set the average wait times for the experiment."""
        self._avg_waits = value

    @property
    def avg_waits_list(self) -> list[float]:
        """Return the average wait times as a list."""
        return [self._avg_waits[mrep] for mrep in sorted(self._avg_waits.keys())]

    def __init__(
        self,
        exp_idx: int,
        decision_val: float,
        fixed_factors: dict[str, Any],
        num_macroreps: int,
        extra_rng_advancements: int,
    ) -> None:
        """Initialize the experiment with configuration settings."""
        # Setup fixed factors
        self.fixed_factors = fixed_factors.copy()
        self.fixed_factors["decision_vector"] = [decision_val] * 3
        # Create the model instance with the fixed factors.
        self.mymodel = TrafficLight(self.fixed_factors)
        if not self.mymodel.check_simulatable_factors():
            raise ValueError(
                "Model is not simulatable with the specified fixed factors."
            )

        # Set additional attributes for the experiment.
        self.exp_idx = exp_idx
        self.num_macroreps = num_macroreps
        self.extra_rng_advancements = extra_rng_advancements

        # Create empty variables for wait times.
        self._avg_wait = 0.0  # Average wait time across all macroreplications.
        self._avg_waits: dict[int, float] = {}  # List of average wait times

    def _run_macroreplication(self, mrep: int) -> tuple[int, float]:
        """Function to run a single macroreplication."""
        rng_list = [
            MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(self.mymodel.n_rngs)
        ]
        for _ in range(mrep + self.extra_rng_advancements):
            for rng in rng_list:
                rng.advance_subsubstream()

        responses, _ = self.mymodel.replicate(rng_list)
        if PRINT_RESPONSES:
            non_dict_responses = {
                key: value
                for key, value in responses.items()
                if not isinstance(value, dict)
            }
            print_table(
                f"Responses - mrep {mrep + 1}",
                ["Response", "Value"],
                non_dict_responses,
            )
        return mrep, responses["AvgWaitTime"]


def main() -> None:
    """Main function to run the data farming experiment."""
    # Print the configuration settings.
    config_header = ["Parameter", "Value"]
    # NOTE: Make sure all the configurable parameters are included in the config.
    config = {
        "decision_vector_min": DECISION_VECTOR_MIN,
        "decision_vector_max": DECISION_VECTOR_MAX,
        "decision_vector_step": DECISION_VECTOR_STEP,
        "runtime": RUNTIME,
        "num_macroreps": NUM_MACROREPS,
        "extra_rng_advancements": EXTRA_RNG_ADVANCEMENTS,
    }
    print_table("Configuration Settings", config_header, config)

    # Set fixed factors for the traffic signal.
    # NOTE: decision_vector_mult will be overridden by the loop below.
    fixed_factors: dict[str, Any] = {"runtime": RUNTIME}
    experiments: list[DecisionVectorExperiment] = []

    # Run each experiment with a different decision vector multiplier.
    import numpy as np

    step_range = np.arange(
        DECISION_VECTOR_MIN,
        DECISION_VECTOR_MAX + DECISION_VECTOR_STEP,
        DECISION_VECTOR_STEP,
    )

    # Create the list of experiments with different decision vector multipliers.
    for idx, mult in enumerate(step_range, start=1):
        try:
            # Create a new experiment instance with the current multiplier.
            experiment = DecisionVectorExperiment(
                exp_idx=idx,
                decision_val=mult,
                fixed_factors=fixed_factors,
                num_macroreps=NUM_MACROREPS,
                extra_rng_advancements=EXTRA_RNG_ADVANCEMENTS,
            )
            experiments.append(experiment)
        except Exception as e:
            print(f"> Error creating experiment {idx} with multiplier {mult}: {e}")
            print("> Exiting script.")
            return

    # Loop through each experiment and run the macroreplications.
    for exp in experiments:
        mult = exp.fixed_factors["decision_vector"][0]
        print(f"\n> Running experiment {exp.exp_idx} with multiplier {mult:.2f} ")

        num_processes = min(NUM_MACROREPS, os.cpu_count() or 1)
        with Pool(num_processes) as process_pool:
            print(
                f"> Running {NUM_MACROREPS} macroreplications in parallel "
                f"using {num_processes} processes..."
            )
            # Use a pool of processes to run macroreplications in parallel.
            for mrep, wait_time in process_pool.imap_unordered(
                exp._run_macroreplication, range(NUM_MACROREPS)
            ):
                # Store the average wait time for each macroreplication.
                exp.avg_waits[mrep] = wait_time
                print(f"> mrep {mrep + 1} - AvgWaitTime: {wait_time:.2f} seconds")

    # Print out the average wait times for each experiment.
    print("\n> Average Wait Time Summary:")
    for exp in experiments:
        exp_idx = exp.exp_idx
        avg_wait = exp.avg_wait
        avg_waits = exp.avg_waits_list
        print(
            f"Experiment {exp_idx}: AvgWaitTime = {avg_wait:.2f} seconds "
            f"[{', '.join(f'{wait:.2f}' for wait in avg_waits)}]"
        )

    avg_waits_graph = [exp.avg_wait for exp in experiments]

    # Plot the average wait times.
    plt.figure(figsize=(10, 6))
    plt.plot(
        step_range,
        avg_waits_graph,
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.title("Average Wait Time by Decision Vector Multiplier")
    plt.xlabel("Decision Vector Multiplier")
    plt.ylabel("Average Wait Time (seconds)")
    plt.xticks(
        step_range,
        rotation=45,
    )
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
