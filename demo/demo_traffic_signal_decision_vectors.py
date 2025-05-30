"""Demo model for experimenting with traffic signal decision vectors.

This script runs a series of experiments on a traffic signal model,
varying the decision vector multiplier to observe its impact on average wait times.
"""

# Import standard libraries
import logging
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from simopt.base import Model

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import modules that might depend on the simopt package
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.models.trafficsignal import TrafficLight
from simopt.utils import print_table

# #####################################################################################
# USER CONFIGURATION:
# To modify model parameters and experiment settings,
# please edit the `get_config()` function located below.
# #####################################################################################


def get_config() -> dict[str, Any]:
    """User-configurable parameters for the data farming experiment.

    Modify these values to change the model's behavior.
    """
    # Import the model class to be demo'd, using the following format:
    # from simopt.models.<filename> import <model_class_name>
    # Since this is a module import, you do not need to include the .py extension.

    return {
        # --- Experiment Settings ---
        # Minimum scale factor for the decision vector.
        "decision_vector_min": 1.0,
        # Maximum scale factor for the decision vector.
        "decision_vector_max": 30.0,
        # Step size for the decision vector multiplier.
        "decision_vector_step": 0.25,
        # Runtime for the traffic signal model in seconds.
        "runtime": 1200,
        # Number of simulation runs (macroreplications) for the same factors.
        # Each macroreplication uses a different random number stream.
        # Must be a positive integer.
        "num_macroreps": 5,
        # Enable debug-level logging for detailed model information.
        # Set to True for debugging, False for normal operation.
        "debug_logging": False,
    }


# #####################################################################################
# Main Script Execution:
# This section contains the core logic and should typically NOT be modified by users.
# All user-specific settings are handled in the `get_config()` function above.
# #####################################################################################


def main() -> None:
    """Main function to run the data farming experiment."""
    # Fetch the configuration settings.
    config = get_config()
    num_macroreps = config["num_macroreps"]
    debug_logging = config["debug_logging"]
    decision_vector_min = config["decision_vector_min"]
    decision_vector_max = config["decision_vector_max"]
    decision_vector_step = config["decision_vector_step"]
    runtime = config["runtime"]

    # Print the configuration settings as a table.
    config_header = ["Parameter", "Value"]
    config_tuples = [
        ("Number of Macroreplications", num_macroreps),
        ("Debug Logging Enabled", debug_logging),
        ("Decision Vector Min", decision_vector_min),
        ("Decision Vector Max", decision_vector_max),
        ("Decision Vector Step", decision_vector_step),
    ]
    print_table("Configuration Settings", config_header, config_tuples)

    # Set the logging level for the model.
    log_level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Set fixed factors for the traffic signal.
    # NOTE: decision_vector_mult will be overridden by the loop below.
    fixed_factors: dict[str, Any] = {"runtime": runtime}

    exp_avg_wait_list: list[list[float]] = []

    # Run each experiment with a different decision vector multiplier.
    import numpy as np

    step_range = np.arange(
        decision_vector_min,
        decision_vector_max + decision_vector_step,
        decision_vector_step,
    )
    for exp_idx, mult in enumerate(step_range, start=1):
        print(f"\n> Running experiment {exp_idx} with vector multiplier {mult}...")

        fixed_factors["decision_vector"] = [float(mult)] * 3

        # Initialize the model with the specified fixed factors and check if it
        # is simulatable. If there is an error or the model is not simulatable,
        # exit the script gracefully.
        try:
            print("> Fixed Factors:")
            # Print each fixed factor in a readable format.
            for key, value in fixed_factors.items():
                print(f"  > {key}: {value}")

            # Initialize the model with the fixed factors.
            print("> Initializing TrafficLight model...", end=" ", flush=True)
            mymodel = TrafficLight(fixed_factors)
            print("success!")

            print("> Checking if model is simulatable...", end=" ", flush=True)
            is_model_simulatable = mymodel.check_simulatable_factors()
            # Exit if not simulatable.
            if not is_model_simulatable:
                raise ValueError("Model is not simulatable with the specified factors.")
            print("success!")

        except Exception as e:
            print(f"> Error: {e}")
            print("> Exiting script.")
            return

        # Create a list of RNG objects for the simulation model to use when
        # running replications.
        # Start with the same RNG for each experiment to keep things consistent.

        # Keep track of responses and gradients between macroreplications so they can be
        # compared/graphed at the end.
        avg_wait_by_mrep: dict[int, float] = {}

        run_mrep_partial = partial(run_macroreplication, mymodel=mymodel)

        num_processes = min(num_macroreps, os.cpu_count() or 1)
        with Pool(num_processes) as process_pool:
            print(
                f"> Running {num_macroreps} macroreplications in parallel "
                f"using {num_processes} processes..."
            )
            # Use a pool of processes to run macroreplications in parallel.
            for mrep, wait_time in process_pool.imap_unordered(
                run_mrep_partial, range(1, num_macroreps + 1)
            ):
                print(f"> mrep {mrep} - AvgWaitTime: {wait_time:.2f} seconds")
                avg_wait_by_mrep[mrep] = wait_time

        print(f"> Finished macroreplications for experiment {exp_idx}.")

        # Store the average wait times for this experiment.
        exp_wait_list = [avg_wait_by_mrep[mrep] for mrep in range(1, num_macroreps + 1)]
        exp_avg_wait_list.append(exp_wait_list)

    # Switch back to normal logging level.
    logging.getLogger().setLevel(logging.INFO)

    # Calculate avg waits once here to use for both printing and plotting.
    avg_waits = [
        sum(exp_wait_list) / len(exp_wait_list) for exp_wait_list in exp_avg_wait_list
    ]

    # Print out the average wait times for each experiment.
    print("\n> Average Wait Times by Experiment:")
    for idx, exp_wait_list in enumerate(exp_avg_wait_list, start=1):
        avg_wait = avg_waits[idx - 1]
        print(f"  Experiment {idx}: {avg_wait:.2f} seconds ", end="")
        # Round the wait times to 2 decimal places for readability.
        waits = ", ".join(f"{wait:.2f}" for wait in exp_wait_list)
        print(f"[{waits}]")
    # Plot the average wait times.
    plt.figure(figsize=(10, 6))
    plt.plot(
        step_range,
        avg_waits,
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


def run_macroreplication(mrep: int, mymodel: Model) -> tuple[int, float]:
    """Function to run a single macroreplication."""
    rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]
    for _ in range(mrep - 1):
        for rng in rng_list:
            rng.advance_subsubstream()

    responses, _ = mymodel.replicate(rng_list)
    # Record the average wait for this index.
    wait_time: float = responses["AvgWaitTime"]
    return mrep, wait_time


if __name__ == "__main__":
    main()
