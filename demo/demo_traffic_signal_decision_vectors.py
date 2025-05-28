"""Demo model for experimenting with traffic signal decision vectors.

This script runs a series of experiments on a traffic signal model,
varying the decision vector multiplier to observe its impact on average wait times.
"""

# Import standard libraries
import logging
import sys
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
    fixed_factors: dict[str, Any] = {"runtime": 1200}

    avg_wait_by_exp: list[float] = []

    # Run each experiment with a different decision vector multiplier.
    import numpy as np

    for exp_idx, mult in enumerate(
        np.arange(
            decision_vector_min,
            decision_vector_max + decision_vector_step,
            decision_vector_step,
        ),
        start=1,
    ):
        print(f"\n> Running experiment {exp_idx} with vector multiplier {mult}...")

        fixed_factors["decision_vector"] = [float(mult)] * 3

        # Initialize the model with the specified fixed factors.
        # If the fixed factors are invalid, the model will raise an error.
        try:
            print("> Initializing model with fixed factors:")
            for key, value in fixed_factors.items():
                print(f"\t{key}: {value}")
            mymodel = TrafficLight(fixed_factors)
        except Exception as e:
            print(f"> Error initializing model: {e}")
            print("> Exiting script.")
            return

        # Check all factors collectively.
        try:
            is_model_simulatable = mymodel.check_simulatable_factors()
            model_sim_str = "IS" if is_model_simulatable else "IS NOT"
            print(
                f"> {mymodel.name} {model_sim_str} simulatable with specified factors."
            )
            # Exit if not simulatable.
            if not is_model_simulatable:
                print("> Exiting script.")
                return
        except Exception as e:
            print(f"> Error checking model simulation: {e}")
            print("> Exiting script.")
            return

        # Create a list of RNG objects for the simulation model to use when
        # running replications.
        # Start with the same RNG for each experiment to keep things consistent.
        rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]

        # Keep track of responses and gradients between macroreplications so they can be
        # compared/graphed at the end.
        avg_wait_by_mrep: list = []

        # Run a single replication of the model.
        for mrep in range(1, num_macroreps + 1):
            print(
                f"> Running macroreplication {mrep} of {num_macroreps}...",
                end="",
                flush=True,
            )
            responses, _ = mymodel.replicate(rng_list)
            print(" done.")
            # Record the average wait for this index.
            wait_time = responses.get("AvgWaitTime", None)
            avg_wait_by_mrep.append(wait_time)
            print(f"> Average wait time for mrep {mrep}: {wait_time:.2f} seconds")

            # Advance RNG
            for rng in rng_list:
                rng.advance_subsubstream()
        print(f"> Finished macroreplications for experiment {exp_idx}.")

        # Store the average wait times for this experiment.
        avg_wait = sum(avg_wait_by_mrep) / len(avg_wait_by_mrep)
        avg_wait_by_exp.append(avg_wait)
        print(f"> Average wait time for experiment {exp_idx}: {avg_wait:.2f} seconds")

    # Switch back to normal logging level.
    logging.getLogger().setLevel(logging.INFO)

    # Print out the average wait times for each experiment.
    print("\n> Average Wait Times by Experiment:")

    for idx, avg_wait in enumerate(avg_wait_by_exp, start=1):
        print(f"  Experiment {idx}: {avg_wait:.2f} seconds")
    # Plot the average wait times.
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(avg_wait_by_exp) + 1),
        avg_wait_by_exp,
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.title("Average Wait Times by Experiment")
    plt.xlabel("Experiment Index")
    plt.ylabel("Average Wait Time (seconds)")
    plt.xticks(range(1, len(avg_wait_by_exp) + 1))
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
