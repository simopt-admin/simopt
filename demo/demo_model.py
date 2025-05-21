"""Demo for Model Debugging.

This script is intended to help with debugging a model.
It imports a model, initializes a model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

# Import standard libraries
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import modules that might depend on the simopt package
from mrg32k3a.mrg32k3a import MRG32k3a
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

    from simopt.models.trafficsignal import TrafficLight

    return {
        # --- Model Class ---
        # Set the model class to be demo'd.
        # Ensure it's imported above.
        "model_class": TrafficLight,
        # --- Model Factors ---
        # Define the fixed factors for the model.
        # To use default values for a factor, omit it from this dictionary.
        # For available factors and their types, check the model's documentation.
        "fixed_factors": {},
        # If 'fixed_factors' dictionary is left empty or set to None,
        # the model will use its default values for all factors.
        # --- Experiment Settings ---
        # Number of simulation runs (macroreplications) for the same factors.
        # Each macroreplication uses a different random number stream.
        # Must be a positive integer.
        "num_macroreps": 1,
        # Enable debug-level logging for detailed model information.
        # Set to True for debugging, False for normal operation.
        "debug_logging": True,
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
    model_class = config["model_class"]
    fixed_factors = config["fixed_factors"]
    num_macroreps = config["num_macroreps"]
    debug_logging = config["debug_logging"]

    # Print the configuration settings as a table.
    config_header = ["Parameter", "Value"]
    config_tuples = [
        ("Model Class", model_class.__name__),
        ("Fixed Factors", fixed_factors if fixed_factors else "None (using defaults)"),
        ("Number of Macroreplications", num_macroreps),
        ("Debug Logging Enabled", debug_logging),
    ]
    print_table("Configuration Settings", config_header, config_tuples)

    if num_macroreps <= 0:
        print(
            f"> {model_class.__name__} has no macroreplications to run. Exiting script."
        )
        return

    # Set the logging level for the model.
    log_level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Initialize the model with the specified fixed factors.
    # If the fixed factors are invalid, the model will raise an error.
    try:
        mymodel = model_class(fixed_factors)
    except Exception as e:
        print(f"> Error initializing {model_class.__name__} model: {e}")
        print("> Exiting script.")
        return

    # Check all factors collectively.
    try:
        is_model_simulatable = mymodel.check_simulatable_factors()
        model_sim_str = "IS" if is_model_simulatable else "IS NOT"
        print(
            f"> {mymodel.name} {model_sim_str} simulatable with the specified factors."
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
    rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]

    # Keep track of responses and gradients between macroreplications so they can be
    # compared/graphed at the end.
    non_dict_resp_between_mreps = {}
    dict_resp_between_mreps = {}
    gradients_between_mreps = {}

    # Run a single replication of the model.
    for mrep in range(1, num_macroreps + 1):
        print(
            f"> Running macroreplication {mrep} of {num_macroreps}...",
            end="",
            flush=True,
        )
        responses, gradients = mymodel.replicate(rng_list)
        print(" done.")
        # Separate the responses into dict and non-dict responses.
        non_dict_responses = []
        dict_responses = []
        for key, value in responses.items():
            if not isinstance(value, (dict)):
                non_dict_responses.append((key, value))
            else:
                dict_responses.append((key, value))
        # Store the responses and gradients for this macroreplication.
        non_dict_resp_between_mreps[mrep] = non_dict_responses
        dict_resp_between_mreps[mrep] = dict_responses
        gradients_between_mreps[mrep] = gradients
        # Advance RNG
        for rng in rng_list:
            rng.advance_subsubstream()
    print("> Finished macroreplications.")

    # Switch back to normal logging level.
    logging.getLogger().setLevel(logging.INFO)

    # Combine the responses and gradients from all macroreplications
    combined_non_dict_responses = defaultdict(list)
    combined_dict_responses = defaultdict(list)
    combined_gradients = defaultdict(list)

    # Combine responses and gradients together to share common keys
    for mrep in range(1, num_macroreps + 1):
        for key, value in non_dict_resp_between_mreps[mrep]:
            combined_non_dict_responses[key].append(value)
        for key, value in dict_resp_between_mreps[mrep]:
            combined_dict_responses[key].append(value)
        for factor, gradient in gradients_between_mreps[mrep].items():
            combined_gradients[factor].append(gradient)

    # Format and print combined non-dict responses
    combined_non_dict_list = [
        (response, *_round_list(values))
        for response, values in combined_non_dict_responses.items()
    ]
    mrep_column_labels = [f"mrep {i}" for i in range(1, num_macroreps + 1)]
    non_dict_headers = ["Response", *mrep_column_labels]
    print_table("Responses by mrep", non_dict_headers, combined_non_dict_list)

    # Format and print combined gradients
    for factor, g_dicts in combined_gradients.items():
        factor_gradient_matrix = defaultdict(list)
        for g_dict in g_dicts:
            for inner_factor, value in g_dict.items():
                factor_gradient_matrix[inner_factor].append(value)

        combined_gradients_list = [
            (inner_factor, *_round_list(values))
            for inner_factor, values in factor_gradient_matrix.items()
        ]
        gradients_headers = ["w.r.t Factor", *mrep_column_labels]
        print_table(
            f"Gradients for {factor}", gradients_headers, combined_gradients_list
        )

    # For each response
    for key in combined_dict_responses:
        # Start a new plot
        plt.figure(dpi=600)  # High res for digital viewing/zooming
        # For each macroreplication
        for mrep in range(1, num_macroreps + 1):
            response = combined_dict_responses[key][mrep - 1]
            # Add the response to the plot
            x_vals = list(response.keys())
            y_vals = list(response.values())
            plt.plot(x_vals, y_vals, label=f"mrep {mrep}")
        # Set the title and labels
        plt.title(f"Response: {key}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        # Show the plot
        save_dir = Path(__file__).resolve().parent
        save_name = mymodel.name + "_" + key + time.strftime("%Y%m%d-%H%M%S")
        save_path = save_dir / save_name
        plt.savefig(save_path.with_suffix(".png"))
        print(f"> Saved plot of {key} to {save_path}")


def _round_list(to_round: list) -> list:
    """Round a list of numbers to the same number of decimal places.

    Args:
        to_round (list): A list of numbers to round.

    Returns:
        list: A list of rounded numbers.
    """
    try:
        # Figure out how many decimal places to round to.
        min_val = abs(min(to_round))
        min_exp = math.floor(math.log10(min_val))
        # Round the values to the calculated number of decimal places.
        return [round(r, 3 - min_exp) if r != 0 else 0 for r in to_round]
    except Exception:
        # If there is an error, return the original list.
        return to_round


if __name__ == "__main__":
    main()
