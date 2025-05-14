"""Demo for Model Debugging.

This script is intended to help with debugging a model.
It imports a model, initializes a model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

# Import standard libraries
import math
import sys
from collections import defaultdict
from pathlib import Path

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import modules that might depend on the simopt package
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.utils import print_table


def main() -> None:
    """Main function to run the data farming experiment."""
    ###################################################################################
    # Demo Setup
    ###################################################################################

    # Import the model class to be demo'd, using the following format:
    # from simopt.models.<filename> import <model_class_name>
    # Since this is a module import, you do not need to include the .py extension.
    from simopt.models.mm1queue import MM1Queue

    # Set the model class (must be imported above).
    model_class = MM1Queue

    # Set the fixed factors for the model.
    # Setting the fixed factors to {} or None will use the default values.
    # For more details on which factors are available, check the definition for the
    # model class you are using.
    fixed_factors = {"lambda": 3.0, "mu": 8.0}

    # Set the number of macroreplications to run.
    # This is the number of times the model will be run with the same factors.
    # Each macroreplication will use a different random number generator.
    # Must be a positive integer.
    num_macroreps = 1

    # The rest of this script requires no changes.

    ###################################################################################

    mymodel = model_class(fixed_factors)

    if num_macroreps <= 0:
        print(f"> {mymodel.name} has no macroreplications to run. Exiting script.")
        return

    # Check that all factors describe a simulatable model.
    # Check fixed factors individually.
    output_tuples = []
    for factor, value in mymodel.factors.items():
        is_factor_simulatable = mymodel.check_simulatable_factor(factor)
        output_tuple = (factor, value, is_factor_simulatable)
        output_tuples.append(output_tuple)
    # Print table
    print_table(
        "Simulatable Factors", ["Factor", "Value", "Simulatable"], output_tuples
    )

    # Check all factors collectively.
    is_model_simulatable = mymodel.check_simulatable_factors()
    model_sim_str = "IS" if is_model_simulatable else "IS NOT"
    print(f"> {mymodel.name} {model_sim_str} simulatable with the specified factors.")

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
        print(f"> Running macroreplication {mrep} of {num_macroreps}...", end="")
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
        non_dict_resp_between_mreps[mrep] = non_dict_responses
        dict_resp_between_mreps[mrep] = dict_responses
        gradients_between_mreps[mrep] = gradients
        # Advance RNG
        for rng in rng_list:
            rng.advance_subsubstream()
    print("> Finished macroreplications.")

    # Combine the responses and gradients from all macroreplications
    combined_non_dict_responses = defaultdict(list)
    combined_gradients = defaultdict(list)

    # Combine non-dict responses
    for mrep in range(1, num_macroreps + 1):
        for key, value in non_dict_resp_between_mreps[mrep]:
            combined_non_dict_responses[key].append(value)
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
        math.log10(min_val)
        min_exp = math.floor(math.log10(min_val))
        # Round the values to the calculated number of decimal places.
        return [round(r, 3 - min_exp) if r != 0 else 0 for r in to_round]
    except Exception:
        # If there is an error, return the original list.
        return to_round


if __name__ == "__main__":
    main()
