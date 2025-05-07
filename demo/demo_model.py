"""Demo for Model Debugging.

This script is intended to help with debugging a model.
It imports a model, initializes a model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

# Import standard libraries
import sys
from pathlib import Path

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import modules that might depend on the simopt package
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.utils import print_table


def main() -> None:
    """Main function to run the data farming experiment."""
    # Import model.
    # from models.<filename> import <model_class_name>
    # Replace <filename> with name of .py file containing model class.
    # Replace <model_class_name> with name of model class.
    # Fix factors of model. Specify a dictionary of factors.
    # fixed_factors = {}  # Resort to all default values.
    # Look at Model class definition to get names of factors.
    # Initialize an instance of the specified model class.
    # mymodel = <model_class_name>(fixed_factors)
    # Replace <model_class_name> with name of model class.
    # Working example for MM1 model.
    # -----------------------------------------------
    from simopt.models.mm1queue import MM1Queue

    fixed_factors = {"lambda": 3.0, "mu": 8.0}
    mymodel = MM1Queue(fixed_factors=fixed_factors)

    num_macroreps = 1
    # -----------------------------------------------

    # The rest of this script requires no changes.

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
    print(f"> The model {model_sim_str} simulatable with the specified factors.")

    # Create a list of RNG objects for the simulation model to use when
    # running replications.
    rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]

    # Run a single replication of the model.
    for mrep in range(num_macroreps):
        print(f"> Running macroreplication {mrep + 1}/{num_macroreps}...")
        responses, gradients = mymodel.replicate(rng_list)
        non_dict_responses = []
        dict_responses = []
        for key, value in responses.items():
            if not isinstance(value, (dict)):
                non_dict_responses.append((key, value))
            else:
                dict_responses.append((key, value))
        # Only specify "non-dict" if there are dict responses, otherwise just refer to
        # them as "responses" for clarity.
        non_dict_title = "Non-Dict Responses" if len(dict_responses) else "Responses"
        print_table(non_dict_title, ["Response", "Value"], non_dict_responses)
        # Print dict responses.
        for factor, dictionary in dict_responses:
            # Split each dict into its own table.
            responses = list(dictionary.items())
            print_table(f"Dict Responses for {factor}", ["Key", "Value"], responses)
        # Print gradients.
        for factor in gradients:
            print_table(
                f"Gradients for {factor}",
                ["w.r.t Factor", "Gradient"],
                gradients[factor],
            )
        # Advance RNG
        for rng in rng_list:
            rng.advance_subsubstream()
    print("> Finished macroreplications.")


if __name__ == "__main__":
    main()
