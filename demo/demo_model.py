"""Demo for Model Debugging.

This script is intended to help with debugging a model.
It imports a model, initializes a model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

import sys
from pathlib import Path

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import random number generator.
from mrg32k3a.mrg32k3a import MRG32k3a


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
    # -----------------------------------------------

    # The rest of this script requires no changes.

    # Check that all factors describe a simulatable model.
    # Check fixed factors individually.
    for key, value in mymodel.factors.items():
        print(
            f"The factor {key} is set as {value}. "
            f"Is this simulatable? {bool(mymodel.check_simulatable_factor(key))}."
        )
    # Check all factors collectively.
    print(
        "Is the specified model simulatable? "
        f"{bool(mymodel.check_simulatable_factors())}."
    )

    # Create a list of RNG objects for the simulation model to use when
    # running replications.
    rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]

    # Run a single replication of the model.
    responses, gradients = mymodel.replicate(rng_list)
    print("\nFor a single replication:")
    print("\nResponses:")
    for key, value in responses.items():
        print(f"\t {key} is {value}.")
    print("\n Gradients:")
    for outerkey in gradients:
        print(f"\tFor the response {outerkey}:")
        for innerkey, value in gradients[outerkey].items():
            print(f"\t\tThe gradient w.r.t. {innerkey} is {value}.")


if __name__ == "__main__":
    main()
