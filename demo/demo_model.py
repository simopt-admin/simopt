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
    print("> Running a single replication of the model...")
    responses, gradients = mymodel.replicate(rng_list)
    print("> For a single replication:")
    print_table("Responses", ["Response", "Value"], responses)
    for outerkey in gradients:
        print_table(
            f"Gradients for {outerkey}",
            ["w.r.t Factor", "Gradient"],
            gradients[outerkey],
        )


def print_table(name: str, headers: list[str], data: list[tuple] | dict) -> None:
    """Print a table with headers and data.

    Args:
        name (str): Name of the table.
        headers (list[str]): List of column headers.
        data (list[tuple]): List of rows, each row is a tuple of values.
    """
    # Convert data out of dict (if necessary)
    if isinstance(data, dict):
        data = list(data.items())
    # Calculate the maximum length of each column
    data_widths = [max(len(str(item)) for item in col) for col in zip(*data)]
    header_widths = [len(header) for header in headers]
    max_widths = [
        max(header_width, col_width)
        for header_width, col_width in zip(header_widths, data_widths)
    ]

    # Compute total width of the table
    # There's 3 seperator characters between each column
    seperator_lengths = 3 * (len(headers) - 1)
    total_width = sum(max_widths) + seperator_lengths
    # If table is shorter than name, expand last column
    if total_width < len(name):
        shortfall = len(name) - total_width
        max_widths[-1] += shortfall
        total_width = len(name)

    # Center title in the table
    title_indent_count = (total_width - len(name)) // 2
    title_indent = " " * title_indent_count

    header_row = " │ ".join(
        f"{header:<{width}}" for header, width in zip(headers, max_widths)
    )

    # Print the table
    print()
    print(f"{title_indent}{name}")
    print("─" * total_width)
    print(header_row)
    print("─┼─".join("─" * width for width in max_widths))
    for row in data:
        print(" │ ".join(f"{item!s:<{width}}" for item, width in zip(row, max_widths)))
    print()


if __name__ == "__main__":
    main()
