"""Print the names and classes of all models, problems, and solvers.

Since the directory is dynamically generated, this script can be used to quickly
check its contents and lookup the abbreviat names of classes. This is especially
useful when it is unclear if the class's name has been overridden from the default.
"""

# Import standard libraries
import sys
from pathlib import Path

from simopt.utils import print_table

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import simopt.directory as directory


def main() -> None:
    """Print the contents of the problem and solver directories."""

    def invert_dict(d: dict) -> dict:
        """Invert a dictionary."""
        return {v: k for k, v in d.items()}

    directory_types = {
        "Model": directory.model_directory,
        "Problem": directory.problem_directory,
        "Solver": directory.solver_directory,
    }

    for dir_type, dir_dict in directory_types.items():
        # Class Name (ABBR) - Class Name (Full) - Class Type
        # Invert the dictionaries to get a common key
        dir_dict_inv = invert_dict(dir_dict)
        dir_dict_full = directory.generate_unabbreviated_mapping(dir_dict)
        dir_dict_full_inv = invert_dict(dir_dict_full)
        entries = []
        for class_module, name_abbr in dir_dict_inv.items():
            full_module_str = class_module.__module__ + "." + class_module.__name__
            entry_tuple = (name_abbr, dir_dict_full_inv[class_module], full_module_str)
            entries.append(entry_tuple)
        print_table(
            f"{dir_type} Directory",
            ["Class Name (Abbr)", "Class Name (Full)", "Class Module"],
            entries,
        )


if __name__ == "__main__":
    main()
