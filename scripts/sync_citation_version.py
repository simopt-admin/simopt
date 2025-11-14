#!/usr/bin/env python

"""A script to synchronize the version from _version.py to CITATION.cff."""

import re
import sys
from pathlib import Path

# --- Configuration ---------------------------------------------------

# Append the parent directory (simopt package) to the system path
SIMOPT_PACKAGE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SIMOPT_PACKAGE_DIR))

# Source directory containing the _version.py file
SIMOPT_DIR = SIMOPT_PACKAGE_DIR / "simopt"

# Source of the citation file
CITATION_FILENAME = "CITATION.cff"
CITATION_FILE: Path = SIMOPT_PACKAGE_DIR / CITATION_FILENAME

# Regex pattern to identify the version line in CITATION.cff

VERSION_PATTERN = re.compile(r"^(version:\s*).+$")

# ---------------------------------------------------------------------


def main() -> None:
    """Reads the version and updates the CITATION.cff file."""
    # --- Get the source version ---
    from simopt._version import __version__

    # --- Read and update the CITATION.cff file ---
    if not CITATION_FILE.exists():
        print(f"Error: File not found at {CITATION_FILE}")
        sys.exit(1)

    # We use 'str(__version__)' to handle non-string version types
    # and add quotes to be safe YAML, especially for pre-release tags.
    replacement_line = f'version: "{__version__!s}"'

    print(f"Updating {CITATION_FILE}...")

    new_lines = []
    found = False

    with CITATION_FILE.open("r") as f:
        for line in f:
            # Check if the line matches our version pattern
            if not found and VERSION_PATTERN.match(line):
                # If it matches, replace it with our new line
                new_lines.append(replacement_line + "\n")
                found = True
                print(f"  - Replaced: {line.strip()}")
                print(f"  + With:     {replacement_line}")
            else:
                # Otherwise, keep the line as-is
                new_lines.append(line)

    if not found:
        print("Error: A 'version:' line was not found in the file.")
        sys.exit(1)

    # Write the modified content back to the file
    with CITATION_FILE.open("w") as f:
        f.writelines(new_lines)

    print("Update complete.")


if __name__ == "__main__":
    main()
