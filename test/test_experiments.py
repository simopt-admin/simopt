"""Dynamically generate test classes for each YAML in the expected results directory.

This script automatically creates test classes for each YAML file in the
`expected_results` directory. This allows for easy addition of new test cases
and, by utilizing the experiment test core, ensures that any changes to test
methodology are automatically applied to all tests. It also essentially
eliminates DRY violations in the experiment testing process.
"""

from pathlib import Path

# these imports aren't used outside of the exec
from experiment_test_core import (  # noqa: F401
    ExperimentTest,
    ExperimentTestMixin,
)

EXPECTED_RESULTS_DIR = Path(__file__).parent / "expected_results"

# Build a mapping of class names to file paths
_FILE_CLASS_MAP = {}


def _filename_to_classname(file: str) -> str:
    name = file.replace(".pickle.zst", "")
    parts = name.replace("-", "_").split("_")
    return "Test" + "".join(part.title() for part in parts)


for yaml_path in EXPECTED_RESULTS_DIR.glob("*.pickle.zst"):
    class_name = _filename_to_classname(yaml_path.name)
    _FILE_CLASS_MAP[class_name] = yaml_path.resolve()  # store full path

# Explicitly define each class so that unittest and vs code can find them
# NOTE: exec is usually not recommended due to security concerns, but in this
# case it should be fine as long as nobody tries injecting malicious filenames
# into the expected results directory.
for class_name, full_path in _FILE_CLASS_MAP.items():
    exec(f"""
class {class_name}(ExperimentTest, ExperimentTestMixin):
    @property
    def filepath(self):
        return Path(r'{full_path!s}')
""")
