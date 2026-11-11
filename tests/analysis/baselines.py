import hashlib
import json
from functools import cache
from pathlib import Path

import pandas as pd
import zstandard as zstd

BASELINE_DIR = Path(__file__).parents[1] / "expected_results" / "plot_baselines"

# pandas >= 3.1 rounds some grouped means differently in the last bit. When solvers find
# solutions whose mean objectives tie exactly, that changes which one is picked as x*, so
# the affected tests have separately recorded baselines.
PANDAS_3_1_BASELINE_DIR = BASELINE_DIR / "pandas_3_1"


def pandas_at_least_3_1() -> bool:
    major, minor = (int(part) for part in pd.__version__.split(".")[:2])
    return (major, minor) >= (3, 1)


@cache
def _manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def _has_record(directory: Path, test_id: str) -> bool:
    return any(record["test_id"] == test_id for record in _manifest(directory)["records"])


def load_recorded_baseline(directory: Path, test_id: str) -> dict:
    """Load a recorded result from `directory`, verifying its identity and checksum."""
    records = {record["test_id"]: record for record in _manifest(directory)["records"]}
    if test_id not in records:
        raise AssertionError(f"Missing recorded plot baseline for {test_id}")
    record = records[test_id]
    data = (directory / record["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == record["sha256"], "Plot baseline checksum mismatch"
    result = json.loads(zstd.ZstdDecompressor().decompress(data))
    assert result["test_id"] == test_id, "Plot baseline test identity mismatch"
    return result


def load_plot_baseline(test_id: str) -> dict:
    """Load the recorded result for `test_id` matching the installed pandas version."""
    if pandas_at_least_3_1() and _has_record(PANDAS_3_1_BASELINE_DIR, test_id):
        return load_recorded_baseline(PANDAS_3_1_BASELINE_DIR, test_id)
    return load_recorded_baseline(BASELINE_DIR, test_id)
