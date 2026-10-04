import hashlib
import json
from functools import cache
from pathlib import Path

import zstandard as zstd

BASELINE_DIR = Path(__file__).parents[1] / "expected_results" / "plot_baselines"


@cache
def _manifest():
    return json.loads((BASELINE_DIR / "manifest.json").read_text())


def load_plot_baseline(test_id: str) -> dict:
    """Load a recorded legacy result, verifying its identity and checksum."""
    records = {record["test_id"]: record for record in _manifest()["records"]}
    if test_id not in records:
        raise AssertionError(f"Missing recorded plot baseline for {test_id}")
    record = records[test_id]
    data = (BASELINE_DIR / record["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == record["sha256"], "Plot baseline checksum mismatch"
    result = json.loads(zstd.ZstdDecompressor().decompress(data))
    assert result["test_id"] == test_id, "Plot baseline test identity mismatch"
    return result
