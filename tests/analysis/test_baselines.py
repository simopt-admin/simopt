import hashlib
import json

import pytest

from tests.analysis import baselines


def test_recorded_baseline_integrity():
    manifest = json.loads((baselines.BASELINE_DIR / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["source_revision"]
    test_ids = [record["test_id"] for record in manifest["records"]]
    assert len(test_ids) == len(set(test_ids)) == 81
    for test_id in test_ids:
        result = baselines.load_plot_baseline(test_id)
        assert {"logs", "curves", "violins"} <= result.keys()


def test_missing_baseline_fails():
    with pytest.raises(AssertionError, match="Missing recorded plot baseline"):
        baselines.load_plot_baseline("tests/analysis/missing.py::test_missing")


def test_changed_baseline_fails(tmp_path, monkeypatch):
    test_id = "changed baseline"
    expected = b"original data"
    (tmp_path / "result.json.zst").write_bytes(b"changed data")
    record = {
        "test_id": test_id,
        "file": "result.json.zst",
        "sha256": hashlib.sha256(expected).hexdigest(),
    }
    monkeypatch.setattr(baselines, "BASELINE_DIR", tmp_path)
    monkeypatch.setattr(baselines, "_manifest", lambda: {"records": [record]})
    with pytest.raises(AssertionError, match="checksum mismatch"):
        baselines.load_plot_baseline(test_id)
