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
        result = baselines.load_recorded_baseline(baselines.BASELINE_DIR, test_id)
        assert {"logs", "curves", "violins"} <= result.keys()


def test_pandas_3_1_baseline_integrity():
    directory = baselines.PANDAS_3_1_BASELINE_DIR
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["source_revision"]
    assert manifest["environment"]["pandas"].startswith("3.1")
    legacy = json.loads((baselines.BASELINE_DIR / "manifest.json").read_text())
    legacy_ids = {record["test_id"] for record in legacy["records"]}
    test_ids = [record["test_id"] for record in manifest["records"]]
    assert len(test_ids) == len(set(test_ids)) == 18
    assert set(test_ids) <= legacy_ids
    for test_id in test_ids:
        assert "logs" in baselines.load_recorded_baseline(directory, test_id)


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
    monkeypatch.setattr(baselines, "_manifest", lambda directory: {"records": [record]})
    with pytest.raises(AssertionError, match="checksum mismatch"):
        baselines.load_recorded_baseline(tmp_path, test_id)
