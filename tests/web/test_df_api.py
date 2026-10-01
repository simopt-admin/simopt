"""Tests for the data-farming web API."""

import csv
from pathlib import Path

from fastapi.testclient import TestClient

from simopt.web import server

client = TestClient(server.app)


def test_models_endpoint():
    names = [m["name"] for m in client.get("/models").json()["models"]]
    assert "MM1" in names


def test_solver_factors():
    response = client.get("/df/factors/solver/ASTRODF")
    assert response.status_code == 200
    factors = {f["name"]: f for f in response.json()["factors"]}
    assert factors["eta_1"]["type"] == "float"
    assert factors["eta_1"]["datafarmable"]
    assert factors["easy_solve"]["type"] == "bool"
    assert all(f["source"] == "solver" for f in factors.values())


def test_problem_factors_include_model():
    response = client.get("/df/factors/problem/CNTNEWS-1")
    assert response.status_code == 200
    assert {f["source"] for f in response.json()["factors"]} == {"model", "problem"}


def test_unknown_factors_name():
    assert client.get("/df/factors/model/NOPE").status_code == 404


def test_preview_and_errors():
    spec = {"kind": "model", "name": "MM1", "varied": {"mu": {"min": 2, "max": 4, "decimals": 1}}}
    body = client.post("/df/preview", json=spec).json()
    assert body["n_points"] == len(body["rows"]) > 1
    assert body["columns"][0] == "mu"

    spec["varied"] = {"bogus": {"min": 0, "max": 1}}
    response = client.post("/df/preview", json=spec)
    assert response.status_code == 422
    assert "bogus" in response.json()["detail"]


def test_run_model(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    spec = {"kind": "model", "name": "MM1", "varied": {"mu": {"min": 2, "max": 4, "decimals": 1}}}
    response = client.post("/df/run_model", json={"spec": spec, "n_reps": 2})
    assert response.status_code == 200
    body = response.json()
    n_points = client.post("/df/preview", json=spec).json()["n_points"]
    assert len(body["rows"]) == n_points * 2

    with Path(body["csv_path"]).open(newline="") as f:
        csv_rows = list(csv.reader(f, delimiter="\t"))
    assert csv_rows[0] == body["columns"]
    assert len(csv_rows) - 1 == n_points * 2

    downloaded = client.get(f"/df/results/{body['run_id']}/raw_results.csv")
    assert downloaded.status_code == 200
    assert client.get("/df/results/..%2F..%2Fetc/raw_results.csv").status_code == 404


def test_run_model_rejects_non_model():
    spec = {"kind": "solver", "name": "ASTRODF"}
    assert client.post("/df/run_model", json={"spec": spec}).status_code == 422
