"""HTTP tests for the model server. Requires FastAPI + httpx."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from quake import api  # noqa: E402
from quake.synthetic import make_catalog  # noqa: E402
from quake.train import run  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    out = tmp_path_factory.mktemp("m")
    run(make_catalog(2000, seed=3), out_dir=out, data_label="synthetic")
    api.MODEL_PATH = out / "models" / "model.joblib"
    api._bundle = None
    return TestClient(api.app)


def test_predict(client):
    events = [
        {"time": "2024-06-12T21:00:00Z", "latitude": 45.55, "longitude": -122.40, "depth": 0.3, "mag": 1.4},
        {"time": "2024-06-15T09:00:00Z", "latitude": 47.9, "longitude": -121.0, "depth": 18, "mag": 2.1},
    ]
    out = client.post("/predict", json=events).json()
    assert [o["label"] for o in out] == ["blast", "earthquake"]


def test_validation(client):
    assert client.post("/predict", json=[{"time": "x", "latitude": 999}]).status_code == 422
