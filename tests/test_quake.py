import itertools
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quake import fetch
from quake.features import FEATURES, build_features, label_and_filter, time_split
from quake.synthetic import make_catalog
from quake.train import best_f1_threshold, evaluate, run


@pytest.fixture(scope="module")
def catalog():
    return make_catalog(3000, seed=1)


# ---- fetch -----------------------------------------------------------------

def test_month_windows_cover_range_without_gaps():
    w = fetch.month_windows(date(2023, 11, 15), date(2024, 2, 10))
    assert w[0] == (date(2023, 11, 15), date(2023, 12, 1))
    assert w[1] == (date(2023, 12, 1), date(2024, 1, 1))
    assert w[-1][1] == date(2024, 2, 10)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(w))


def test_build_params():
    p = fetch.build_params(date(2024, 1, 1), date(2024, 2, 1))
    assert p["format"] == "csv" and p["starttime"] == "2024-01-01" and p["minlatitude"] == 42.0


class FakeResp:
    def __init__(self, status, text):
        self.status_code, self.text, self.ok = status, text, 200 <= status < 300

    def raise_for_status(self):
        raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, resps):
        self.resps, self.calls = list(resps), 0

    def get(self, url, params, timeout):
        self.calls += 1
        return self.resps.pop(0)


def test_fetch_window_parses_csv_and_retries(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    csv = "time,latitude,longitude,depth,mag,type,id\n2024-01-01T00:00:00Z,45,-122,1,1.2,earthquake,a\n"
    s = FakeSession([FakeResp(503, "busy"), FakeResp(200, csv)])
    df = fetch.fetch_window(s, date(2024, 1, 1), date(2024, 2, 1))
    assert len(df) == 1 and s.calls == 2


def test_fetch_window_empty(monkeypatch):
    assert fetch.fetch_window(FakeSession([FakeResp(204, "")]), date(2024, 1, 1), date(2024, 2, 1)).empty


# ---- features --------------------------------------------------------------

def test_label_and_filter_drops_other_types():
    df = pd.DataFrame({"type": ["earthquake", "quarry blast", "ice quake", "explosion"]})
    out = label_and_filter(df)
    assert out["is_blast"].tolist() == [0, 1, 1]


def test_build_features_local_time():
    # 2024-07-03 20:30 UTC = 13:30 PDT on a Wednesday
    X = build_features(pd.DataFrame([{"time": "2024-07-03T20:30:00Z", "latitude": 45, "longitude": -122,
                                       "depth": 0.2, "mag": 1.5, "magType": "md"}]))
    assert list(X.columns) == FEATURES
    row = X.iloc[0]
    assert row["local_hour"] == pytest.approx(13.5) and row["day_of_week"] == 2
    assert row["is_weekend"] == 0 and row["magtype_md"] == 1
    assert np.isnan(row["gap"])  # missing columns become NaN, handled by the models


def test_time_split_is_chronological(catalog):
    train, test = time_split(catalog, 0.25)
    assert train["time"].max() <= test["time"].min()
    assert len(test) == pytest.approx(len(catalog) * 0.25, abs=1)


# ---- training ----------------------------------------------------------------

def test_threshold_and_evaluate():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.35, 0.8])
    thr = best_f1_threshold(y, p)
    m = evaluate(y, p, thr)
    assert m["confusion"]["tp"] + m["confusion"]["fn"] == 2 and 0 < m["f1"] <= 1


def test_full_pipeline_beats_baseline(catalog, tmp_path):
    s = run(catalog, out_dir=tmp_path, data_label="synthetic")
    base = s["models"]["baseline_prior"]["test"]["pr_auc"]
    best = s["models"][s["best_model"]]["test"]["pr_auc"]
    assert best > base + 0.3
    assert (tmp_path / "models" / "model.joblib").exists()
    assert "SYNTHETIC" in (tmp_path / "reports" / "REPORT.md").read_text()
    assert (tmp_path / "reports" / "pr_curve.png").exists()
