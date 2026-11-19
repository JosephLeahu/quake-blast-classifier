"""Synthetic catalog generator — for tests and offline demos ONLY.

Produces rows shaped like the USGS CSV with roughly realistic patterns (daytime,
weekday, shallow blasts near a few fixed quarries). Never report metrics from
this data as real results; train on the real catalog from `quake.fetch`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUARRIES = [(45.55, -122.40), (47.10, -122.55), (46.60, -120.50), (44.05, -123.10)]


def make_catalog(n: int = 4000, blast_frac: float = 0.12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_blast = int(n * blast_frac)
    n_eq = n - n_blast
    start = pd.Timestamp("2020-01-01", tz="UTC")

    # Earthquakes: any time of day, deeper, spread across the region.
    eq_times = start + pd.to_timedelta(rng.uniform(0, 4 * 365, n_eq), unit="D")
    eq = pd.DataFrame({
        "time": eq_times,
        "latitude": rng.uniform(42.5, 49.0, n_eq),
        "longitude": rng.uniform(-124.5, -117.0, n_eq),
        "depth": np.abs(rng.normal(12, 8, n_eq)),
        "mag": np.round(np.abs(rng.gumbel(1.0, 0.45, n_eq)), 2),
        "type": "earthquake",
    })

    # Blasts: weekdays ~10:00-16:00 local (UTC-8 => 18:00-24:00 UTC), shallow, near quarries.
    days = rng.integers(0, 4 * 365, n_blast)
    days = days[pd.to_datetime(start + pd.to_timedelta(days, unit="D")).dayofweek < 5][: n_blast]
    while len(days) < n_blast:
        extra = rng.integers(0, 4 * 365, n_blast)
        extra = extra[pd.to_datetime(start + pd.to_timedelta(extra, unit="D")).dayofweek < 5]
        days = np.concatenate([days, extra])[:n_blast]
    hours_utc = rng.normal(21, 1.6, n_blast)
    q = np.array(QUARRIES)[rng.integers(0, len(QUARRIES), n_blast)]
    blast = pd.DataFrame({
        "time": start + pd.to_timedelta(days, unit="D") + pd.to_timedelta(hours_utc, unit="h"),
        "latitude": q[:, 0] + rng.normal(0, 0.03, n_blast),
        "longitude": q[:, 1] + rng.normal(0, 0.03, n_blast),
        "depth": np.abs(rng.normal(0.5, 1.0, n_blast)),
        "mag": np.round(np.abs(rng.normal(1.4, 0.4, n_blast)), 2),
        "type": rng.choice(["quarry blast", "explosion"], n_blast, p=[0.9, 0.1]),
    })

    df = pd.concat([eq, blast], ignore_index=True)
    m = len(df)
    df["nst"] = rng.integers(4, 60, m)
    df["gap"] = rng.uniform(40, 250, m)
    df["dmin"] = rng.uniform(0.01, 0.5, m)
    df["rms"] = rng.uniform(0.05, 0.6, m)
    df["magType"] = rng.choice(["ml", "md"], m)
    df["id"] = [f"syn{i:06d}" for i in range(m)]
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return df.sort_values("time").reset_index(drop=True)
