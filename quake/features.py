"""Turn raw catalog rows into model features.

Why these features work for blast discrimination:
- Quarry blasts happen during working hours on weekdays; earthquakes don't care
  what time it is. Local hour and day of week are the strongest signals.
- Blasts are at or near the surface; most earthquakes are kilometers deep.
- Blasts are small and cluster at a few fixed quarry and mine sites (latitude/longitude).
- Location-quality columns (gap, rms, nst) differ for shallow surface sources.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
BLAST_TYPES = {"quarry blast", "explosion", "mining explosion", "chemical explosion", "experimental explosion"}
KEEP_TYPES = BLAST_TYPES | {"earthquake"}

NUMERIC = ["latitude", "longitude", "depth", "mag", "nst", "gap", "dmin", "rms"]
FEATURES = NUMERIC + ["hour_sin", "hour_cos", "local_hour", "day_of_week", "is_weekend", "magtype_md"]


def label_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep earthquakes and human-made blasts; drop ice quakes, sonic booms, etc."""
    df = df[df["type"].isin(KEEP_TYPES)].copy()
    df["is_blast"] = df["type"].isin(BLAST_TYPES).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with exactly FEATURES columns. Works for a batch or a single event."""
    out = pd.DataFrame(index=df.index)
    t = pd.to_datetime(df["time"], utc=True, format="ISO8601").dt.tz_convert(LOCAL_TZ)
    hour = t.dt.hour + t.dt.minute / 60
    out["local_hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)   # cyclic encoding: 23:00 is close to 01:00
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_of_week"] = t.dt.dayofweek
    out["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    for col in NUMERIC:
        out[col] = pd.to_numeric(df.get(col), errors="coerce")
    mag_type = df.get("magType", pd.Series("", index=df.index)).fillna("").str.lower()
    out["magtype_md"] = (mag_type == "md").astype(int)
    return out[FEATURES]


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: train on the past, test on the future (no look-ahead leakage)."""
    df = df.sort_values("time").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]
