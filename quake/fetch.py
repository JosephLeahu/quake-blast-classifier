"""Download a Pacific Northwest event catalog from the USGS FDSN Event web service.

    python -m quake.fetch --start 2015-01-01 --end 2025-01-01

The catalog includes both earthquakes and human-made events (quarry blasts,
explosions) labeled by the seismic network, which is what makes this a
supervised-learning problem. Requests go month by month to stay under the
API's 20,000-events-per-query limit.
"""
from __future__ import annotations

import argparse
import io
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
# Washington + Oregon + a margin; covers the Pacific Northwest Seismic Network region.
PNW_BOX = {"minlatitude": 42.0, "maxlatitude": 49.5, "minlongitude": -125.0, "maxlongitude": -116.5}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows, cur = [], start
    while cur < end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        windows.append((cur, min(nxt, end)))
        cur = nxt
    return windows


def build_params(start: date, end: date, min_mag: float = 0.0) -> dict:
    return {
        "format": "csv",
        "starttime": start.isoformat(),
        "endtime": end.isoformat(),
        "minmagnitude": min_mag,
        "orderby": "time-asc",
        "limit": 20000,
        **PNW_BOX,
    }


def fetch_window(session: requests.Session, start: date, end: date, min_mag: float = 0.0) -> pd.DataFrame:
    for attempt in range(4):
        resp = session.get(API, params=build_params(start, end, min_mag), timeout=60)
        if resp.status_code == 204 or not resp.text.strip():
            return pd.DataFrame()
        if resp.ok:
            return pd.read_csv(io.StringIO(resp.text))
        time.sleep(2 ** attempt)  # back off on 429 / 5xx
    resp.raise_for_status()
    return pd.DataFrame()


def fetch_catalog(start: date, end: date, min_mag: float = 0.0, session: requests.Session | None = None) -> pd.DataFrame:
    session = session or requests.Session()
    frames = []
    for a, b in month_windows(start, end):
        df = fetch_window(session, a, b, min_mag)
        print(f"{a} -> {b}: {len(df):6d} events")
        if len(df) >= 20000:
            print("  warning: hit the 20k limit; use smaller windows for this period")
        frames.append(df)
        time.sleep(0.5)  # be polite to a free public API
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--min-mag", type=float, default=0.0)
    ap.add_argument("--out", default=str(DATA_DIR / "pnw_catalog.csv"))
    args = ap.parse_args()

    df = fetch_catalog(date.fromisoformat(args.start), date.fromisoformat(args.end), args.min_mag)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nsaved {len(df):,} events to {args.out}")
    if "type" in df:
        print(df["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
