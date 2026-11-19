"""Serve the trained model:  uvicorn quake.api:app --reload"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import build_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
app = FastAPI(title="Earthquake vs. Blast Classifier", version="1.0.0")
_bundle: Optional[dict] = None


def bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "model not trained yet: run python -m quake.train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class Event(BaseModel):
    time: str = Field(..., examples=["2024-06-12T21:15:00Z"], description="UTC, ISO 8601")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    depth: float = Field(..., description="km")
    mag: float
    magType: str = "ml"
    nst: Optional[float] = None
    gap: Optional[float] = None
    dmin: Optional[float] = None
    rms: Optional[float] = None


@app.get("/health")
def health():
    b = bundle()
    return {"status": "ok", "model": b["summary"]["best_model"], "data": b["summary"]["data"]}


@app.post("/predict")
def predict(events: list[Event]):
    b = bundle()
    X = build_features(pd.DataFrame([e.model_dump() for e in events]))
    proba = b["model"].predict_proba(X)[:, 1]
    return [
        {"blast_probability": round(float(p), 4),
         "label": "blast" if p >= b["threshold"] else "earthquake"}
        for p in proba
    ]
