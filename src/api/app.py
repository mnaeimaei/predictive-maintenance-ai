"""
app.py
──────
FastAPI REST service for predictive maintenance inference.

Start with:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.lstm_predictor import LSTMPredictor
from src.models.transformer_predictor import TransformerPredictor
from src.utils.preprocessing import load_scaler, SENSOR_COLS


# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "outputs/best_model.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "outputs/scaler.pkl")
DEVICE = torch.device("cpu")  # Use CPU for serving; GPU optional

ALERT_LEVELS = {
    (0.0, 0.40): "NORMAL",
    (0.40, 0.65): "WARNING",
    (0.65, 0.85): "HIGH",
    (0.85, 1.01): "CRITICAL",
}


def get_alert_level(prob: float) -> str:
    for (lo, hi), level in ALERT_LEVELS.items():
        if lo <= prob < hi:
            return level
    return "CRITICAL"


# ─── Model loading ────────────────────────────────────────────────────────────

_model = None
_scaler = None
_cfg = None
_model_meta = {}


def load_model():
    global _model, _scaler, _cfg, _model_meta

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    _cfg = ckpt["config"]
    n_features = ckpt["n_features"]
    model_type = ckpt["model_type"]

    if model_type == "lstm":
        mc = _cfg["model"]["lstm"]
        _model = LSTMPredictor(
            n_features=n_features,
            hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"],
            dropout=mc["dropout"],
            bidirectional=mc["bidirectional"],
            fc_hidden=_cfg["model"]["fc_hidden"],
        )
    else:
        mc = _cfg["model"]["transformer"]
        _model = TransformerPredictor(
            n_features=n_features,
            d_model=mc["d_model"],
            nhead=mc["nhead"],
            num_encoder_layers=mc["num_encoder_layers"],
            dim_feedforward=mc["dim_feedforward"],
            dropout=mc["dropout"],
        )

    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    _scaler = load_scaler(SCALER_PATH)

    _model_meta = {
        "model_type": model_type,
        "n_features": n_features,
        "n_parameters": _model.n_parameters,
        "trained_epoch": ckpt["epoch"],
        "val_f1": ckpt["val_f1"],
        "val_auc": ckpt.get("val_auc", None),
    }


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Predictive Maintenance AI",
    description=(
        "Deep learning API for industrial sensor failure detection "
        "and remaining useful life (RUL) estimation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    try:
        load_model()
        print(f"✅ Model loaded: {_model_meta['model_type'].upper()}  |  Params: {_model_meta['n_parameters']:,}")
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}. API will return 503 until model is present.")


# ─── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Single prediction request.
    sensor_window: list of timesteps, each with 14 sensor values.
    """
    sensor_window: list[list[float]] = Field(
        ...,
        description="2D array (T, 14) of sensor readings — one row per cycle",
        min_length=1,
    )
    unit_id: Optional[str] = Field(default="unknown", description="Equipment identifier")

    @field_validator("sensor_window")
    @classmethod
    def check_features(cls, v):
        for row in v:
            if len(row) != len(SENSOR_COLS):
                raise ValueError(f"Each timestep must have {len(SENSOR_COLS)} sensor values, got {len(row)}")
        return v


class PredictResponse(BaseModel):
    unit_id: str
    failure_probability: float
    failure_predicted: bool
    remaining_useful_life_cycles: float
    alert_level: str
    confidence: float
    timestamp: str
    inference_ms: float


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def preprocess_window(sensor_window: list[list[float]]) -> torch.Tensor:
    """Normalise and convert to tensor (1, T, F)."""
    arr = np.array(sensor_window, dtype=np.float32)          # (T, 14)
    arr[:, :len(SENSOR_COLS)] = _scaler.transform(arr[:, :len(SENSOR_COLS)])
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)     # (1, T, 14)


@torch.no_grad()
def run_prediction(sensor_window: list[list[float]]) -> dict:
    t0 = time.perf_counter()
    x = preprocess_window(sensor_window)
    prob, rul = _model(x)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    prob_val = float(prob.item())
    rul_val = float(rul.item()) * _cfg["data"]["rul_clip"]
    threshold = _cfg["evaluation"]["threshold"]

    return {
        "failure_probability": round(prob_val, 4),
        "failure_predicted": prob_val >= threshold,
        "remaining_useful_life_cycles": round(max(0.0, rul_val), 1),
        "alert_level": get_alert_level(prob_val),
        "confidence": round(max(prob_val, 1 - prob_val), 4),
        "inference_ms": round(elapsed_ms, 2),
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    ready = _model is not None
    return {"status": "ok" if ready else "model_not_loaded", "ready": ready}


@app.get("/model/info", tags=["Model"])
def model_info():
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    return _model_meta


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded — train a model first.")
    result = run_prediction(req.sensor_window)
    return PredictResponse(
        unit_id=req.unit_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        **result,
    )


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(req: BatchPredictRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")
    results = []
    for r in req.requests:
        result = run_prediction(r.sensor_window)
        results.append({"unit_id": r.unit_id, "timestamp": datetime.now(timezone.utc).isoformat(), **result})
    return {"predictions": results, "count": len(results)}
