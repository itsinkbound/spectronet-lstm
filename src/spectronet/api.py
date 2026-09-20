"""FastAPI serving layer for SpectroNet-LSTM single-file inference.

Run with: uvicorn spectronet.api:app --reload
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from spectronet.config import PipelineConfig
from spectronet.inference import PredictionError, Predictor

log = logging.getLogger("spectronet.api")

app = FastAPI(title="SpectroNet-LSTM Inference API")

# Allow a local frontend (served from a different port/file://) to call this API.
# Tighten this to your actual frontend's origin before deploying anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Predictor | None = None

# Below this gap between the top two classes, treat the call as uncertain
# rather than confidently reporting the top class. Tune this once you have
# a feel for how the model behaves across more real uploads.
LOW_CONFIDENCE_MARGIN = 0.15


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    low_confidence: bool
    probabilities: dict[str, float]


@app.on_event("startup")
def load_model():
    global _predictor
    cfg = PipelineConfig.from_yaml("configs/config.yaml")
    _predictor = Predictor(cfg)
    log.info("Model loaded, API ready")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        probs = _predictor.predict_file(tmp_path)
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_class, top_conf = ranked[0]
    runner_up_conf = ranked[1][1]
    low_confidence = (top_conf - runner_up_conf) < LOW_CONFIDENCE_MARGIN

    return PredictionResponse(
        prediction=top_class,
        confidence=top_conf,
        low_confidence=low_confidence,
        probabilities=probs,
    )