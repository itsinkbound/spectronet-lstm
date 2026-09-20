"""Single-file inference for SpectroNet-LSTM.

Loads a trained checkpoint once and exposes `Predictor.predict_file()`,
which takes one arbitrary uploaded audio file (no label, no manifest) and
returns a per-class probability dict. This is the code path a future
API/frontend layer should call — it is deliberately separate from
pipeline.py's --stage evaluate, which requires a labeled manifest.
"""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from spectronet.config import CLASSES, PipelineConfig
from spectronet.data.preprocessing import preprocess_signal
from spectronet.features.spectrogram import signal_to_backbone_input
from spectronet.models.fusion_model import build_fusion_model

log = logging.getLogger("spectronet.inference")

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg"}


class PredictionError(ValueError):
    """Raised when an uploaded file can't be turned into a valid prediction."""


class Predictor:
    """Loads the model once; reuse the instance across many predictions.

    Usage:
        predictor = Predictor(cfg)          # load weights once, e.g. at API startup
        result = predictor.predict_file("clip.wav")
    """

    def __init__(self, cfg: PipelineConfig, checkpoint_path: str | None = None):
        self.cfg = cfg
        default_path = Path(cfg.train.checkpoint_dir) / "spectronet_lstm_fine_tuned.keras"
        path = Path(checkpoint_path) if checkpoint_path else default_path
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {path}. Train a model first (--stage train/all) "
                f"or point Predictor at an existing .keras file."
            )
        self.model = build_fusion_model(cfg.model)
        self.model.load_weights(str(path))
        log.info("Predictor ready - loaded %s", path)

    def predict_file(self, audio_path: str) -> dict[str, float]:
        path = Path(audio_path)
        if not path.exists():
            raise PredictionError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise PredictionError(
                f"Unsupported file type '{path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        try:
            signal, _ = librosa.load(str(path), sr=self.cfg.data.sample_rate, mono=True)
        except Exception as exc:
            raise PredictionError(f"Could not decode audio file: {exc}") from exc

        if signal.size == 0:
            raise PredictionError("Audio file contains no samples (empty or silent-only decode).")

        duration_s = signal.size / self.cfg.data.sample_rate
        if duration_s < 0.5:
            raise PredictionError(
                f"Clip too short ({duration_s:.2f}s) to analyze reliably; need at least 0.5s."
            )

        clean = preprocess_signal(
            signal,
            sample_rate=self.cfg.data.sample_rate,
            clip_seconds=self.cfg.data.clip_seconds,
            bandpass_low_hz=self.cfg.preprocess.bandpass_low_hz,
            bandpass_high_hz=self.cfg.preprocess.bandpass_high_hz,
            wavelet=self.cfg.preprocess.wavelet,
            wavelet_level=self.cfg.preprocess.wavelet_level,
        )
        image = signal_to_backbone_input(clean, self.cfg.data.sample_rate, self.cfg.spectrogram)
        batch = np.expand_dims(image.astype(np.float32), axis=0)

        probs = self.model.predict(batch, verbose=0)[0]
        result = {cls: float(p) for cls, p in zip(CLASSES, probs, strict=True)}
        log.info("Predicted %s -> %s", path.name, max(result, key=result.get))
        return result


def predict_file(
    audio_path: str,
    config_path: str = "configs/config.yaml",
    checkpoint_path: str | None = None,
) -> dict[str, float]:
    """One-shot convenience wrapper for a CLI/script. For a server handling
    many requests, instantiate Predictor once instead - reloading three CNN
    backbones per request is slow."""
    cfg = PipelineConfig.from_yaml(config_path)
    predictor = Predictor(cfg, checkpoint_path=checkpoint_path)
    return predictor.predict_file(audio_path)


def _cli():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run SpectroNet-LSTM inference on one audio file")
    parser.add_argument("audio_path")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    result = predict_file(args.audio_path, config_path=args.config, checkpoint_path=args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()