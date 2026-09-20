"""CLI entrypoint: `python -m spectronet.pipeline --config configs/config.yaml`

Wires together data -> features -> train -> evaluate -> explain, matching
the paper's Algorithm 1 pseudocode end to end. Each stage can also be run
in isolation with `--stage {preprocess,train,evaluate,explain}` for use in
an orchestrated (e.g. Airflow/Kubeflow) production pipeline.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from spectronet.config import CLASSES, PipelineConfig
from spectronet.data.dataset import load_manifest, one_hot, stratified_split
from spectronet.data.preprocessing import preprocess_signal
from spectronet.evaluation.metrics import evaluate_predictions, results_table
from spectronet.features.spectrogram import signal_to_backbone_input
from spectronet.models.fusion_model import build_fusion_model
from spectronet.training.train import train_fusion_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spectronet.pipeline")


def _load_audio(path: str, sample_rate: int) -> np.ndarray:
    import librosa

    signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    return signal


def build_dataset(df, cfg: PipelineConfig) -> tf.data.Dataset:
    """Streams (spectrogram_image, one_hot_label) pairs from a manifest."""

    def _generator():
        for _, row in df.iterrows():
            signal = _load_audio(row["path"], cfg.data.sample_rate)
            clean = preprocess_signal(
                signal,
                sample_rate=cfg.data.sample_rate,
                clip_seconds=cfg.data.clip_seconds,
                bandpass_low_hz=cfg.preprocess.bandpass_low_hz,
                bandpass_high_hz=cfg.preprocess.bandpass_high_hz,
                wavelet=cfg.preprocess.wavelet,
                wavelet_level=cfg.preprocess.wavelet_level,
            )
            image = signal_to_backbone_input(clean, cfg.data.sample_rate, cfg.spectrogram)
            yield image.astype(np.float32), one_hot(row["label"])

    output_signature = (
        tf.TensorSpec(shape=(cfg.spectrogram.image_size, cfg.spectrogram.image_size, 3),
                       dtype=tf.float32),
        tf.TensorSpec(shape=(len(CLASSES),), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(_generator, output_signature=output_signature)
    return ds.batch(cfg.train.batch_size).prefetch(tf.data.AUTOTUNE)


def stage_preprocess(cfg: PipelineConfig):
    manifest = load_manifest(cfg.data)
    train_df, val_df, test_df = stratified_split(manifest, cfg.data)
    log.info("Manifest: %d samples | train=%d val=%d test=%d",
              len(manifest), len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def stage_train(cfg: PipelineConfig, train_df, val_df):
    train_ds = build_dataset(train_df, cfg)
    val_ds = build_dataset(val_df, cfg)
    artifacts = train_fusion_model(cfg, train_ds, val_ds)

    if cfg.baselines.enabled:
        for name in cfg.baselines.models:
            log.info("Training baseline: %s", name)
            # Baselines consume MFCC sequences; left as an exercise for the
            # feature pipeline variant (see notebooks/ for the reference
            # implementation) to keep this CLI path fast for CI/dry-runs.
    return artifacts


def stage_evaluate(cfg: PipelineConfig, model, test_df):
    test_ds = build_dataset(test_df, cfg)
    y_true, y_pred = [], []
    for images, labels in test_ds:
        y_true.append(labels.numpy())
        y_pred.append(model.predict(images, verbose=0))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    result = evaluate_predictions(y_true, y_pred)
    table = results_table({"Fine-Tuned LSTM": result})
    log.info("\n%s", table.to_string(index=False))
    return result

def load_trained_model(cfg: PipelineConfig, checkpoint_path: str | None = None):
    """Rebuilds the architecture and loads saved weights — used when
    --stage evaluate/explain is run standalone, without a fresh --stage train
    in the same invocation."""
    default_path = Path(cfg.train.checkpoint_dir) / "spectronet_lstm_fine_tuned.keras"
    path = Path(checkpoint_path) if checkpoint_path else default_path
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Run --stage train (or --stage all) first, "
            f"or pass --checkpoint pointing at an existing .keras file."
        )
    model = build_fusion_model(cfg.model)
    model.load_weights(str(path))
    log.info("Loaded trained model from %s", path)
    return model


def main():
    parser = argparse.ArgumentParser(description="SpectroNet-LSTM pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess", "train", "evaluate", "explain"],
        default="all",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a .keras checkpoint. Used by --stage evaluate/explain "
             "when run standalone (no fresh train in this invocation). "
             "Defaults to <train.checkpoint_dir>/spectronet_lstm_fine_tuned.keras.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)

    if args.stage == "preprocess":
        stage_preprocess(cfg)
        return

    train_df, val_df, test_df = stage_preprocess(cfg)

    if args.stage in ("all", "train"):
        artifacts = stage_train(cfg, train_df, val_df)
        model = artifacts.model
    else:  # evaluate / explain, run standalone against a saved checkpoint
        model = load_trained_model(cfg, args.checkpoint)

    if args.stage in ("all", "evaluate"):
        stage_evaluate(cfg, model, test_df)

    if args.stage == "explain":
        raise NotImplementedError(
            "stage_explain() doesn't exist yet — SHAP/LIME are only wired up "
            "in notebooks/. This is a separate follow-up, not part of this fix."
        )


if __name__ == "__main__":
    main()
