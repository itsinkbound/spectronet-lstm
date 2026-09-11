"""Two-phase training loop matching Section 3.2/3.3 of the paper:

Phase 1: all backbone layers frozen, only the LSTM head trains (Adam,
lr=1e-4, categorical cross-entropy, early stopping patience=15).
Phase 2 (fine-tuning): last `fine_tune_last_n_layers` layers of each
backbone unlocked and trained jointly with the head at a low learning rate.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import tensorflow as tf

from spectronet.config import PipelineConfig
from spectronet.models.fusion_model import build_fusion_model, enable_fine_tuning


@dataclasses.dataclass
class TrainingArtifacts:
    model: tf.keras.Model
    history_head: tf.keras.callbacks.History
    history_fine_tune: tf.keras.callbacks.History | None


def _callbacks(cfg: PipelineConfig, tag: str):
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.train.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / f"spectronet_lstm_{tag}.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def train_fusion_model(
    cfg: PipelineConfig,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
) -> TrainingArtifacts:
    """Runs both training phases and returns the fitted model + histories."""
    model = build_fusion_model(cfg.model, image_size=cfg.spectrogram.image_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.train.epochs_head,
        callbacks=_callbacks(cfg, tag="head"),
        verbose=2,
    )

    enable_fine_tuning(model, cfg.model.fine_tune_last_n_layers)
    # Lower learning rate during fine-tuning to avoid destroying pretrained
    # weights, per standard transfer-learning practice referenced in the paper.
    fine_tune_lr = cfg.train.learning_rate / 10.0
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine_tune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.train.epochs_fine_tune,
        callbacks=_callbacks(cfg, tag="fine_tuned"),
        verbose=2,
    )

    return TrainingArtifacts(
        model=model, history_head=history_head, history_fine_tune=history_fine_tune
    )


def train_baseline(
    name: str,
    build_fn,
    cfg: PipelineConfig,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    input_shape,
) -> TrainingArtifacts:
    """Trains one of the Table-1 baseline models on MFCC sequences."""
    model = build_fn(input_shape=input_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.train.epochs_head,
        callbacks=_callbacks(cfg, tag=name),
        verbose=2,
    )
    return TrainingArtifacts(model=model, history_head=history, history_fine_tune=None)
