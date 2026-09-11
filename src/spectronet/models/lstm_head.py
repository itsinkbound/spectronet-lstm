"""The Conv1D + stacked-LSTM head described in Section 3.5 of the paper:

Conv1D -> MaxPooling -> BatchNorm -> LSTM x2 -> Dense -> Dropout -> Softmax.

This head consumes the fused CNN-backbone feature vector (reshaped into a
pseudo-sequence via RepeatVector, as shown in the paper's architecture
diagram, Fig. 2) and predicts one of the 5 heartbeat classes.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from spectronet.config import CLASSES, ModelConfig


def build_lstm_head(
    input_dim: int,
    model_cfg: ModelConfig,
    num_classes: int = len(CLASSES),
    sequence_length: int = 8,
) -> tf.keras.Model:
    """Builds the fine-tuned LSTM classification head.

    `sequence_length` recreates the "Repeat Vector" step in the paper's
    architecture diagram, turning the fused feature vector into a short
    pseudo-sequence so the stacked LSTM layers have temporal structure to
    operate over.
    """
    inputs = layers.Input(shape=(input_dim,), name="fused_features_input")

    x = layers.RepeatVector(sequence_length)(inputs)
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = layers.BatchNormalization()(x)

    for i, units in enumerate(model_cfg.lstm_units):
        return_sequences = i < len(model_cfg.lstm_units) - 1
        x = layers.LSTM(units, return_sequences=return_sequences)(x)

    x = layers.Dense(model_cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(model_cfg.dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="spectronet_lstm_head")
