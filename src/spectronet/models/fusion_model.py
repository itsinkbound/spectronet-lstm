"""End-to-end SpectroNet-LSTM model (ResNet101 + VGG16 + InceptionV3 fusion
-> LSTM head), plus the five baseline architectures used for the paper's
Table 1 / radar-plot comparison (Simple RNN, GRU, TCN, MLP, CNN-LSTM which
the paper itself calls "SpectroNet-LSTM" as a baseline distinct from the
fine-tuned combined model -- kept here as `cnn_lstm_baseline` to avoid
ambiguity with the final fine-tuned model).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from spectronet.config import CLASSES, ModelConfig
from spectronet.features.backbone_extractor import (
    build_multi_backbone_feature_extractor,
    fuse_backbone_outputs,
    unfreeze_last_n_layers,
)
from spectronet.models.lstm_head import build_lstm_head


def build_fusion_model(
    model_cfg: ModelConfig, image_size: int = 224, num_classes: int = len(CLASSES)
) -> tf.keras.Model:
    """The full combined model: 3 CNN backbones -> feature fusion -> LSTM head."""
    image_input = layers.Input(shape=(image_size, image_size, 3), name="spectrogram_image")
    backbones = build_multi_backbone_feature_extractor(model_cfg.backbones, trainable=False)
    fused = fuse_backbone_outputs(image_input, backbones)

    head = build_lstm_head(input_dim=fused.shape[-1], model_cfg=model_cfg, num_classes=num_classes)
    outputs = head(fused)

    model = models.Model(inputs=image_input, outputs=outputs, name="spectronet_lstm_fusion")
    model._backbones = backbones  # keep handles for the fine-tuning phase
    return model


def enable_fine_tuning(model: tf.keras.Model, n_layers: int) -> None:
    """Phase 2 training: unlock the last `n_layers` of every backbone."""
    for backbone in getattr(model, "_backbones", {}).values():
        unfreeze_last_n_layers(backbone, n_layers)


# ---------------------------------------------------------------------------
# Baseline sequence models (Table 1 comparison), operating directly on MFCC
# sequences of shape (T, n_mfcc) rather than spectrogram images.
# ---------------------------------------------------------------------------

def build_simple_rnn(input_shape, num_classes: int = len(CLASSES)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.SimpleRNN(64, return_sequences=True)(inputs)
    x = layers.SimpleRNN(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="simple_rnn")


def build_gru(input_shape, num_classes: int = len(CLASSES)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.GRU(64, return_sequences=True)(inputs)
    x = layers.GRU(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="gru")


def _temporal_block(x, filters: int, dilation_rate: int):
    y = layers.Conv1D(filters, kernel_size=3, padding="causal", dilation_rate=dilation_rate,
                       activation="relu")(x)
    y = layers.BatchNormalization()(y)
    return layers.Conv1D(filters, kernel_size=3, padding="causal", dilation_rate=dilation_rate,
                          activation="relu")(y)


def build_tcn(input_shape, num_classes: int = len(CLASSES)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for dilation in (1, 2, 4, 8):
        x = _temporal_block(x, filters=64, dilation_rate=dilation)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="tcn")


def build_mlp(input_shape, num_classes: int = len(CLASSES)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="mlp")


def build_cnn_lstm_baseline(input_shape, num_classes: int = len(CLASSES)) -> tf.keras.Model:
    """The paper's own "SpectroNet-LSTM" baseline row in Table 1: a plain
    CNN+LSTM hybrid without the multi-backbone fusion."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="cnn_lstm_baseline")


BASELINE_BUILDERS = {
    "simple_rnn": build_simple_rnn,
    "gru": build_gru,
    "tcn": build_tcn,
    "mlp": build_mlp,
    "cnn_lstm_baseline": build_cnn_lstm_baseline,
}
