"""Typed configuration objects for the SpectroNet-LSTM pipeline.

Mirrors the hyperparameters described in the paper (Section 3.2-3.5):
- fixed-length audio normalization
- band-pass + wavelet denoising
- 72/8/20 train/val/test split
- Adam(lr=1e-4), categorical cross-entropy, early stopping (patience=15)
- fine-tuning: freeze all but output layer, then unfreeze last 50 layers
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict

import yaml

CLASSES = ["normal", "murmur", "extrastole", "artifact", "extrahls"]


@dataclasses.dataclass
class DataConfig:
    root_dir: str = "data/raw"
    sample_rate: int = 4000
    clip_seconds: float = 5.0
    classes: list = dataclasses.field(default_factory=lambda: list(CLASSES))
    train_split: float = 0.72
    val_split: float = 0.08
    test_split: float = 0.20
    random_seed: int = 42


@dataclasses.dataclass
class PreprocessConfig:
    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 400.0
    wavelet: str = "db4"
    wavelet_level: int = 4


@dataclasses.dataclass
class SpectrogramConfig:
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    image_size: int = 224
    n_mfcc: int = 40


@dataclasses.dataclass
class ModelConfig:
    backbones: list = dataclasses.field(
        default_factory=lambda: ["resnet101", "vgg16", "inception_v3"]
    )
    lstm_units: list = dataclasses.field(default_factory=lambda: [128, 64])
    dense_units: int = 128
    dropout: float = 0.3
    fine_tune_last_n_layers: int = 50


@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 32
    epochs_head: int = 20
    epochs_fine_tune: int = 20
    learning_rate: float = 1e-4
    early_stopping_patience: int = 15
    checkpoint_dir: str = "artifacts/checkpoints"


@dataclasses.dataclass
class BaselinesConfig:
    enabled: bool = True
    models: list = dataclasses.field(
        default_factory=lambda: ["simple_rnn", "gru", "tcn", "mlp"]
    )


@dataclasses.dataclass
class ExplainConfig:
    n_shap_background: int = 32
    n_lime_samples: int = 500
    output_dir: str = "artifacts/xai"


@dataclasses.dataclass
class PipelineConfig:
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    preprocess: PreprocessConfig = dataclasses.field(default_factory=PreprocessConfig)
    spectrogram: SpectrogramConfig = dataclasses.field(default_factory=SpectrogramConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    baselines: BaselinesConfig = dataclasses.field(default_factory=BaselinesConfig)
    explain: ExplainConfig = dataclasses.field(default_factory=ExplainConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "PipelineConfig":
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text())
        cfg = PipelineConfig()
        for section, values in (raw or {}).items():
            if not hasattr(cfg, section) or values is None:
                continue
            sub = getattr(cfg, section)
            for k, v in values.items():
                setattr(sub, k, v)
        return cfg
