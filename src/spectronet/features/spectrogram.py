"""Spectrogram / MFCC generation.

The paper converts denoised audio to (a) MFCCs for the LSTM branch's raw
acoustic features and (b) mel-spectrogram "heat map" images fed to the
image-pretrained CNN backbones (ResNet101 / VGG16 / InceptionV3).
"""
from __future__ import annotations

import numpy as np
import librosa

from spectronet.config import SpectrogramConfig


def compute_mfcc(signal: np.ndarray, sample_rate: int, cfg: SpectrogramConfig) -> np.ndarray:
    """Mel-Frequency Cepstral Coefficients, shape (n_mfcc, T)."""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    return mfcc.astype(np.float32)


def compute_mel_spectrogram(
    signal: np.ndarray, sample_rate: int, cfg: SpectrogramConfig
) -> np.ndarray:
    """Log-mel spectrogram, shape (n_mels, T)."""
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def spectrogram_to_rgb_image(log_mel: np.ndarray, image_size: int) -> np.ndarray:
    """Normalize a log-mel spectrogram to a 3-channel uint8 image sized for
    ImageNet-pretrained backbones (224x224x3 by default)."""
    normed = (log_mel - log_mel.min()) / (np.ptp(log_mel) + 1e-8)
    normed = (normed * 255.0).astype(np.uint8)

    # resize via simple nearest-neighbour index mapping to avoid a PIL/cv2
    # hard dependency in the core library
    freq_bins, time_bins = normed.shape
    freq_idx = np.linspace(0, freq_bins - 1, image_size).astype(int)
    time_idx = np.linspace(0, time_bins - 1, image_size).astype(int)
    resized = normed[freq_idx][:, time_idx]

    rgb = np.stack([resized, resized, resized], axis=-1)
    return rgb


def signal_to_backbone_input(
    signal: np.ndarray, sample_rate: int, cfg: SpectrogramConfig
) -> np.ndarray:
    """End-to-end: denoised signal -> mel spectrogram -> RGB image tensor."""
    log_mel = compute_mel_spectrogram(signal, sample_rate, cfg)
    return spectrogram_to_rgb_image(log_mel, cfg.image_size)
