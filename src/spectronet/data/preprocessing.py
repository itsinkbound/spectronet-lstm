"""Signal preprocessing matching the paper's pipeline:

1. Flatten/normalize each recording to a fixed length (audio files in DHD
   range 1-120s; the paper fixes length before feature extraction).
2. Band-pass filter to isolate heart-sound frequency range (removes
   environmental / equipment noise).
3. Wavelet decomposition to denoise while preserving diagnostic transients.
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt


def fix_length(signal: np.ndarray, sample_rate: int, clip_seconds: float) -> np.ndarray:
    """Pad with zeros or truncate so every clip has identical length."""
    target_len = int(sample_rate * clip_seconds)
    if len(signal) >= target_len:
        return signal[:target_len]
    pad = target_len - len(signal)
    return np.pad(signal, (0, pad), mode="constant")


def bandpass_filter(
    signal: np.ndarray, sample_rate: int, low_hz: float, high_hz: float, order: int = 4
) -> np.ndarray:
    """Isolate the heart-sound frequency band (typically 20-400 Hz)."""
    nyquist = 0.5 * sample_rate
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.999)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def wavelet_denoise(signal: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Wavelet decomposition + soft-threshold denoising (removes noise while
    preserving S1/S2 transients, per Section 1 of the paper)."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    threshold = sigma * np.sqrt(2 * np.log(max(len(signal), 2)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
    ]
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    return reconstructed[: len(signal)]


def preprocess_signal(
    signal: np.ndarray,
    sample_rate: int,
    clip_seconds: float,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    wavelet: str,
    wavelet_level: int,
) -> np.ndarray:
    """Full preprocessing chain: flatten -> band-pass -> wavelet denoise."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        signal = np.zeros(int(sample_rate * clip_seconds), dtype=np.float32)
    signal = fix_length(signal, sample_rate, clip_seconds)
    signal = bandpass_filter(signal, sample_rate, bandpass_low_hz, bandpass_high_hz)
    signal = wavelet_denoise(signal, wavelet, wavelet_level)
    peak = np.max(np.abs(signal)) or 1.0
    return (signal / peak).astype(np.float32)
