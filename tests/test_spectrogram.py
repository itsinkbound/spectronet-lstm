import numpy as np

from spectronet.config import SpectrogramConfig
from spectronet.features.spectrogram import (
    compute_mel_spectrogram,
    compute_mfcc,
    spectrogram_to_rgb_image,
)


def _tone(sr=4000, seconds=2.0, freq=120.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_compute_mfcc_shape():
    cfg = SpectrogramConfig()
    sig = _tone()
    mfcc = compute_mfcc(sig, sample_rate=4000, cfg=cfg)
    assert mfcc.shape[0] == cfg.n_mfcc
    assert mfcc.ndim == 2


def test_compute_mel_spectrogram_shape():
    cfg = SpectrogramConfig()
    sig = _tone()
    mel = compute_mel_spectrogram(sig, sample_rate=4000, cfg=cfg)
    assert mel.shape[0] == cfg.n_mels


def test_spectrogram_to_rgb_image_shape_and_range():
    cfg = SpectrogramConfig(image_size=64)
    sig = _tone()
    mel = compute_mel_spectrogram(sig, sample_rate=4000, cfg=cfg)
    img = spectrogram_to_rgb_image(mel, cfg.image_size)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255
