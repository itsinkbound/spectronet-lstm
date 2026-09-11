import numpy as np

from spectronet.data.preprocessing import (
    bandpass_filter,
    fix_length,
    preprocess_signal,
    wavelet_denoise,
)


def test_fix_length_pads_short_signal():
    sig = np.ones(10, dtype=np.float32)
    out = fix_length(sig, sample_rate=10, clip_seconds=2.0)
    assert len(out) == 20
    assert np.all(out[10:] == 0.0)


def test_fix_length_truncates_long_signal():
    sig = np.arange(100, dtype=np.float32)
    out = fix_length(sig, sample_rate=10, clip_seconds=1.0)
    assert len(out) == 10
    np.testing.assert_array_equal(out, sig[:10])


def test_bandpass_filter_preserves_length():
    sr = 4000
    t = np.linspace(0, 1, sr, endpoint=False)
    sig = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    out = bandpass_filter(sig, sr, low_hz=20, high_hz=400)
    assert out.shape == sig.shape
    assert np.isfinite(out).all()


def test_wavelet_denoise_preserves_length():
    rng = np.random.RandomState(0)
    sig = rng.randn(2000).astype(np.float32)
    out = wavelet_denoise(sig)
    assert len(out) == len(sig)


def test_preprocess_signal_normalized_and_bounded():
    rng = np.random.RandomState(0)
    sig = rng.randn(4000).astype(np.float32) * 5
    out = preprocess_signal(
        sig,
        sample_rate=4000,
        clip_seconds=1.0,
        bandpass_low_hz=20,
        bandpass_high_hz=400,
        wavelet="db4",
        wavelet_level=3,
    )
    assert len(out) == 4000
    assert np.max(np.abs(out)) <= 1.0 + 1e-5


def test_preprocess_signal_handles_empty_input():
    out = preprocess_signal(
        np.array([]),
        sample_rate=100,
        clip_seconds=1.0,
        bandpass_low_hz=20,
        bandpass_high_hz=40,
        wavelet="db4",
        wavelet_level=2,
    )
    assert len(out) == 100
