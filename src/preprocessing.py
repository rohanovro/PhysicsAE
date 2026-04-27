"""
Signal preprocessing helpers.

Separated from data_loader.py so preprocessing logic can be tested
and reused independently.
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_signal(signal: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """Polyphase resample signal to *fs_target* Hz."""
    g = gcd(fs_target, fs_orig)
    return resample_poly(signal, fs_target // g, fs_orig // g).astype(np.float32)


def normalize_segment(segment: np.ndarray) -> np.ndarray:
    """Unit-variance normalisation."""
    std = segment.std()
    return segment / std if std > 1e-8 else segment


def compute_fft_magnitude(
    signal: np.ndarray,
    window_size: int,
    hop_size: int,
    apply_hann: bool = True,
) -> np.ndarray:
    """
    Compute sliding-window FFT magnitude spectrogram.

    Parameters
    ----------
    signal      : 1-D vibration signal
    window_size : samples per window
    hop_size    : hop between windows
    apply_hann  : if True, apply Hann window before FFT

    Returns
    -------
    spectrogram : (n_windows, n_bins) float32 array
    """
    hann = np.hanning(window_size) if apply_hann else np.ones(window_size)
    n_bins = window_size // 2 + 1
    n_windows = (len(signal) - window_size) // hop_size + 1
    out = np.zeros((n_windows, n_bins), dtype=np.float32)

    for i in range(n_windows):
        seg = signal[i * hop_size: i * hop_size + window_size]
        if len(seg) < window_size:
            break
        seg = normalize_segment(seg) * hann
        out[i] = np.abs(np.fft.rfft(seg)) / window_size

    return out


def freq_axis(n_fft: int, fs: int) -> np.ndarray:
    """Return the frequency axis corresponding to rfft output bins."""
    return np.fft.rfftfreq(n_fft, d=1.0 / fs)
