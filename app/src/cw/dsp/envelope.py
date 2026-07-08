from __future__ import annotations

import numpy as np


def baseband_envelope(signal: np.ndarray, sample_rate: int, carrier_hz: float, *, lowpass_ms: float) -> np.ndarray:
    """Mix one carrier to baseband and return its smoothed amplitude envelope."""

    mono = signal.astype(np.float32, copy=False)
    if len(mono) == 0:
        return np.asarray([], dtype=np.float32)
    time = np.arange(len(mono), dtype=np.float32) / float(sample_rate)
    mixed = mono * np.exp(-2j * np.pi * float(carrier_hz) * time)
    window_len = max(5, int(round(sample_rate * lowpass_ms / 1000)))
    if window_len % 2 == 0:
        window_len += 1
    window = np.hanning(window_len).astype(np.float32)
    if float(np.sum(window)) <= 0:
        window = np.ones(window_len, dtype=np.float32)
    window = window / np.sum(window)
    filtered_real = np.convolve(mixed.real, window, mode="same")
    filtered_imag = np.convolve(mixed.imag, window, mode="same")
    envelope = np.sqrt(filtered_real * filtered_real + filtered_imag * filtered_imag)
    return envelope.astype(np.float32, copy=False)
