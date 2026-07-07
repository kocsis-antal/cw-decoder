from __future__ import annotations

import numpy as np


def baseband_envelope(signal: np.ndarray, sample_rate: int, carrier_hz: float, *, lowpass_ms: float) -> np.ndarray:
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


def envelope_energy_frames(envelope: np.ndarray, sample_rate: int, *, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(round(sample_rate * hop_ms / 1000)))
    if len(envelope) == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    starts = np.arange(0, len(envelope), hop, dtype=np.int64)
    energy = np.empty(len(starts), dtype=np.float32)
    for out_index, start in enumerate(starts):
        frame = envelope[start : min(start + hop, len(envelope))]
        energy[out_index] = float(np.mean(frame * frame)) if len(frame) else 0.0
    times = starts.astype(np.float32) / float(sample_rate)
    return energy, times
