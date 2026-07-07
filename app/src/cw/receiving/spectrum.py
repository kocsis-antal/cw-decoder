from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectrumPeak:
    carrier_hz: float
    relative_power: float
    power: float
    snr_db: float = 0.0


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    mono = signal.astype(np.float32, copy=False)
    if len(mono) < frame_length:
        mono = np.pad(mono, (0, frame_length - len(mono)))
    starts = range(0, len(mono) - frame_length + 1, hop_length)
    frames = [mono[start : start + frame_length] for start in starts]
    return np.stack(frames).astype(np.float32, copy=False)


def power_spectrum_frames(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame_length = max(1, round(sample_rate * frame_ms / 1000))
    hop_length = max(1, round(sample_rate * hop_ms / 1000))
    frames = frame_signal(signal, frame_length, hop_length)
    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame_length, 1 / sample_rate)
    return spectrum, freqs


def detect_carriers_in_audio(
    signal: np.ndarray,
    sample_rate: int,
    *,
    min_tone_hz: float,
    max_tone_hz: float,
    max_carriers: int,
    min_separation_hz: float,
    relative_threshold: float,
    min_snr_db: float = 0.0,
    frame_ms: float = 30.0,
    hop_ms: float,
) -> tuple[SpectrumPeak, ...]:
    if max_carriers <= 0 or len(signal) == 0:
        return ()
    spectrum, freqs = power_spectrum_frames(signal, sample_rate, frame_ms, hop_ms)
    return detect_carriers_from_spectrum(
        spectrum,
        freqs,
        min_tone_hz=min_tone_hz,
        max_tone_hz=max_tone_hz,
        max_carriers=max_carriers,
        min_separation_hz=min_separation_hz,
        relative_threshold=relative_threshold,
        min_snr_db=min_snr_db,
    )


def detect_carriers_from_spectrum(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    min_tone_hz: float,
    max_tone_hz: float,
    max_carriers: int,
    min_separation_hz: float,
    relative_threshold: float,
    min_snr_db: float = 0.0,
) -> tuple[SpectrumPeak, ...]:
    if max_carriers <= 0 or len(spectrum) == 0:
        return ()
    summed = np.sum(spectrum, axis=0)
    mask = (freqs >= min_tone_hz) & (freqs <= max_tone_hz)
    if not np.any(mask):
        return ()
    powers = summed[mask]
    search_freqs = freqs[mask]
    max_power = float(np.max(powers)) if len(powers) else 0.0
    if max_power <= 0:
        return ()

    noise_floor = _spectrum_noise_floor(powers)
    candidates = _local_peak_indices(powers)
    candidates.sort(key=lambda index: float(powers[index]), reverse=True)
    selected: list[SpectrumPeak] = []
    for index in candidates:
        power = float(powers[index])
        relative = power / max_power
        snr_db = 10.0 * float(np.log10(max(power, 1e-30) / max(noise_floor, 1e-30)))
        if relative < relative_threshold:
            continue
        if snr_db < min_snr_db:
            continue
        carrier_hz = float(search_freqs[index])
        if any(abs(carrier_hz - existing.carrier_hz) < min_separation_hz for existing in selected):
            continue
        selected.append(SpectrumPeak(round(carrier_hz, 3), round(relative, 6), power, round(snr_db, 3)))
        if len(selected) >= max_carriers:
            break
    return tuple(selected)


def _spectrum_noise_floor(powers: np.ndarray) -> float:
    if len(powers) == 0:
        return 1e-30
    finite = powers[np.isfinite(powers)]
    if len(finite) == 0:
        return 1e-30
    # Median-bin power is a robust local noise reference for live carrier
    # observation.  A relative peak threshold alone accepts arbitrary FFT
    # maxima during silence, because every noise window has a strongest bin.
    return max(float(np.median(finite)), 1e-30)


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


def _local_peak_indices(values: np.ndarray) -> list[int]:
    if len(values) == 1:
        return [0]
    peaks: list[int] = []
    for index, value in enumerate(values):
        left = values[index - 1] if index > 0 else -np.inf
        right = values[index + 1] if index < len(values) - 1 else -np.inf
        if value >= left and value >= right and (value > left or value > right):
            peaks.append(index)
    return peaks
