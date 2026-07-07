from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

@dataclass(frozen=True)
class CarrierCandidate:
    carrier_hz: float
    relative_power: float
    power: float

def _detect_carriers_from_spectrum(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    min_tone_hz: float,
    max_tone_hz: float,
    max_carriers: int,
    min_separation_hz: float,
    relative_threshold: float,
) -> tuple[CarrierCandidate, ...]:
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

    # Keep carrier detection deliberately simple and conservative: choose real
    # accumulated spectral peaks.  The previous temporal/local-peak admission was
    # too eager in streaming WebSDR audio and promoted sidebands/noise shadows to
    # separate public carriers, which made the receiver look busy but less useful.
    candidates = _local_peak_indices(powers)
    candidates.sort(key=lambda index: float(powers[index]), reverse=True)
    selected: list[CarrierCandidate] = []
    for index in candidates:
        power = float(powers[index])
        relative = power / max_power
        if relative < relative_threshold:
            continue
        carrier_hz = float(search_freqs[index])
        if any(abs(carrier_hz - existing.carrier_hz) < min_separation_hz for existing in selected):
            continue
        selected.append(CarrierCandidate(round(carrier_hz, 3), round(relative, 6), power))
        if len(selected) >= max_carriers:
            break
    return tuple(selected)
