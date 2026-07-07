from __future__ import annotations

import numpy as np
from cw.tools.legacy_decoder.base import _power_spectrum_frames
from cw.tools.legacy_decoder.carrier_detection import CarrierCandidate, _detect_carriers_from_spectrum

def detect_carriers_in_audio(
    signal: np.ndarray,
    sample_rate: int,
    *,
    min_tone_hz: float,
    max_tone_hz: float,
    max_carriers: int,
    min_separation_hz: float,
    relative_threshold: float,
    frame_ms: float,
    hop_ms: float,
) -> tuple[CarrierCandidate, ...]:
    if max_carriers <= 0 or len(signal) == 0:
        return ()
    spectrum, freqs = _power_spectrum_frames(signal, sample_rate, frame_ms, hop_ms)
    return _detect_carriers_from_spectrum(
        spectrum,
        freqs,
        min_tone_hz=min_tone_hz,
        max_tone_hz=max_tone_hz,
        max_carriers=max_carriers,
        min_separation_hz=min_separation_hz,
        relative_threshold=relative_threshold,
    )
