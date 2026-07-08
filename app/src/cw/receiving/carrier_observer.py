from __future__ import annotations

import numpy as np

from cw.receiving.config import ReceivingConfig, carrier_alias_compare_hz, carrier_peak_separation_hz, effective_tracker_frame_ms, effective_tracker_hop_ms
from cw.receiving.models import CarrierObservation
from cw.receiving.spectrum import SpectrumPeak, baseband_envelope, detect_carriers_in_audio


class CarrierObserver:
    """Short-window spectrum observer for the receiving front-end."""

    def __init__(self, sample_rate: int, config: ReceivingConfig) -> None:
        self.sample_rate = int(sample_rate)
        self.config = config

    def observe(self, signal: np.ndarray) -> tuple[CarrierObservation, ...]:
        if len(signal) < max(1, int(self.sample_rate * 0.20)):
            return ()
        detected = detect_carriers_in_audio(
            signal,
            self.sample_rate,
            min_tone_hz=self.config.min_tone_hz,
            max_tone_hz=self.config.max_tone_hz,
            max_carriers=self.config.max_tracks,
            peak_separation_hz=carrier_peak_separation_hz(self.config),
            relative_threshold=self.config.peak_relative_threshold,
            min_snr_db=self.config.carrier_min_snr_db,
            frame_ms=effective_tracker_frame_ms(self.config),
            hop_ms=effective_tracker_hop_ms(self.config),
        )
        filtered = suppress_correlated_carrier_aliases(signal, self.sample_rate, detected, self.config)
        return tuple(
            CarrierObservation(
                carrier_hz=float(candidate.carrier_hz),
                relative_power=float(candidate.relative_power),
                snr_db=float(candidate.snr_db),
                power=float(candidate.power),
            )
            for candidate in filtered
        )


def suppress_correlated_carrier_aliases(
    signal: np.ndarray,
    sample_rate: int,
    candidates: tuple[SpectrumPeak, ...],
    config: ReceivingConfig,
) -> tuple[SpectrumPeak, ...]:
    candidates = tuple(candidates)
    if not config.alias_suppression or len(candidates) <= 1:
        return candidates
    selected: list[SpectrumPeak] = []
    traces: list[np.ndarray] = []
    for candidate in candidates:
        trace = _carrier_keying_trace(signal, sample_rate, float(candidate.carrier_hz), config)
        is_alias = False
        for kept, kept_trace in zip(selected, traces):
            if abs(float(candidate.carrier_hz) - float(kept.carrier_hz)) > carrier_alias_compare_hz(config):
                continue
            if _trace_correlation(trace, kept_trace) >= config.alias_correlation:
                is_alias = True
                break
        if is_alias:
            continue
        selected.append(candidate)
        traces.append(trace)
    return tuple(selected)


def _carrier_keying_trace(signal: np.ndarray, sample_rate: int, carrier_hz: float, config: ReceivingConfig) -> np.ndarray:
    envelope = baseband_envelope(
        signal,
        sample_rate,
        carrier_hz,
        lowpass_ms=max(5.0, config.hop_ms * 2.0),
    )
    if len(envelope) == 0:
        return envelope
    threshold = float(np.percentile(envelope, 70))
    return (envelope >= threshold).astype(np.float32)


def _trace_correlation(left: np.ndarray, right: np.ndarray) -> float:
    size = min(len(left), len(right))
    if size < 4:
        return 0.0
    a = left[:size].astype(np.float32, copy=False)
    b = right[:size].astype(np.float32, copy=False)
    if float(np.std(a)) <= 1e-6 or float(np.std(b)) <= 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
