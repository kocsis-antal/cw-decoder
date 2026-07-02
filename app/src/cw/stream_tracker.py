from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cw.multi_decoder import _local_peak_indices
from cw.stream_models import (
    SpectrumFrame,
    StreamingConfig,
    channel_merge_hz,
    peak_min_separation_hz,
    track_match_hz,
)


@dataclass(frozen=True)
class SpectralPeak:
    time_s: float
    frequency_hz: float
    power: float
    relative_power: float
    snr_db: float = 0.0


@dataclass
class CarrierTrack:
    track_id: int
    frequency_hz: float
    first_seen_s: float
    last_seen_s: float
    hits: int = 1
    power: float = 0.0
    max_power: float = 0.0
    max_snr_db: float = 0.0
    state: str = "active"

    def observe(self, peak: SpectralPeak, smoothing: float) -> None:
        smoothing = min(1.0, max(0.0, smoothing))
        self.frequency_hz = (1.0 - smoothing) * self.frequency_hz + smoothing * peak.frequency_hz
        self.last_seen_s = peak.time_s
        self.hits += 1
        self.power = peak.power
        self.max_power = max(self.max_power, peak.power)
        self.max_snr_db = max(self.max_snr_db, peak.snr_db)
        self.state = "active"

    def mark_time(self, time_s: float, max_gap_s: float) -> None:
        if self.state == "closed":
            return
        gap_s = time_s - self.last_seen_s
        if gap_s > max_gap_s:
            self.state = "dormant"
        else:
            self.state = "active"


class CarrierTracker:
    """Frame-by-frame carrier tracker for streaming CW.

    This is intentionally lightweight: every STFT frame yields local spectral
    peaks; peaks are greedily matched to existing carrier tracks by frequency;
    tracks keep a smoothed carrier estimate and become dormant when they have
    not been seen for ``max_track_gap_s``.
    """

    def __init__(self, config: StreamingConfig) -> None:
        self.config = config
        self._next_track_id = 1
        self._tracks: list[CarrierTrack] = []

    @property
    def tracks(self) -> list[CarrierTrack]:
        return self._tracks

    def update(self, frame: SpectrumFrame) -> list[CarrierTrack]:
        peaks = detect_frame_peaks(frame, self.config)
        self.update_peaks(peaks, frame.start_s)
        return self.active_tracks(frame.start_s)

    def update_peaks(self, peaks: list[SpectralPeak], time_s: float) -> None:
        match_hz = track_match_hz(self.config)
        unmatched_tracks = [track for track in self._tracks if track.state != "closed"]

        for peak in sorted(peaks, key=lambda item: item.power, reverse=True):
            candidates = [
                track
                for track in unmatched_tracks
                if abs(track.frequency_hz - peak.frequency_hz) <= match_hz
            ]
            if candidates:
                track = min(candidates, key=lambda item: abs(item.frequency_hz - peak.frequency_hz))
                track.observe(peak, self.config.carrier_smoothing)
                unmatched_tracks.remove(track)
            else:
                self._tracks.append(
                    CarrierTrack(
                        track_id=self._next_track_id,
                        frequency_hz=peak.frequency_hz,
                        first_seen_s=peak.time_s,
                        last_seen_s=peak.time_s,
                        hits=1,
                        power=peak.power,
                        max_power=peak.power,
                        max_snr_db=peak.snr_db,
                    )
                )
                self._next_track_id += 1

        for track in self._tracks:
            track.mark_time(time_s, self.config.max_track_gap_s)

    def active_tracks(self, time_s: float | None = None) -> list[CarrierTrack]:
        if time_s is not None:
            for track in self._tracks:
                track.mark_time(time_s, self.config.max_track_gap_s)
        tracks = [
            track
            for track in self._tracks
            if track.state == "active" and track.hits >= self.config.min_track_hits
        ]
        tracks.sort(key=lambda item: item.max_power, reverse=True)
        return tracks[: self.config.max_tracks]

    def candidate_carriers(self, time_s: float | None = None) -> list[tuple[float, float, float]]:
        tracks = self.active_tracks(time_s)
        if not tracks:
            return []
        reference_power = max((track.max_power for track in self._tracks), default=0.0)
        if reference_power <= 0:
            return []

        carriers: list[tuple[float, float, float]] = []
        for track in tracks:
            relative_power = track.max_power / reference_power
            if relative_power < self.config.track_relative_threshold:
                continue
            if track.max_snr_db < self.config.min_peak_snr_db:
                continue
            if any(abs(track.frequency_hz - carrier_hz) < channel_merge_hz(self.config) for carrier_hz, _r, _p in carriers):
                continue
            carriers.append((float(track.frequency_hz), float(relative_power), float(track.max_power)))
            if len(carriers) >= self.config.max_tracks:
                break
        return carriers


def detect_frame_peaks(frame: SpectrumFrame, config: StreamingConfig) -> list[SpectralPeak]:
    freqs = frame.freqs
    mask = (freqs >= config.min_tone_hz) & (freqs <= config.max_tone_hz)
    if not np.any(mask):
        return []

    search_freqs = freqs[mask]
    powers = frame.spectrum[mask]
    if len(powers) == 0:
        return []

    max_power = float(np.max(powers))
    if max_power <= 0:
        return []

    floor_power = _spectral_floor_power(powers)

    candidates = _local_peak_indices(powers)
    if not candidates:
        candidates = list(range(len(powers)))
    candidates.sort(key=lambda index: float(powers[index]), reverse=True)

    selected: list[SpectralPeak] = []
    for index in candidates:
        power = float(powers[index])
        relative_power = power / max_power if max_power > 0 else 0.0
        if relative_power < config.peak_relative_threshold:
            continue
        snr_db = _power_ratio_db(power, floor_power)
        if snr_db < config.min_peak_snr_db:
            continue
        frequency_hz = float(search_freqs[index])
        if any(abs(frequency_hz - existing.frequency_hz) < peak_min_separation_hz(config) for existing in selected):
            continue
        selected.append(
            SpectralPeak(
                time_s=round(frame.start_s, 6),
                frequency_hz=frequency_hz,
                power=power,
                relative_power=relative_power,
                snr_db=snr_db,
            )
        )
        if len(selected) >= config.max_tracks:
            break
    return selected


def _spectral_floor_power(powers: np.ndarray) -> float:
    """Robust per-frame noise floor estimate for carrier squelch."""

    if len(powers) == 0:
        return 0.0
    return float(np.percentile(powers, 50))


def _power_ratio_db(power: float, floor_power: float) -> float:
    if power <= 0:
        return float("-inf")
    if floor_power <= 0:
        return float("inf")
    return float(10.0 * np.log10(power / floor_power))
