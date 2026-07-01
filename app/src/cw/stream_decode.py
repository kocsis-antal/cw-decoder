from __future__ import annotations

import numpy as np

from cw.decoder import (
    DecodeResult,
    DecoderConfig,
    DetectedRun,
    _classified_runs_to_tokens,
    _energy_threshold,
    _runs_from_activity,
    classify_runs,
)
from cw.morse_table import decode_tokens
from cw.multi_decoder import _local_peak_indices
from cw.quality import score_decode_result
from cw.stream_models import SpectrumFrame, StreamingConfig, StreamSessionResult, peak_min_separation_hz


def detect_accumulated_carriers(
    frames: list[SpectrumFrame],
    config: StreamingConfig,
) -> list[tuple[float, float, float]]:
    if not frames:
        return []

    freqs = frames[-1].freqs
    summed = np.sum([frame.spectrum for frame in frames], axis=0)
    search_mask = (freqs >= config.min_tone_hz) & (freqs <= config.max_tone_hz)
    if not np.any(search_mask):
        return []

    search_freqs = freqs[search_mask]
    powers = summed[search_mask]
    if len(powers) == 0:
        return []
    max_power = float(np.max(powers))
    if max_power <= 0:
        return []

    candidates = _local_peak_indices(powers)
    candidates.sort(key=lambda index: float(powers[index]), reverse=True)

    selected: list[tuple[float, float, float]] = []
    for index in candidates:
        power = float(powers[index])
        relative_power = power / max_power
        if relative_power < config.peak_relative_threshold:
            continue
        frequency_hz = float(search_freqs[index])
        if any(abs(frequency_hz - existing_hz) < peak_min_separation_hz(config) for existing_hz, _r, _p in selected):
            continue
        selected.append((frequency_hz, relative_power, power))
        if len(selected) >= config.max_tracks:
            break
    return selected


def decode_carrier_sessions_from_frames(
    frames: list[SpectrumFrame],
    carrier_hz: float,
    config: StreamingConfig,
    final_time_s: float,
) -> list[StreamSessionResult]:
    if not frames:
        return []

    energy = np.asarray(
        [_band_energy(frame.spectrum, frame.freqs, carrier_hz, config.bandwidth_hz) for frame in frames],
        dtype=np.float32,
    )
    if len(energy) == 0 or float(np.max(energy)) <= 0:
        return []

    decoder_config = DecoderConfig(
        frame_ms=config.frame_ms,
        hop_ms=config.hop_ms,
        min_tone_hz=config.min_tone_hz,
        max_tone_hz=config.max_tone_hz,
        bandwidth_hz=config.bandwidth_hz,
        threshold_ratio=config.threshold_ratio,
        target_tone_hz=carrier_hz,
    )
    threshold = _energy_threshold(energy, decoder_config)
    active = energy > threshold
    runs = _offset_runs(_runs_from_activity(active, config.hop_ms / 1000), frames[0].start_s)
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return []

    gap_threshold_s = max(config.min_session_gap_s, config.session_gap_units * unit_s)
    segments = _split_runs_by_session_gap(runs, gap_threshold_s, final_time_s)
    sessions: list[StreamSessionResult] = []
    for index, (segment_runs, session_final_time_s, reason) in enumerate(segments, start=1):
        decoded = _decode_run_segment(segment_runs, carrier_hz, threshold)
        if not decoded.text:
            continue
        quality = score_decode_result(decoded)
        first_seen_s, last_seen_s = _active_time_bounds(segment_runs)
        hits = _tone_hit_count(segment_runs, config.hop_ms / 1000)
        sessions.append(
            StreamSessionResult(
                session_id=index,
                first_seen_s=round(first_seen_s, 3),
                last_seen_s=round(last_seen_s, 3),
                hits=hits,
                final_time_s=round(session_final_time_s, 3),
                final_reason=reason,
                quality=quality,
                decoded=decoded,
            )
        )
    return sessions



def _offset_runs(runs: list[DetectedRun], offset_s: float) -> list[DetectedRun]:
    if not runs or abs(offset_s) < 1e-12:
        return runs
    return [
        DetectedRun(
            kind=run.kind,
            start_s=run.start_s + offset_s,
            duration_s=run.duration_s,
        )
        for run in runs
    ]

def _split_runs_by_session_gap(
    runs: list[DetectedRun],
    gap_threshold_s: float,
    final_time_s: float,
) -> list[tuple[list[DetectedRun], float, str]]:
    segments: list[tuple[list[DetectedRun], float, str]] = []
    current: list[DetectedRun] = []

    for run in runs:
        if run.kind == "gap" and run.duration_s >= gap_threshold_s and _has_tone(current):
            segments.append((current, run.start_s + gap_threshold_s, "silence_gap"))
            current = []
            continue
        if current or run.kind == "tone":
            current.append(run)

    if _has_tone(current):
        segments.append((current, final_time_s, "end_of_stream"))
    return segments


def _decode_run_segment(
    runs: list[DetectedRun],
    carrier_hz: float,
    threshold: float,
) -> DecodeResult:
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return _empty_decode(carrier_hz, threshold=threshold, runs=runs)
    classified_runs = classify_runs(runs, unit_s)
    tokens = _classified_runs_to_tokens(classified_runs)
    text = decode_tokens(tokens)
    return DecodeResult(
        text=text,
        tokens=tokens,
        runs=runs,
        classified_runs=classified_runs,
        carrier_hz=carrier_hz,
        unit_s=unit_s,
        threshold=threshold,
    )


def _has_tone(runs: list[DetectedRun]) -> bool:
    return any(run.kind == "tone" for run in runs)


def _tone_hit_count(runs: list[DetectedRun], hop_s: float) -> int:
    if hop_s <= 0:
        return sum(1 for run in runs if run.kind == "tone")
    return sum(max(1, round(run.duration_s / hop_s)) for run in runs if run.kind == "tone")


def _active_time_bounds(runs: list[DetectedRun]) -> tuple[float, float]:
    tones = [run for run in runs if run.kind == "tone"]
    if not tones:
        return 0.0, 0.0
    return tones[0].start_s, tones[-1].start_s + tones[-1].duration_s


def _estimate_unit_from_runs(runs: list[DetectedRun]) -> float:
    from cw.decoder import _estimate_unit_s

    return _estimate_unit_s(runs)


def _empty_decode(
    carrier_hz: float,
    *,
    threshold: float = 0.0,
    runs: list[DetectedRun] | None = None,
) -> DecodeResult:
    return DecodeResult(
        text="",
        tokens=[],
        runs=runs or [],
        classified_runs=[],
        carrier_hz=carrier_hz,
        unit_s=0.0,
        threshold=threshold,
    )


def _band_energy(spectrum: np.ndarray, freqs: np.ndarray, carrier_hz: float, bandwidth_hz: float) -> float:
    mask = np.abs(freqs - carrier_hz) <= bandwidth_hz
    if not np.any(mask):
        mask[np.argmin(np.abs(freqs - carrier_hz))] = True
    return float(spectrum[mask].sum())


