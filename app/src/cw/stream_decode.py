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
    runs = smooth_keying_runs(
        runs,
        merge_short_gaps_s=config.merge_short_gaps_ms / 1000,
        drop_short_tones_s=config.drop_short_tones_ms / 1000,
    )
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return []

    gap_threshold_s = max(config.min_session_gap_s, config.session_gap_units * unit_s)
    segments = _split_runs_by_session_gap(runs, gap_threshold_s, final_time_s)
    sessions: list[StreamSessionResult] = []
    for index, (segment_runs, session_final_time_s, reason) in enumerate(segments, start=1):
        decoded = _decode_run_segment(segment_runs, carrier_hz, threshold, config)
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




def smooth_keying_runs(
    runs: list[DetectedRun],
    *,
    merge_short_gaps_s: float = 0.0,
    drop_short_tones_s: float = 0.0,
) -> list[DetectedRun]:
    """Remove short keying glitches before estimating Morse timing.

    Real received audio is not a clean rectangular keying envelope.  AGC,
    filters, flutter, WebSDR audio processing, and noise can briefly push a
    dash below threshold.  Without this small debounce step a single dash can
    become several dots, producing plausible but wrong punctuation such as
    ``C=`` instead of ``CQ``.
    """

    if not runs or (merge_short_gaps_s <= 0 and drop_short_tones_s <= 0):
        return runs

    cleaned = list(runs)
    if drop_short_tones_s > 0:
        cleaned = _drop_short_tone_runs(cleaned, drop_short_tones_s)
    if merge_short_gaps_s > 0:
        cleaned = _merge_short_internal_gaps(cleaned, merge_short_gaps_s)
    return cleaned


def _drop_short_tone_runs(runs: list[DetectedRun], min_tone_s: float) -> list[DetectedRun]:
    converted = [
        DetectedRun("gap", run.start_s, run.duration_s)
        if run.kind == "tone" and run.duration_s < min_tone_s
        else run
        for run in runs
    ]
    return _merge_adjacent_runs(converted)


def _merge_short_internal_gaps(runs: list[DetectedRun], max_gap_s: float) -> list[DetectedRun]:
    merged: list[DetectedRun] = []
    index = 0
    while index < len(runs):
        run = runs[index]
        if run.kind != "tone":
            merged.append(run)
            index += 1
            continue

        start_s = run.start_s
        end_s = run.start_s + run.duration_s
        index += 1
        while (
            index + 1 < len(runs)
            and runs[index].kind == "gap"
            and runs[index].duration_s <= max_gap_s
            and runs[index + 1].kind == "tone"
        ):
            end_s = runs[index + 1].start_s + runs[index + 1].duration_s
            index += 2
        merged.append(DetectedRun("tone", start_s, round(end_s - start_s, 10)))
    return _merge_adjacent_runs(merged)


def _merge_adjacent_runs(runs: list[DetectedRun]) -> list[DetectedRun]:
    if not runs:
        return []
    merged: list[DetectedRun] = []
    for run in runs:
        if merged and merged[-1].kind == run.kind:
            previous = merged[-1]
            end_s = max(previous.start_s + previous.duration_s, run.start_s + run.duration_s)
            merged[-1] = DetectedRun(previous.kind, previous.start_s, round(end_s - previous.start_s, 10))
        else:
            merged.append(run)
    return merged


def _unit_candidates(unit_s: float, spread: float, steps: int) -> list[float]:
    if unit_s <= 0:
        return []
    if spread <= 0 or steps <= 1:
        return [unit_s]
    if steps % 2 == 0:
        steps += 1
    lower = max(unit_s * (1.0 - spread), 0.001)
    upper = unit_s * (1.0 + spread)
    candidates = [round(float(value), 10) for value in np.linspace(lower, upper, steps)]
    if unit_s not in candidates:
        candidates.append(unit_s)
    return sorted(set(candidates))


def _decode_choice_score(result: DecodeResult, punctuation_penalty: float) -> tuple[float, int, int, float]:
    quality = score_decode_result(result)
    punctuation_count = _punctuation_count(result.text)
    score = quality.score + punctuation_count * punctuation_penalty
    known_chars = len([char for char in result.text.replace(" ", "") if char != "?"])
    # Prefer lower score, then more known characters, fewer punctuation marks,
    # and finally the original unit estimate neighbourhood by lower raw score.
    return (score, -known_chars, punctuation_count, quality.score)


def _punctuation_count(text: str) -> int:
    return sum(1 for char in text if char and not char.isspace() and not char.isalnum() and char != "?")


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
    config: StreamingConfig,
) -> DecodeResult:
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return _empty_decode(carrier_hz, threshold=threshold, runs=runs)

    candidates = _unit_candidates(unit_s, config.unit_candidate_spread, config.unit_candidate_steps)
    decoded_candidates = [_decode_with_unit(runs, carrier_hz, threshold, candidate) for candidate in candidates]
    return min(
        decoded_candidates,
        key=lambda result: _decode_choice_score(result, config.punctuation_penalty),
    )


def _decode_with_unit(
    runs: list[DetectedRun],
    carrier_hz: float,
    threshold: float,
    unit_s: float,
) -> DecodeResult:
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


