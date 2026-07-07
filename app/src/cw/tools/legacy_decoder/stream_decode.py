from __future__ import annotations

from dataclasses import replace

import numpy as np

from cw.tools.legacy_decoder.base import (
    ClassifiedRun,
    DecodeResult,
    DecoderConfig as BasicDecoderConfig,
    DetectedRun,
    _classified_runs_to_tokens,
    _energy_threshold,
    _runs_from_activity,
    classify_runs,
)
from cw.morse_table import decode_tokens
from cw.tools.legacy_decoder.quality import score_decode_result
from cw.tools.legacy_decoder.stream_models import SpectrumFrame, StreamSessionResult
from cw.tools.legacy_decoder.config import DecoderConfig, peak_min_separation_hz




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

def detect_accumulated_carriers(
    frames: list[SpectrumFrame],
    config: DecoderConfig,
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
    config: DecoderConfig,
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

    threshold_ratios = _threshold_candidate_ratios(config)
    if len(threshold_ratios) == 1:
        return _decode_sessions_for_threshold_ratio(
            energy,
            frames[0].start_s,
            carrier_hz,
            threshold_ratios[0],
            config,
            final_time_s,
        )

    candidates: list[StreamSessionResult] = []
    for threshold_ratio in threshold_ratios:
        candidates.extend(
            _decode_sessions_for_threshold_ratio(
                energy,
                frames[0].start_s,
                carrier_hz,
                threshold_ratio,
                config,
                final_time_s,
            )
        )
    return _select_threshold_session_candidates(candidates, config)


def _decode_sessions_for_threshold_ratio(
    energy: np.ndarray,
    first_frame_start_s: float,
    carrier_hz: float,
    threshold_ratio: float,
    config: DecoderConfig,
    final_time_s: float,
) -> list[StreamSessionResult]:
    decoder_config = BasicDecoderConfig(
        frame_ms=config.frame_ms,
        hop_ms=config.hop_ms,
        min_tone_hz=config.min_tone_hz,
        max_tone_hz=config.max_tone_hz,
        bandwidth_hz=config.bandwidth_hz,
        threshold_ratio=threshold_ratio,
        target_tone_hz=carrier_hz,
    )
    threshold = _energy_threshold(energy, decoder_config)
    active = energy > threshold
    runs = _offset_runs(_runs_from_activity(active, config.hop_ms / 1000), first_frame_start_s)
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
    segments = _split_runs_by_session_gap(
        runs,
        gap_threshold_s,
        final_time_s,
        finalization_delay_s=config.finalization_delay_s,
    )
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


def _threshold_candidate_ratios(config: DecoderConfig) -> tuple[float, ...]:
    ratios = [config.threshold_ratio, *config.threshold_ratios]
    unique: list[float] = []
    for ratio in ratios:
        rounded = round(float(ratio), 6)
        if rounded not in unique:
            unique.append(rounded)
    return tuple(sorted(unique))


def _select_threshold_session_candidates(
    candidates: list[StreamSessionResult],
    config: DecoderConfig,
) -> list[StreamSessionResult]:
    if not candidates:
        return []

    # A single fixed threshold can split a real transmission into several
    # neat-looking fragments while a lower threshold keeps the whole over
    # together.  The earlier implementation grouped candidates by *any*
    # overlap and picked the lowest raw timing score; that often discarded the
    # longer, more complete decode just because a short fragment had a prettier
    # score.  Select a non-overlapping candidate set instead: alternatives that
    # cover the same time compete, but two genuinely separate overs can both be
    # kept.  The reward is content-neutral: decoded evidence length minus signal
    # quality penalties, with no CQ/DE/callsign bias.
    tolerance_s = max(config.hop_ms / 1000 * 8, 0.06)
    ordered = sorted(candidates, key=lambda item: (item.last_seen_s, item.first_seen_s, item.quality.score))
    previous_indices = [_previous_non_overlapping_index(ordered, index, tolerance_s) for index in range(len(ordered))]

    # Dynamic programming table.  Each value is (selection_score, tiebreak_tuple, chosen_indices).
    # The tiebreak tuple is maximized, so negative quality means lower aggregate
    # quality score wins when the evidence reward is effectively tied.
    dp: list[tuple[float, tuple[float, int, int, int], list[int]]] = [(0.0, (0.0, 0, 0, 0), [])]
    for index, candidate in enumerate(ordered):
        skip = dp[index]
        prev = dp[previous_indices[index] + 1]
        reward = _threshold_candidate_reward(candidate, config)
        tie = _threshold_candidate_tiebreak(candidate, config)
        take = (
            prev[0] + reward,
            (
                prev[1][0] + tie[0],
                prev[1][1] + tie[1],
                prev[1][2] + tie[2],
                prev[1][3] + tie[3],
            ),
            [*prev[2], index],
        )
        dp.append(max(skip, take, key=lambda item: (round(item[0], 6), item[1])))

    chosen_indices = dp[-1][2]
    if not chosen_indices:
        chosen = [min(candidates, key=lambda item: _threshold_choice_score(item, config))]
    else:
        chosen = [ordered[index] for index in chosen_indices]
    chosen.sort(key=lambda item: (item.first_seen_s, item.last_seen_s))
    return [replace(session, session_id=index) for index, session in enumerate(chosen, start=1)]


def _previous_non_overlapping_index(
    sessions: list[StreamSessionResult],
    index: int,
    tolerance_s: float,
) -> int:
    current = sessions[index]
    for previous_index in range(index - 1, -1, -1):
        if sessions[previous_index].last_seen_s <= current.first_seen_s - tolerance_s:
            return previous_index
    return -1


def _sessions_overlap(left: StreamSessionResult, right: StreamSessionResult, tolerance_s: float) -> bool:
    return left.first_seen_s <= right.last_seen_s + tolerance_s and right.first_seen_s <= left.last_seen_s + tolerance_s


def _threshold_choice_score(session: StreamSessionResult, config: DecoderConfig) -> tuple[float, int, int, float]:
    text = session.decoded.text
    unknowns = text.count("?")
    known_chars = _known_text_chars(text)
    punctuation_count = _punctuation_count(text)
    score = session.quality.score + punctuation_count * config.punctuation_penalty
    return (score, unknowns, -known_chars, session.quality.score)


def _threshold_candidate_reward(session: StreamSessionResult, config: DecoderConfig) -> float:
    text = session.decoded.text
    known_chars = _known_text_chars(text)
    if known_chars <= 0:
        return -1000.0
    unknowns = text.count("?")
    punctuation_count = _punctuation_count(text)
    score = session.quality.score + punctuation_count * config.punctuation_penalty
    # A known decoded character is useful evidence; bad timing/unknowns/punctuation
    # reduce confidence.  These constants intentionally do not encode QSO words.
    return known_chars * 2.0 - score * 0.25 - unknowns * 1.5 - punctuation_count * 0.5


def _threshold_candidate_tiebreak(session: StreamSessionResult, config: DecoderConfig) -> tuple[float, int, int, int]:
    text = session.decoded.text
    punctuation_count = _punctuation_count(text)
    score = session.quality.score + punctuation_count * config.punctuation_penalty
    return (-score, _known_text_chars(text), -text.count("?"), -punctuation_count)


def _known_text_chars(text: str) -> int:
    return sum(1 for char in text if not char.isspace() and char != "?")


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
    *,
    finalization_delay_s: float = 0.0,
) -> list[tuple[list[DetectedRun], float, str]]:
    segments: list[tuple[list[DetectedRun], float, str]] = []
    current: list[DetectedRun] = []

    for run in runs:
        if run.kind == "gap" and run.duration_s >= gap_threshold_s and _has_tone(current):
            close_time_s = run.start_s + gap_threshold_s
            if close_time_s + finalization_delay_s <= final_time_s:
                segments.append((current, close_time_s, "silence_gap"))
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
    config: DecoderConfig,
) -> DecodeResult:
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return _empty_decode(carrier_hz, threshold=threshold, runs=runs)

    candidates = _unit_candidates(unit_s, config.unit_candidate_spread, config.unit_candidate_steps)
    decoded_candidates = [_decode_with_unit(runs, carrier_hz, threshold, candidate, config) for candidate in candidates]
    return min(
        decoded_candidates,
        key=lambda result: _decode_choice_score(result, config.punctuation_penalty),
    )


def _decode_with_unit(
    runs: list[DetectedRun],
    carrier_hz: float,
    threshold: float,
    unit_s: float,
    config: DecoderConfig | None = None,
) -> DecodeResult:
    classified_runs = (
        _classify_runs_with_adaptive_gaps(runs, unit_s, config) if config is not None else classify_runs(runs, unit_s)
    )
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


def _classify_runs_with_adaptive_gaps(
    runs: list[DetectedRun],
    unit_s: float,
    config: DecoderConfig,
) -> list[ClassifiedRun]:
    if not config.adaptive_gap_thresholds:
        return classify_runs(runs, unit_s)

    element_letter_units = config.element_letter_gap_units
    letter_word_units = _adaptive_letter_word_boundary_units(runs, unit_s, config)
    classified_runs: list[ClassifiedRun] = []

    for run in runs:
        units = run.duration_s / unit_s
        if run.kind == "tone":
            symbol = "." if units < 2 else "-"
        elif units < element_letter_units:
            symbol = "element_gap"
        elif units < letter_word_units:
            symbol = "letter_gap"
        else:
            symbol = "word_gap"

        classified_runs.append(
            ClassifiedRun(
                kind=run.kind,
                start_s=run.start_s,
                duration_s=run.duration_s,
                symbol=symbol,
                units=round(units, 3),
            )
        )

    return classified_runs


def _adaptive_letter_word_boundary_units(
    runs: list[DetectedRun],
    unit_s: float,
    config: DecoderConfig,
) -> float:
    """Estimate a session-local letter/word gap split from received timing.

    Human CW and filtered streaming audio often stretch normal inter-letter gaps
    beyond the textbook 3 units.  The old fixed boundary at 5 units then turned
    ``CQ`` into ``C Q``.  This estimator treats gaps as timing observations: it
    only creates a word-gap cluster when the longer gaps are clearly separated
    from the letter-gap cluster.  Without a reliable split, moderately long gaps
    remain letters, while very long pauses still become words.
    """

    if unit_s <= 0:
        return config.default_word_gap_units

    element_letter_units = config.element_letter_gap_units
    gaps = sorted(
        run.duration_s / unit_s
        for run in runs
        if run.kind == "gap" and run.duration_s > 0 and run.duration_s / unit_s >= element_letter_units
    )
    if len(gaps) < 2:
        return config.default_word_gap_units

    best_index = _best_letter_word_gap_split(gaps, config, min_upper_count=2)
    if best_index is None:
        best_index = _best_letter_word_gap_split(gaps, config, min_upper_count=1)

    if best_index is not None:
        left = gaps[best_index]
        right = gaps[best_index + 1]
        return (left * right) ** 0.5

    return config.default_word_gap_units


def _best_letter_word_gap_split(
    gaps: list[float],
    config: DecoderConfig,
    *,
    min_upper_count: int,
) -> int | None:
    best_index: int | None = None
    best_ratio = 0.0
    best_delta = 0.0
    for index, (left, right) in enumerate(zip(gaps, gaps[1:])):
        if left <= 0:
            continue
        lower_count = index + 1
        upper_count = len(gaps) - lower_count
        if lower_count < config.gap_cluster_min_lower_count or upper_count < min_upper_count:
            continue
        ratio = right / left
        delta = right - left
        if ratio < config.gap_cluster_min_ratio or delta < config.gap_cluster_min_delta_units:
            continue
        # Prefer the largest absolute separation.  A second pass with a single
        # upper outlier is allowed only when no repeated word-gap cluster exists.
        if delta > best_delta or (delta == best_delta and ratio > best_ratio):
            best_index = index
            best_ratio = ratio
            best_delta = delta
    return best_index

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
    from cw.tools.legacy_decoder.base import _estimate_unit_s

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


