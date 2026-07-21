from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from cw.morse_table import CHAR_BY_MORSE, DECODE_ERROR_MARKER, decode_tokens_detailed
from cw.decoder.tokens import DecodeToken, gap_token, char_token, unknown_token, tokens_to_text
from cw.signal.models import SignalRun as SignalLayerRun
from cw.signal.models import SignalState, SignalTrack


class RunState(Enum):
    MARK = "mark"
    SPACE = "space"
    UNKNOWN = "unknown"


class HardRunKind(Enum):
    TONE = "tone"
    GAP = "gap"


@dataclass(frozen=True)
class TimedRun:
    state: RunState
    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class HardRun:
    kind: HardRunKind
    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class ClassifiedRun:
    kind: HardRunKind
    start_s: float
    duration_s: float
    symbol: str
    units: float


@dataclass(frozen=True)
class TimedDecode:
    text: str
    unresolved_tokens: int
    tokens: tuple[str, ...]
    decode_tokens: tuple[DecodeToken, ...]
    runs: tuple[HardRun, ...]
    classified_runs: tuple[ClassifiedRun, ...]
    unit_s: float


@dataclass(frozen=True)
class TimingBoundaries:
    dot_dash_units: float
    element_letter_units: float
    letter_word_units: float


def timed_runs_from_signal_track(track: SignalTrack) -> tuple[TimedRun, ...]:
    output: list[TimedRun] = []
    cursor = 0.0
    for run in track.runs:
        duration_s = max(0.0, float(run.duration_s))
        if duration_s <= 0:
            continue
        output.append(TimedRun(state=_to_decoder_state(run), start_s=round(cursor, 6), duration_s=round(duration_s, 6)))
        cursor += duration_s
    return tuple(_merge_adjacent_timed_runs(output))


def _to_decoder_state(run: SignalLayerRun) -> RunState:
    if run.state is SignalState.MARK:
        return RunState.MARK
    if run.state is SignalState.SPACE:
        return RunState.SPACE
    return RunState.UNKNOWN


def expand_unknown_runs(runs: tuple[TimedRun, ...]) -> list[list[HardRun]]:
    """Expand each UNKNOWN run locally into every MARK/SPACE alternative.

    This is intentionally not a global "all unknowns are MARK" / "all unknowns
    are SPACE" switch.  Every UNKNOWN run branches independently, and each
    produced hard-run path can therefore contain a different local assignment.

    The decoder does not apply an answer-count limit here.  Resource control is
    handled before this step by UNKNOWN-count / UNKNOWN-ratio gates.  Once a
    track is accepted as sufficiently known, all local UNKNOWN interpretations
    are kept for selection.
    """

    paths: list[list[HardRun]] = [[]]
    for run in runs:
        states = (RunState.MARK, RunState.SPACE) if run.state is RunState.UNKNOWN else (run.state,)
        expanded: list[list[HardRun]] = []
        for path in paths:
            for state in states:
                next_path = list(path)
                _append_hard_run(next_path, _hard_run_from_state(run, state))
                expanded.append(next_path)
        paths = expanded
    return paths


def _hard_run_from_state(run: TimedRun, state: RunState) -> HardRun:
    if state is RunState.UNKNOWN:
        raise ValueError("UNKNOWN must be resolved before hard-run decoding")
    kind = HardRunKind.TONE if state is RunState.MARK else HardRunKind.GAP
    return HardRun(kind=kind, start_s=run.start_s, duration_s=run.duration_s)


def _append_hard_run(runs: list[HardRun], run: HardRun) -> None:
    if runs and runs[-1].kind is run.kind:
        previous = runs[-1]
        end_s = max(previous.end_s, run.end_s)
        runs[-1] = HardRun(previous.kind, previous.start_s, round(end_s - previous.start_s, 10))
    else:
        runs.append(run)


def decode_segment_with_unit(runs: list[HardRun], unit_s: float, config) -> TimedDecode:
    decoded = decode_segments_with_unit(runs, unit_s, config)
    if decoded:
        return decoded[0]
    classified_runs = classify_runs(runs, unit_s, config)
    decode_tokens = _classified_runs_to_decode_tokens(classified_runs, trim_edges=False)
    morse_tokens = _decode_tokens_to_legacy_morse_tokens(decode_tokens)
    return TimedDecode(
        text=tokens_to_text(decode_tokens),
        unresolved_tokens=sum(1 for token in decode_tokens if token.kind == "unknown"),
        tokens=morse_tokens,
        decode_tokens=decode_tokens,
        runs=tuple(runs),
        classified_runs=tuple(classified_runs),
        unit_s=unit_s,
    )


def decode_segments_with_unit(runs: list[HardRun], unit_s: float, config) -> list[TimedDecode]:
    """Decode one hard-run path with channel-local adaptive timing.

    Hand-sent CW is not decoded with fixed machine timing.  The dot/dash and
    element/letter/word boundaries are estimated from the current channel
    window and then clamped to sane Morse ranges.  The decoder deliberately does
    not use QSO-wording hints.  For each candidate unit it returns the single
    timing interpretation implied by the measured channel boundaries.
    """

    if unit_s <= 0:
        return []
    classified_runs = classify_runs(runs, unit_s, config)
    decode_tokens = _classified_runs_to_decode_tokens(classified_runs, trim_edges=False)
    return [
        TimedDecode(
            text=tokens_to_text(decode_tokens),
            unresolved_tokens=sum(1 for token in decode_tokens if token.kind == "unknown"),
            tokens=_decode_tokens_to_legacy_morse_tokens(decode_tokens),
            decode_tokens=decode_tokens,
            runs=tuple(runs),
            classified_runs=tuple(classified_runs),
            unit_s=unit_s,
        )
    ]

def classify_runs(runs: list[HardRun], unit_s: float, config) -> list[ClassifiedRun]:
    boundaries = _timing_boundaries(runs, unit_s, config)
    classified: list[ClassifiedRun] = []
    for run in runs:
        classified_run = _classify_run(run, unit_s, boundaries, config)
        if classified_run is not None:
            classified.append(classified_run)
    return classified


def _timing_boundaries(runs: list[HardRun], unit_s: float, config) -> TimingBoundaries:
    dot_dash_units = (
        _adaptive_dot_dash_boundary_units(runs, unit_s, config)
        if bool(getattr(config, "adaptive_tone_thresholds", True))
        else float(getattr(config, "dot_dash_boundary_units", 2.0))
    )
    element_letter_units = (
        _adaptive_element_letter_boundary_units(runs, unit_s, config)
        if config.adaptive_gap_thresholds and getattr(config, "adaptive_element_letter_gap", False)
        else config.element_letter_gap_units
    )
    letter_word_units = _adaptive_letter_word_boundary_units(runs, unit_s, config) if config.adaptive_gap_thresholds else 5.0
    # Keep the boundary order sane even for very short or very noisy windows.
    min_gap_between_boundaries = float(getattr(config, "min_gap_boundary_separation_units", 0.55))
    letter_word_units = max(float(letter_word_units), float(element_letter_units) + min_gap_between_boundaries)
    return TimingBoundaries(
        dot_dash_units=round(float(dot_dash_units), 6),
        element_letter_units=round(float(element_letter_units), 6),
        letter_word_units=round(float(letter_word_units), 6),
    )


def _classify_run(
    run: HardRun,
    unit_s: float,
    boundaries: TimingBoundaries,
    config,
) -> ClassifiedRun | None:
    units = run.duration_s / unit_s if unit_s > 0 else 0.0
    if run.kind is HardRunKind.TONE:
        symbol = "." if units < float(boundaries.dot_dash_units) else "-"
    else:
        symbol = _gap_symbol(units, boundaries, config)
    return ClassifiedRun(
        kind=run.kind,
        start_s=round(float(run.start_s), 6),
        duration_s=round(float(run.duration_s), 6),
        symbol=symbol,
        units=round(float(units), 3),
    )


def _gap_symbol(units: float, boundaries: TimingBoundaries, config) -> str:
    element_letter = float(boundaries.element_letter_units)
    letter_word = float(boundaries.letter_word_units)
    session_gap = float(getattr(config, "session_gap_units", float(getattr(config, "default_word_gap_units", 7.0)) * 2.0))
    if units >= session_gap:
        return "session_gap"
    if units < element_letter:
        return "element_gap"
    if units < letter_word:
        return "letter_gap"
    return "word_gap"


def _adaptive_dot_dash_boundary_units(runs: list[HardRun], unit_s: float, config) -> float:
    fallback = float(getattr(config, "dot_dash_boundary_units", 2.0))
    if unit_s <= 0:
        return fallback
    tones = sorted(
        run.duration_s / unit_s
        for run in runs
        if run.kind is HardRunKind.TONE and run.duration_s > 0
    )
    if len(tones) < 3:
        return min(max(fallback, float(getattr(config, "min_dot_dash_boundary_units", 1.45))), float(getattr(config, "max_dot_dash_boundary_units", 2.65)))

    min_boundary = float(getattr(config, "min_dot_dash_boundary_units", 1.45))
    max_boundary = float(getattr(config, "max_dot_dash_boundary_units", 2.65))
    min_ratio = float(getattr(config, "tone_cluster_min_ratio", 1.55))
    min_delta = float(getattr(config, "tone_cluster_min_delta_units", 0.55))

    best_boundary: float | None = None
    best_score = 0.0
    for index, (lower, upper) in enumerate(zip(tones, tones[1:])):
        if lower <= 0:
            continue
        lower_count = index + 1
        upper_count = len(tones) - lower_count
        if lower_count < 1 or upper_count < 1:
            continue
        ratio = upper / lower
        delta = upper - lower
        boundary = (lower * upper) ** 0.5
        if ratio < min_ratio or delta < min_delta:
            continue
        if boundary < min_boundary or boundary > max_boundary:
            continue
        # Prefer a split where both clusters are populated, but keep it simple.
        balance = min(lower_count, upper_count) / max(lower_count, upper_count)
        score = ratio * delta * (0.5 + balance)
        if score > best_score:
            best_score = score
            best_boundary = boundary

    if best_boundary is None:
        return min(max(fallback, min_boundary), max_boundary)
    return round(float(best_boundary), 6)


def _adaptive_element_letter_boundary_units(runs: list[HardRun], unit_s: float, config) -> float:
    """Estimate the element/letter gap boundary from the current gap cluster.

    A fixed 2.6-unit boundary is too rigid for live CW: envelope thresholds can
    shave a nominal 3-unit letter gap down to roughly 2.1 units, while real
    element gaps stay near 1 unit.  Use the observed gap distribution when it
    contains a clear low/high split, but clamp the result so a noisy long pause
    cannot make every small space a letter boundary.
    """

    fallback = float(config.element_letter_gap_units)
    if unit_s <= 0:
        return fallback
    gaps = sorted(
        run.duration_s / unit_s
        for run in runs
        if run.kind is HardRunKind.GAP and run.duration_s > 0
    )
    if len(gaps) < 3:
        return fallback

    # Ignore very long word/session gaps while estimating the element/letter
    # split.  They belong to the next boundary and would otherwise dominate the
    # largest-ratio search.
    candidate_gaps = [gap for gap in gaps if gap <= float(config.default_word_gap_units) * 0.90]
    if len(candidate_gaps) < 3:
        candidate_gaps = gaps

    best_boundary: float | None = None
    best_score = 0.0
    for index, (lower, upper) in enumerate(zip(candidate_gaps, candidate_gaps[1:])):
        if lower <= 0:
            continue
        lower_count = index + 1
        upper_count = len(candidate_gaps) - lower_count
        if lower_count < int(config.gap_cluster_min_lower_count) or upper_count < 1:
            continue
        ratio = upper / lower
        delta = upper - lower
        if ratio < float(config.gap_cluster_min_ratio) or delta < 0.45:
            continue
        boundary = (lower + upper) / 2.0
        if boundary < float(config.min_element_letter_gap_units) or boundary > float(config.max_element_letter_gap_units):
            continue
        score = ratio * delta
        if score > best_score:
            best_score = score
            best_boundary = boundary

    if best_boundary is None:
        return min(max(fallback, float(config.min_element_letter_gap_units)), float(config.max_element_letter_gap_units))
    return round(float(best_boundary), 6)


def estimate_unit_s(runs: list[HardRun]) -> float:
    durations = np.asarray(
        [run.duration_s for run in runs if run.kind is HardRunKind.TONE and run.duration_s > 0],
        dtype=np.float64,
    )
    if len(durations) == 0:
        raise ValueError("No tone runs found")

    if len(durations) == 1:
        return round(float(durations[0]), 10)

    minimum = float(np.min(durations))
    maximum = float(np.max(durations))
    lower = max(minimum * 0.6, 0.001)
    upper = max(min(maximum / 2, maximum), lower)
    candidates = np.linspace(lower, upper, 300, dtype=np.float64)

    # This function is called for every UNKNOWN-expansion path.  The original
    # implementation evaluated the same grid with Python's min/sum loops, which
    # became the live bottleneck: a few seconds of CW could spend many seconds
    # just estimating the unit length.  Keep the same scoring model, but evaluate
    # the full candidate grid in NumPy so every audio block can continue moving
    # through the live pipeline.
    dot_error = np.abs(durations[:, None] - candidates[None, :])
    dash_error = np.abs(durations[:, None] - 3.0 * candidates[None, :])
    costs = np.minimum(dot_error, dash_error).sum(axis=0)
    return round(float(candidates[int(np.argmin(costs))]), 10)


def unit_candidates(unit_s: float, spread: float, steps: int) -> list[float]:
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


def timing_quality_score(decoded: TimedDecode) -> float:
    """Measure only the physical Morse timing fit. Lower is better.

    Unresolved-token count, text length and character content deliberately do
    not belong here. Those are visible to selection independently.
    """

    tones = [run for run in decoded.classified_runs if run.kind is HardRunKind.TONE]
    gaps = [run for run in decoded.classified_runs if run.kind is HardRunKind.GAP]
    dots = [run for run in tones if run.symbol == "."]
    dashes = [run for run in tones if run.symbol == "-"]

    return (
        _tone_ratio_error(dots, dashes) * 120.0
        + _gap_min_error(gaps) * 80.0
        + _unit_cv(dots) * 60.0
    )


def interval_unknown_ratio(original_runs: tuple[TimedRun, ...], start_s: float, end_s: float) -> float:
    duration_s = max(0.0, end_s - start_s)
    if duration_s <= 0:
        return 0.0
    unknown_s = 0.0
    for run in original_runs:
        overlap = max(0.0, min(run.end_s, end_s) - max(run.start_s, start_s))
        if overlap > 0 and run.state is RunState.UNKNOWN:
            unknown_s += overlap
    return round(max(0.0, min(1.0, unknown_s / duration_s)), 6)


def total_duration_s(runs: tuple[TimedRun, ...]) -> float:
    if not runs:
        return 0.0
    return max(run.end_s for run in runs)


def active_bounds(runs: list[HardRun]) -> tuple[float, float]:
    tones = [run for run in runs if run.kind is HardRunKind.TONE]
    if not tones:
        return 0.0, 0.0
    return tones[0].start_s, tones[-1].end_s


def _merge_adjacent_timed_runs(runs: list[TimedRun]) -> list[TimedRun]:
    if not runs:
        return []
    merged: list[TimedRun] = []
    for run in runs:
        if merged and merged[-1].state is run.state:
            previous = merged[-1]
            end_s = max(previous.end_s, run.end_s)
            merged[-1] = TimedRun(previous.state, previous.start_s, round(end_s - previous.start_s, 10))
        else:
            merged.append(run)
    return merged


def _merge_adjacent_hard_runs(runs: list[HardRun]) -> list[HardRun]:
    if not runs:
        return []
    merged: list[HardRun] = []
    for run in runs:
        if merged and merged[-1].kind is run.kind:
            previous = merged[-1]
            end_s = max(previous.end_s, run.end_s)
            merged[-1] = HardRun(previous.kind, previous.start_s, round(end_s - previous.start_s, 10))
        else:
            merged.append(run)
    return merged


def _trim_segment(runs: list[HardRun]) -> list[HardRun]:
    start = 0
    end = len(runs)
    while start < end and runs[start].kind is not HardRunKind.TONE:
        start += 1
    while end > start and runs[end - 1].kind is not HardRunKind.TONE:
        end -= 1
    return runs[start:end]


def _classified_runs_to_decode_tokens(runs: list[ClassifiedRun], *, trim_edges: bool) -> tuple[DecodeToken, ...]:
    tokens: list[DecodeToken] = []
    current = ""
    current_start_s = 0.0
    current_end_s = 0.0
    current_runs: list[ClassifiedRun] = []

    def flush_current() -> None:
        nonlocal current, current_start_s, current_end_s, current_runs
        if not current:
            current_runs = []
            return
        char = CHAR_BY_MORSE.get(current)
        if char is None:
            repaired = _repair_unknown_character_tokens(current_runs)
            if repaired:
                tokens.extend(repaired)
            else:
                tokens.append(unknown_token(start_s=current_start_s, end_s=current_end_s))
        else:
            tokens.append(char_token(char, start_s=current_start_s, end_s=current_end_s))
        current = ""
        current_start_s = 0.0
        current_end_s = 0.0
        current_runs = []

    for run in runs:
        if run.kind is HardRunKind.TONE:
            if not current:
                current_start_s = run.start_s
            current += run.symbol
            current_end_s = run.start_s + run.duration_s
            current_runs.append(run)
            continue
        if run.symbol == "element_gap":
            if current:
                current_runs.append(run)
            continue
        flush_current()
        if run.symbol == "word_gap":
            tokens.append(gap_token("word_gap", start_s=run.start_s, end_s=run.start_s + run.duration_s))
        elif run.symbol == "session_gap":
            tokens.append(gap_token("session_gap", start_s=run.start_s, end_s=run.start_s + run.duration_s))
    flush_current()
    if trim_edges:
        while tokens and tokens[0].is_gap:
            tokens.pop(0)
        while tokens and tokens[-1].is_gap:
            tokens.pop()
    return tuple(tokens)


def _repair_unknown_character_tokens(runs: list[ClassifiedRun]) -> tuple[DecodeToken, ...]:
    """Try one conservative inner split for an invalid Morse character.

    This is deliberately not a general Morse search.  It only runs after the
    primary interpretation already produced an unknown character.  A split is
    accepted only when one internal element gap clearly stands out from the
    other element gaps and both sides become valid Morse characters.
    """
    tone_count = sum(1 for run in runs if run.kind is HardRunKind.TONE)
    if tone_count < 2:
        return ()

    candidates = _unknown_character_split_candidates(runs)
    if not candidates:
        return ()
    best_index, best_gap = max(candidates, key=lambda item: (item[1].units, item[1].duration_s))

    other_gaps = [gap for index, gap in candidates if index != best_index]
    if not _unknown_split_gap_is_significant(best_gap, other_gaps):
        return ()

    left_runs = runs[:best_index]
    right_runs = runs[best_index + 1 :]
    left = _character_from_classified_piece(left_runs)
    right = _character_from_classified_piece(right_runs)
    if left is None or right is None:
        return ()
    left_char, left_start, left_end, left_symbols = left
    right_char, right_start, right_end, right_symbols = right

    # Avoid turning arbitrary noise into two one-element characters.  Splitting
    # T|U or E|K is allowed; E|T / T|E style repairs are too cheap.
    if len(left_symbols) == 1 and len(right_symbols) == 1:
        return ()

    return (
        char_token(left_char, start_s=left_start, end_s=left_end),
        char_token(right_char, start_s=right_start, end_s=right_end),
    )


def _unknown_character_split_candidates(runs: list[ClassifiedRun]) -> list[tuple[int, ClassifiedRun]]:
    candidates: list[tuple[int, ClassifiedRun]] = []
    for index, run in enumerate(runs):
        if run.kind is not HardRunKind.GAP or run.symbol != "element_gap":
            continue
        if not _contains_tone(runs[:index]) or not _contains_tone(runs[index + 1 :]):
            continue
        left = _character_from_classified_piece(runs[:index])
        right = _character_from_classified_piece(runs[index + 1 :])
        if left is not None and right is not None:
            candidates.append((index, run))
    return candidates


def _unknown_split_gap_is_significant(gap: ClassifiedRun, other_gaps: list[ClassifiedRun]) -> bool:
    if gap.units <= 0:
        return False
    if not other_gaps:
        # With a single possible split there is no local contrast.  Keep the
        # unknown instead of guessing.
        return False
    reference = max(other.units for other in other_gaps if other.units > 0)
    if reference <= 0:
        return False
    min_ratio = 1.25
    min_delta_units = 0.35
    return gap.units / reference >= min_ratio and gap.units - reference >= min_delta_units


def _character_from_classified_piece(runs: list[ClassifiedRun]) -> tuple[str, float, float, str] | None:
    tones = [run for run in runs if run.kind is HardRunKind.TONE]
    if not tones:
        return None
    symbols = "".join(run.symbol for run in tones)
    char = CHAR_BY_MORSE.get(symbols)
    if char is None:
        return None
    return char, tones[0].start_s, tones[-1].start_s + tones[-1].duration_s, symbols


def _contains_tone(runs: list[ClassifiedRun]) -> bool:
    return any(run.kind is HardRunKind.TONE for run in runs)


def _decode_tokens_to_legacy_morse_tokens(tokens: tuple[DecodeToken, ...]) -> tuple[str, ...]:
    output: list[str] = []
    for token in tokens:
        if token.kind == "word_gap":
            output.append("/")
        elif token.kind == "session_gap":
            output.append("///")
        elif token.kind == "unknown":
            output.append("?")
        elif token.kind == "char":
            # Legacy field cannot reconstruct the exact Morse pattern from the
            # decoded character without importing the encoder.  It remains only
            # for old tests/debug helpers; commit and JSON now use decode_tokens.
            output.append(token.value)
    return tuple(output)


def _classified_runs_to_tokens(runs: list[ClassifiedRun]) -> tuple[str, ...]:
    tokens: list[str] = []
    current = ""
    for run in runs:
        if run.kind is HardRunKind.TONE:
            current += run.symbol
            continue
        if run.symbol == "element_gap":
            continue
        if current:
            tokens.append(current)
            current = ""
        if run.symbol == "word_gap":
            tokens.append("/")
        elif run.symbol == "session_gap":
            tokens.append("///")
    if current:
        tokens.append(current)
    return tuple(_trim_word_separators(tokens))


def _trim_word_separators(tokens: list[str]) -> list[str]:
    while tokens and tokens[0] in {"/", "///"}:
        tokens.pop(0)
    while tokens and tokens[-1] in {"/", "///"}:
        tokens.pop()
    return tokens


def _adaptive_letter_word_boundary_units(runs: list[HardRun], unit_s: float, config) -> float:
    if unit_s <= 0:
        return config.default_word_gap_units

    element_letter_units = (
        _adaptive_element_letter_boundary_units(runs, unit_s, config)
        if getattr(config, "adaptive_element_letter_gap", False)
        else config.element_letter_gap_units
    )
    gaps = sorted(
        run.duration_s / unit_s
        for run in runs
        if run.kind is HardRunKind.GAP and run.duration_s > 0 and run.duration_s / unit_s >= element_letter_units
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


def _best_letter_word_gap_split(gaps: list[float], config, *, min_upper_count: int) -> int | None:
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
        if delta > best_delta or (delta == best_delta and ratio > best_ratio):
            best_index = index
            best_ratio = ratio
            best_delta = delta
    return best_index


def _tone_ratio_error(dots: list[ClassifiedRun], dashes: list[ClassifiedRun]) -> float:
    if not dots or not dashes:
        return 0.0
    dot_mean = sum(run.duration_s for run in dots) / len(dots)
    dash_mean = sum(run.duration_s for run in dashes) / len(dashes)
    if dot_mean <= 0:
        return 10.0
    return _soft_ratio_error(dash_mean / dot_mean, target=3.0, tolerance=0.75)


def _gap_min_error(gaps: list[ClassifiedRun]) -> float:
    error = 0.0
    for gap in gaps:
        if gap.symbol == "element_gap":
            error += _below_min_error(gap.units, minimum=0.75)
        elif gap.symbol == "letter_gap":
            error += _below_min_error(gap.units, minimum=2.5)
        elif gap.symbol == "word_gap":
            error += _below_min_error(gap.units, minimum=5.5)
        elif gap.symbol == "session_gap":
            error += _below_min_error(gap.units, minimum=10.0)
    return error / max(len(gaps), 1)


def _unit_cv(dots: list[ClassifiedRun]) -> float:
    if len(dots) < 2:
        return 0.0
    durations = [run.duration_s for run in dots]
    mean = sum(durations) / len(durations)
    if mean <= 0:
        return 10.0
    variance = sum((duration - mean) ** 2 for duration in durations) / len(durations)
    return variance**0.5 / mean


def _below_min_error(value: float, minimum: float) -> float:
    if value >= minimum:
        return 0.0
    return (minimum - value) / minimum


def _soft_ratio_error(value: float, target: float, tolerance: float) -> float:
    distance = abs(value - target)
    if distance <= tolerance:
        return 0.0
    return (distance - tolerance) / target


__all__ = [
    "ClassifiedRun",
    "HardRun",
    "HardRunKind",
    "RunState",
    "TimedDecode",
    "TimedRun",
    "active_bounds",
    "decode_segment_with_unit",
    "decode_segments_with_unit",
    "estimate_unit_s",
    "expand_unknown_runs",
    "timed_runs_from_signal_track",
    "timing_quality_score",
    "total_duration_s",
    "unit_candidates",
]
