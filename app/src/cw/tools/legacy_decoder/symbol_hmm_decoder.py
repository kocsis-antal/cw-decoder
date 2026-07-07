from __future__ import annotations

import numpy as np

from cw.tools.legacy_decoder.base import ClassifiedRun, DecodeResult, DetectedRun, _runs_from_activity
from cw.morse_table import decode_tokens
from cw.tools.legacy_decoder.config import DecoderConfig
from cw.tools.legacy_decoder.quality import score_decode_result
from cw.tools.legacy_decoder.stream_decode import _estimate_unit_from_runs, _unit_candidates
from cw.tools.legacy_decoder.models import DecodeCandidate, SignalRun, _SymbolHmmState
from cw.tools.legacy_decoder.signal_analysis import _activity_probability, _candidate_evidence_score, _mean_run_confidence, _signal_runs
from cw.tools.legacy_decoder.soft_decoder import _viterbi_activity
from cw.tools.legacy_decoder.character_hmm_decoder import _decode_character_hmm_range
from cw.tools.legacy_decoder.hmm_common import (
    _advance_over_idle_frames,
    _cost_prefix,
    _duration_options,
    _remaining_tone_probability,
    _segment_mean_cost,
    _symbol_hmm_candidate_is_plausible,
)
from cw.morse_table import CHAR_BY_MORSE
_VALID_MORSE_TOKENS = frozenset(CHAR_BY_MORSE.keys())
_VALID_MORSE_PREFIXES = frozenset(
    code[:index]
    for code in _VALID_MORSE_TOKENS
    for index in range(1, len(code) + 1)
)

def _decode_symbol_hmm_energy_candidates(
    energy: np.ndarray,
    frame_times: np.ndarray,
    *,
    carrier_hz: float,
    start_s: float,
    session_gap_s: float,
    config: DecoderConfig,
    include_character_templates: bool = True,
) -> list[DecodeCandidate]:
    """Decode a carrier directly from activity probabilities with a duration model.

    This is intentionally below the run/lattice layer.  It does not receive a
    pre-cut tone/gap run list.  Instead it searches the probability frames for
    a sequence of Morse tone durations (dit/dah) and gap durations
    (element/letter/word) with a small beam.  The only assumptions are the CW
    duration ratios and the signal-vs-silence probabilities extracted from the
    carrier envelope.
    """

    if len(energy) == 0 or len(frame_times) == 0:
        return []
    noise_floor = float(np.percentile(energy, 15))
    signal_floor = float(np.percentile(energy, 95))
    if signal_floor <= noise_floor:
        return []
    probabilities = _activity_probability(energy, noise_floor, signal_floor)
    if float(np.max(probabilities)) < 0.20:
        return []

    active_hint = _viterbi_activity(
        probabilities,
        transition_penalty=config.viterbi_transition_penalty,
    )
    raw_runs = _runs_from_activity(active_hint, config.hop_ms / 1000)
    hint_runs = [DetectedRun(run.kind, run.start_s + start_s, run.duration_s) for run in raw_runs]
    try:
        initial_unit_s = _estimate_unit_from_runs(hint_runs)
    except ValueError:
        initial_unit_s = _estimate_unit_from_probability_autocorrelation(probabilities, config.hop_ms / 1000)
    if initial_unit_s is None or initial_unit_s <= 0:
        return []

    base_unit_candidates = _filter_symbol_hmm_unit_candidates(
        _unit_candidates(
            initial_unit_s,
            config.symbol_hmm_unit_spread,
            config.symbol_hmm_unit_steps,
        ),
        config,
    )
    if not base_unit_candidates:
        return []
    threshold = noise_floor + (signal_floor - noise_floor) * 0.5
    ranges = _probability_session_ranges(
        probabilities,
        frame_times,
        start_s,
        session_gap_s=session_gap_s,
        hop_s=config.hop_ms / 1000,
    )
    decoded_candidates: list[DecodeCandidate] = []
    for range_start, range_end in ranges:
        range_units = _symbol_hmm_range_unit_candidates(
            probabilities,
            frame_times,
            range_start,
            range_end,
            base_unit_candidates,
            config,
        )
        first_pass: list[DecodeCandidate] = []
        for unit_s in range_units:
            first_pass.extend(
                _decode_symbol_hmm_range(
                    probabilities,
                    frame_times,
                    range_start,
                    range_end,
                    carrier_hz=carrier_hz,
                    start_s=start_s,
                    unit_s=unit_s,
                    threshold=threshold,
                    noise_floor=noise_floor,
                    signal_floor=signal_floor,
                    config=config,
                )
            )
        decoded_candidates.extend(first_pass)
        for refined_unit_s in _symbol_hmm_refined_unit_candidates(first_pass, range_units, config):
            decoded_candidates.extend(
                _decode_symbol_hmm_range(
                    probabilities,
                    frame_times,
                    range_start,
                    range_end,
                    carrier_hz=carrier_hz,
                    start_s=start_s,
                    unit_s=refined_unit_s,
                    threshold=threshold,
                    noise_floor=noise_floor,
                    signal_floor=signal_floor,
                    config=config,
                )
            )
        if include_character_templates:
            for unit_s in range_units:
                decoded_candidates.extend(
                    _decode_character_hmm_range(
                        probabilities,
                        frame_times,
                        range_start,
                        range_end,
                        carrier_hz=carrier_hz,
                        start_s=start_s,
                        unit_s=unit_s,
                        threshold=threshold,
                        noise_floor=noise_floor,
                        signal_floor=signal_floor,
                        config=config,
                    )
                )
    return decoded_candidates

def _estimate_unit_from_probability_autocorrelation(probabilities: np.ndarray, hop_s: float) -> float | None:
    if len(probabilities) < 8 or hop_s <= 0:
        return None
    p = probabilities.astype(np.float64, copy=False)
    centered = p - float(np.mean(p))
    if float(np.max(np.abs(centered))) <= 1e-6:
        return None
    # Search a realistic 8-40 WPM unit range.  This is only a fallback; normal
    # operation gets the unit from the activity-HMM hint path.
    min_lag = max(1, int(round(0.03 / hop_s)))
    max_lag = min(len(centered) // 3, int(round(0.15 / hop_s)))
    if max_lag <= min_lag:
        return None
    best_lag = min_lag
    best_score = -1e9
    for lag in range(min_lag, max_lag + 1):
        a = centered[:-lag]
        b = centered[lag:]
        score = float(np.dot(a, b)) / max(1, len(a))
        # Prefer fundamental-ish shorter lags a little; harmonics at 2u/3u are
        # common in Morse envelopes.
        score -= lag * 1e-5
        if score > best_score:
            best_lag = lag
            best_score = score
    return best_lag * hop_s

def _probability_session_ranges(
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    start_s: float,
    *,
    session_gap_s: float,
    hop_s: float,
) -> list[tuple[int, int]]:
    if len(probabilities) == 0:
        return []
    active = probabilities >= 0.22
    if not np.any(active):
        return []
    max_gap_frames = max(1, int(round(max(session_gap_s, hop_s) / max(hop_s, 1e-6))))
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    last_active: int | None = None
    for index, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = index
            last_active = index
        elif start is not None and last_active is not None and index - last_active >= max_gap_frames:
            ranges.append(_expand_frame_range(start, last_active + 1, len(probabilities), margin=4))
            start = None
            last_active = None
    if start is not None and last_active is not None:
        ranges.append(_expand_frame_range(start, last_active + 1, len(probabilities), margin=4))
    return ranges

def _expand_frame_range(start: int, end: int, size: int, *, margin: int) -> tuple[int, int]:
    return max(0, start - margin), min(size, end + margin)

def _symbol_hmm_range_unit_candidates(
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    range_start: int,
    range_end: int,
    base_units: tuple[float, ...],
    config: DecoderConfig,
) -> tuple[float, ...]:
    """Return unit hypotheses for one probability session range.

    The old direct HMM used one unit estimate for the whole rolling window.  A
    streaming channel can contain several operators/turns with slightly different
    speed, and a long retained window can bias the estimate toward the wrong
    part of the signal.  The HMM now starts from the global estimate but adds a
    local estimate derived from the probability range itself.
    """

    units: list[float] = list(base_units)
    if range_end <= range_start:
        return _filter_symbol_hmm_unit_candidates(_unique_unit_candidates(units), config)
    hop_s = config.hop_ms / 1000
    local_probs = probabilities[range_start:range_end]
    local_frame_times = frame_times[range_start:range_end] if len(frame_times) else np.asarray([], dtype=np.float32)
    local_unit = _estimate_unit_from_probability_autocorrelation(local_probs, hop_s)
    if local_unit is not None:
        units.extend(_unit_candidates(local_unit, config.symbol_hmm_unit_spread, config.symbol_hmm_unit_steps))

    # The Viterbi activity hint is still only a hint, not the decoder input.  It
    # is useful for a second local unit estimate because tone/gap durations carry
    # more timing information than an autocorrelation peak alone.
    if len(local_probs) and len(local_frame_times):
        active_hint = _viterbi_activity(local_probs, transition_penalty=config.viterbi_transition_penalty)
        raw_runs = _runs_from_activity(active_hint, hop_s)
        absolute_offset_s = float(frame_times[range_start]) if len(frame_times) and range_start < len(frame_times) else 0.0
        hint_runs = [DetectedRun(run.kind, run.start_s + absolute_offset_s, run.duration_s) for run in raw_runs]
        try:
            hint_unit = _estimate_unit_from_runs(hint_runs)
        except ValueError:
            hint_unit = None
        if hint_unit is not None and hint_unit > 0:
            units.extend(_unit_candidates(hint_unit, config.symbol_hmm_unit_spread, config.symbol_hmm_unit_steps))
    return _filter_symbol_hmm_unit_candidates(_unique_unit_candidates(units), config)

def _symbol_hmm_refined_unit_candidates(
    first_pass: list[DecodeCandidate],
    existing_units: tuple[float, ...],
    config: DecoderConfig,
) -> tuple[float, ...]:
    """Derive second-pass unit hypotheses from the HMM's own best paths.

    This is the structural step that moves unit estimation inside the symbol
    model: after the first duration-HMM pass, the classified dit/dah/gap path can
    tell us the operator's actual unit better than the initial activity hint.
    """

    if not first_pass:
        return ()
    ranked = sorted(first_pass, key=lambda c: (-c.evidence_score, c.quality_score or 1e9))[: max(1, config.symbol_hmm_max_candidates)]
    refined: list[float] = []
    for candidate in ranked:
        unit_s = _estimate_unit_from_symbol_runs(candidate.runs)
        if unit_s is None:
            continue
        # Do not let a bad first-pass text yank the unit estimate into an
        # unrelated speed regime.  A 45% window is intentionally wider than the
        # normal unit spread; it allows hand-keyed drift but rejects harmonics.
        if existing_units:
            nearest = min(existing_units, key=lambda value: abs(value - unit_s))
            if nearest > 0 and abs(unit_s - nearest) / nearest > 0.45:
                continue
        refined.extend(_unit_candidates(unit_s, min(0.10, config.symbol_hmm_unit_spread), 3))
    return tuple(
        unit
        for unit in _filter_symbol_hmm_unit_candidates(_unique_unit_candidates(refined), config)
        if not _unit_already_present(unit, existing_units)
    )

def _filter_symbol_hmm_unit_candidates(units: tuple[float, ...], config: DecoderConfig) -> tuple[float, ...]:
    return tuple(
        unit
        for unit in units
        if config.symbol_hmm_min_unit_s <= unit <= config.symbol_hmm_max_unit_s
    )

def _estimate_unit_from_symbol_runs(runs: tuple[SignalRun, ...]) -> float | None:
    estimates: list[tuple[float, float]] = []
    for run in runs:
        target_units = _symbol_run_target_units(run)
        if target_units is None or target_units <= 0:
            continue
        unit_s = run.duration_s / target_units
        if not 0.025 <= unit_s <= 0.250:
            continue
        weight = max(0.05, min(1.0, run.confidence))
        if run.kind == "tone":
            weight *= 1.6
        elif run.symbol == "word_gap":
            # Word gaps are useful but operator-dependent; they should not
            # dominate the dit-time estimate.
            weight *= 0.35
        estimates.append((unit_s, weight))
    if len(estimates) < 3:
        return None
    return _weighted_median(estimates)

def _symbol_run_target_units(run: SignalRun) -> float | None:
    if run.kind == "tone":
        if run.symbol == ".":
            return 1.0
        if run.symbol == "-":
            return 3.0
        return None
    if run.symbol == "element_gap":
        return 1.0
    if run.symbol == "letter_gap":
        return 3.0
    if run.symbol == "word_gap":
        return 7.0
    return None

def _weighted_median(values: list[tuple[float, float]]) -> float:
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(max(0.0, weight) for _, weight in ordered)
    if total <= 0:
        return float(ordered[len(ordered) // 2][0])
    running = 0.0
    for value, weight in ordered:
        running += max(0.0, weight)
        if running >= total / 2:
            return float(value)
    return float(ordered[-1][0])

def _unique_unit_candidates(units: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    output: list[float] = []
    for unit in sorted(float(value) for value in units if value and value > 0):
        if _unit_already_present(unit, tuple(output)):
            continue
        output.append(unit)
    return tuple(output)

def _unit_already_present(unit: float, units: tuple[float, ...]) -> bool:
    return any(abs(unit - existing) <= max(0.0015, existing * 0.025) for existing in units)

def _decode_symbol_hmm_range(
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    range_start: int,
    range_end: int,
    *,
    carrier_hz: float,
    start_s: float,
    unit_s: float,
    threshold: float,
    noise_floor: float,
    signal_floor: float,
    config: DecoderConfig,
) -> list[DecodeCandidate]:
    hop_s = config.hop_ms / 1000
    if unit_s <= 0 or hop_s <= 0 or range_end <= range_start:
        return []
    unit_frames = max(1.0, unit_s / hop_s)
    # Trim extremely low-probability padding.  This trims only leading/trailing
    # idle frames; the decoder still chooses all internal tone/gap durations.
    while range_start < range_end and probabilities[range_start] < 0.16:
        range_start += 1
    while range_end > range_start and probabilities[range_end - 1] < 0.16:
        range_end -= 1
    if range_end <= range_start:
        return []

    tone_cost_prefix = _cost_prefix(-np.log(np.clip(probabilities, 1e-5, 1.0)))
    gap_cost_prefix = _cost_prefix(-np.log(np.clip(1.0 - probabilities, 1e-5, 1.0)))
    states = [
        _SymbolHmmState(
            position=range_start,
            tokens=(),
            current_token="",
            classified_runs=(),
            cost=0.0,
        )
    ]
    finished: list[_SymbolHmmState] = []
    max_steps = max(12, min(180, int((range_end - range_start) / max(1.0, unit_frames * 1.2)) + 20))
    for _ in range(max_steps):
        next_states: list[_SymbolHmmState] = []
        for state in states:
            position = _advance_over_idle_frames(state.position, range_end, probabilities, max_skip_frames=int(round(unit_frames * 1.8)))
            if position >= range_end:
                finished.append(_finalize_symbol_hmm_state(state))
                continue
            if _remaining_tone_probability(probabilities, position, range_end) < 0.20:
                finished.append(_finalize_symbol_hmm_state(state))
                continue
            for tone_symbol, tone_frames, tone_penalty in _symbol_hmm_tone_options(unit_frames):
                tone_end = position + tone_frames
                if tone_end > range_end:
                    continue
                tone_cost = _segment_mean_cost(tone_cost_prefix, position, tone_end) + tone_penalty
                run_start_s = start_s + float(frame_times[position])
                run_duration_s = max(hop_s, tone_frames * hop_s)
                tone_run = ClassifiedRun(
                    kind="tone",
                    start_s=run_start_s,
                    duration_s=run_duration_s,
                    symbol=tone_symbol,
                    units=round(run_duration_s / unit_s, 3),
                )
                token_after_tone = state.current_token + tone_symbol
                if not _is_valid_morse_prefix(token_after_tone):
                    continue
                token_overflow_penalty = max(0, len(token_after_tone) - 6) * 3.0
                base_state = _SymbolHmmState(
                    position=tone_end,
                    tokens=state.tokens,
                    current_token=token_after_tone,
                    classified_runs=(*state.classified_runs, tone_run),
                    cost=state.cost + tone_cost + token_overflow_penalty,
                )
                if tone_end >= range_end - max(1, int(round(unit_frames * 1.5))):
                    finished.append(_finalize_symbol_hmm_state(base_state))
                for gap_symbol, gap_frames, gap_penalty in _symbol_hmm_gap_options(
                    unit_frames,
                    allow_intra=len(token_after_tone) < 6,
                ):
                    gap_end = tone_end + gap_frames
                    if gap_end > range_end:
                        continue
                    gap_cost = _segment_mean_cost(gap_cost_prefix, tone_end, gap_end) + gap_penalty
                    gap_start_s = start_s + float(frame_times[tone_end]) if tone_end < len(frame_times) else run_start_s + run_duration_s
                    gap_run = ClassifiedRun(
                        kind="gap",
                        start_s=gap_start_s,
                        duration_s=max(hop_s, gap_frames * hop_s),
                        symbol=gap_symbol,
                        units=round(max(hop_s, gap_frames * hop_s) / unit_s, 3),
                    )
                    next_states.append(
                        _advance_symbol_hmm_gap(
                            base_state,
                            gap_run,
                            gap_end,
                            gap_cost + config.symbol_hmm_transition_penalty,
                        )
                    )
        finished.extend(
            _finalize_symbol_hmm_state(state)
            for state in next_states
            if state.position >= range_end - max(1, int(round(unit_frames * 1.2)))
        )
        states = _prune_symbol_hmm_states(
            [state for state in next_states if state.position < range_end],
            beam_width=config.symbol_hmm_beam_width,
        )
        if not states:
            break
    finished.extend(_finalize_symbol_hmm_state(state) for state in states)
    ranked = _rank_symbol_hmm_final_states(finished, max_candidates=config.symbol_hmm_max_candidates)
    output: list[DecodeCandidate] = []
    for state in ranked:
        tokens = list(state.tokens)
        if not tokens:
            continue
        text = decode_tokens(tokens)
        if not text:
            continue
        decoded = DecodeResult(
            text=text,
            tokens=tokens,
            runs=[],
            classified_runs=list(state.classified_runs),
            carrier_hz=carrier_hz,
            unit_s=unit_s,
            threshold=threshold,
        )
        quality = score_decode_result(decoded)
        confidence_runs = _signal_runs(decoded, probabilities, frame_times, start_s, hop_s)
        confidence = _mean_run_confidence(confidence_runs)
        normalized_cost = state.cost / max(1, len(state.classified_runs))
        token_count = len([token for token in decoded.tokens if token != "/"])
        et_only_count = sum(1 for char in decoded.text if char in "ET")
        known_count = sum(1 for char in decoded.text if char and not char.isspace() and char != "?")
        et_density = et_only_count / max(1, known_count)
        dense_token_penalty = token_count * 1.75 + max(0.0, et_density - 0.65) * token_count * 2.0
        adjusted_quality = quality.score + normalized_cost * 3.0 + dense_token_penalty
        evidence_score = (
            _candidate_evidence_score(decoded, adjusted_quality, confidence)
            - normalized_cost * 1.5
            - dense_token_penalty
        )
        tone_runs = [run for run in state.classified_runs if run.kind == "tone"]
        if not tone_runs:
            continue
        if not _symbol_hmm_candidate_is_plausible(
            decoded,
            unit_s=unit_s,
            confidence=confidence,
            quality_score=adjusted_quality,
            detector="symbol-hmm",
        ):
            continue
        output.append(
            DecodeCandidate(
                carrier_hz=round(float(carrier_hz), 3),
                detector="symbol-hmm",
                threshold_ratio=0.0,
                threshold=float(threshold),
                noise_floor=noise_floor,
                signal_floor=signal_floor,
                duty_cycle=round(float(np.mean(probabilities[range_start:range_end] >= 0.5)), 6),
                unit_s=round(float(unit_s), 6),
                wpm=round(float(1.2 / unit_s), 3) if unit_s > 0 else None,
                text=text,
                tokens=tuple(tokens),
                quality_score=round(float(adjusted_quality), 6),
                confidence=round(float(confidence), 6),
                evidence_score=round(float(evidence_score), 6),
                start_s=round(float(min(run.start_s for run in tone_runs)), 6),
                end_s=round(float(max(run.start_s + run.duration_s for run in tone_runs)), 6),
                runs=tuple(confidence_runs),
            )
        )
    return output

def _symbol_hmm_tone_options(unit_frames: float) -> tuple[tuple[str, int, float], ...]:
    options: list[tuple[str, int, float]] = []
    for frames, penalty in _duration_options(unit_frames, 1.0, relative_width=0.48):
        options.append((".", frames, penalty + 0.02))
    for frames, penalty in _duration_options(unit_frames, 3.0, relative_width=0.38):
        options.append(("-", frames, penalty + 0.04))
    return tuple(sorted(options, key=lambda item: item[2])[:5])

def _symbol_hmm_gap_options(unit_frames: float, *, allow_intra: bool) -> tuple[tuple[str, int, float], ...]:
    options: list[tuple[str, int, float]] = []
    if allow_intra:
        for frames, penalty in _duration_options(unit_frames, 1.0, relative_width=0.55, max_options=4):
            options.append(("element_gap", frames, penalty + 0.02))
    for frames, penalty in _duration_options(unit_frames, 3.0, relative_width=0.48, max_options=4):
        options.append(("letter_gap", frames, penalty + 0.04))
    for frames, penalty in _duration_options(unit_frames, 7.0, relative_width=0.45, max_options=3):
        options.append(("word_gap", frames, penalty + 0.10))
    return tuple(sorted(options, key=lambda item: item[2])[:6])

def _advance_symbol_hmm_gap(
    state: _SymbolHmmState,
    gap_run: ClassifiedRun,
    position: int,
    extra_cost: float,
) -> _SymbolHmmState:
    tokens = list(state.tokens)
    current = state.current_token
    close_penalty = 0.0
    if gap_run.symbol == "letter_gap":
        if current:
            close_penalty += _symbol_hmm_token_close_penalty(current)
            tokens.append(current)
            current = ""
    elif gap_run.symbol == "word_gap":
        if current:
            close_penalty += _symbol_hmm_token_close_penalty(current)
            tokens.append(current)
            current = ""
        if tokens and tokens[-1] != "/":
            tokens.append("/")
    return _SymbolHmmState(
        position=position,
        tokens=tuple(tokens),
        current_token=current,
        classified_runs=(*state.classified_runs, gap_run),
        cost=state.cost + extra_cost + close_penalty,
    )

def _finalize_symbol_hmm_state(state: _SymbolHmmState) -> _SymbolHmmState:
    if not state.current_token:
        return state
    return _SymbolHmmState(
        position=state.position,
        tokens=(*state.tokens, state.current_token),
        current_token="",
        classified_runs=state.classified_runs,
        cost=state.cost + _symbol_hmm_token_close_penalty(state.current_token),
    )

def _is_valid_morse_prefix(token: str) -> bool:
    return token in _VALID_MORSE_PREFIXES

def _symbol_hmm_token_close_penalty(token: str) -> float:
    if token not in _VALID_MORSE_TOKENS:
        return 8.0
    # A pure duration model otherwise tends to explain uncertain stretches as a
    # long series of one-element E/T characters.  This is not a QSO/content
    # prior; it is a generic anti-degeneracy prior for the Morse grammar.
    if len(token) == 1:
        return 0.42
    return 0.0

def _prune_symbol_hmm_states(states: list[_SymbolHmmState], *, beam_width: int) -> list[_SymbolHmmState]:
    best_by_key: dict[tuple[int, tuple[str, ...], str], _SymbolHmmState] = {}
    for state in states:
        # Quantize position slightly so equivalent timing paths can compete.
        key = (state.position // 2, state.tokens[-4:], state.current_token)
        existing = best_by_key.get(key)
        if existing is None or _symbol_hmm_state_sort_key(state) < _symbol_hmm_state_sort_key(existing):
            best_by_key[key] = state
    return sorted(best_by_key.values(), key=_symbol_hmm_state_sort_key)[:beam_width]

def _rank_symbol_hmm_final_states(states: list[_SymbolHmmState], *, max_candidates: int) -> list[_SymbolHmmState]:
    best_by_text: dict[str, _SymbolHmmState] = {}
    for state in states:
        finalized = _finalize_symbol_hmm_state(state)
        if not finalized.tokens:
            continue
        text = decode_tokens(list(finalized.tokens))
        if not text:
            continue
        existing = best_by_text.get(text)
        if existing is None or _symbol_hmm_state_sort_key(finalized) < _symbol_hmm_state_sort_key(existing):
            best_by_text[text] = finalized
    return sorted(best_by_text.values(), key=_symbol_hmm_state_sort_key)[: max(1, max_candidates)]

def _symbol_hmm_state_sort_key(state: _SymbolHmmState) -> tuple[float, int, int, int]:
    tokens = (*state.tokens, state.current_token) if state.current_token else state.tokens
    token_list = list(tokens)
    text = decode_tokens(token_list) if token_list else ""
    unknowns = text.count("?")
    known = sum(1 for char in text if not char.isspace() and char != "?")
    punctuation = sum(1 for char in text if char and not char.isspace() and not char.isalnum() and char != "?")
    avg_cost = state.cost / max(1, len(state.classified_runs))
    long_token_penalty = sum(max(0, len(token) - 6) for token in token_list if token != "/") * 4
    return (avg_cost + unknowns * 2.6 + punctuation * 0.7 + long_token_penalty - known * 0.05, unknowns, punctuation, -known)
