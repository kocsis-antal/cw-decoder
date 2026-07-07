from __future__ import annotations

import numpy as np
from cw.tools.legacy_decoder.base import ClassifiedRun, DecodeResult, DetectedRun
from cw.morse_table import decode_tokens
from cw.tools.legacy_decoder.quality import score_decode_result
from cw.tools.legacy_decoder.stream_decode import _adaptive_letter_word_boundary_units
from cw.tools.legacy_decoder.config import DecoderConfig
from cw.tools.legacy_decoder.models import DecodeCandidate, _LatticeState, _SymbolOption
from cw.tools.legacy_decoder.signal_analysis import _candidate_evidence_score, _mean_run_confidence, _signal_runs

def _decode_lattice_candidates(
    runs: list[DetectedRun],
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    *,
    carrier_hz: float,
    start_s: float,
    threshold_ratio: float,
    detector: str,
    threshold: float,
    noise_floor: float,
    signal_floor: float,
    duty_cycle: float,
    unit_s: float,
    config: DecoderConfig,
) -> list[DecodeCandidate]:
    """Decode a run segment with a small Morse timing lattice.

    The threshold and Viterbi activity gates still decide where plausible runs are,
    but this layer no longer forces every run to a single symbol immediately.
    Tone durations close to the dot/dash boundary and gap durations close to
    class boundaries branch into neighbouring interpretations.  A compact beam
    keeps the best timed paths alive until the full token sequence is known.
    """

    if unit_s <= 0 or not runs:
        return []
    states = [_LatticeState(tokens=(), current_token="", classified_runs=(), penalty=0.0)]
    letter_word_boundary = (
        _adaptive_letter_word_boundary_units(runs, unit_s, config)
        if config.adaptive_gap_thresholds
        else 5.0
    )
    for run in runs:
        options = _lattice_symbol_options(run, unit_s, letter_word_boundary, config)
        next_states: list[_LatticeState] = []
        for state in states:
            for option in options:
                next_states.append(_advance_lattice_state(state, run, option, unit_s))
        states = _prune_lattice_states(next_states, beam_width=config.lattice_beam_width)
        if not states:
            return []

    final_states = [_finalize_lattice_state(state) for state in states]
    best_by_text: dict[str, _LatticeState] = {}
    for state in final_states:
        if not state.tokens:
            continue
        text = decode_tokens(list(state.tokens))
        if not text:
            continue
        existing = best_by_text.get(text)
        if existing is None or _lattice_state_sort_key(state) < _lattice_state_sort_key(existing):
            best_by_text[text] = state
    ranked = sorted(best_by_text.values(), key=_lattice_state_sort_key)[: config.lattice_max_candidates]
    output: list[DecodeCandidate] = []
    for state in ranked:
        decoded = DecodeResult(
            text=decode_tokens(list(state.tokens)),
            tokens=list(state.tokens),
            runs=runs,
            classified_runs=list(state.classified_runs),
            carrier_hz=carrier_hz,
            unit_s=unit_s,
            threshold=threshold,
        )
        quality = score_decode_result(decoded)
        confidence_runs = _signal_runs(decoded, probabilities, frame_times, start_s, config.hop_ms / 1000)
        confidence = _mean_run_confidence(confidence_runs)
        adjusted_quality = quality.score + state.penalty * 4.0
        evidence_score = _candidate_evidence_score(decoded, adjusted_quality, confidence) - state.penalty * 1.8
        segment_start = min((run.start_s for run in runs if run.kind == "tone"), default=runs[0].start_s)
        segment_end = max(
            (run.start_s + run.duration_s for run in runs if run.kind == "tone"),
            default=runs[-1].start_s + runs[-1].duration_s,
        )
        output.append(
            DecodeCandidate(
                carrier_hz=round(float(carrier_hz), 3),
                detector=detector,
                threshold_ratio=round(float(threshold_ratio), 6),
                threshold=float(threshold),
                noise_floor=noise_floor,
                signal_floor=signal_floor,
                duty_cycle=duty_cycle,
                unit_s=round(float(unit_s), 6),
                wpm=round(float(1.2 / unit_s), 3) if unit_s > 0 else None,
                text=decoded.text,
                tokens=tuple(decoded.tokens),
                quality_score=round(float(adjusted_quality), 6),
                confidence=round(float(confidence), 6),
                evidence_score=round(float(evidence_score), 6),
                start_s=round(float(segment_start), 6),
                end_s=round(float(segment_end), 6),
                runs=tuple(confidence_runs),
            )
        )
    return output

def _lattice_symbol_options(
    run: DetectedRun,
    unit_s: float,
    letter_word_boundary: float,
    config: DecoderConfig,
) -> tuple[_SymbolOption, ...]:
    units = run.duration_s / unit_s
    if run.kind == "tone":
        options = [_SymbolOption("." if units < 2.0 else "-", _tone_symbol_penalty(units, "." if units < 2.0 else "-"))]
        alternate = "-" if options[0].symbol == "." else "."
        if abs(units - 2.0) <= config.lattice_tone_margin_units:
            options.append(_SymbolOption(alternate, _tone_symbol_penalty(units, alternate) + 0.35))
        return tuple(sorted(options, key=lambda option: option.penalty))

    element_letter_boundary = config.element_letter_gap_units if config.adaptive_gap_thresholds else 2.0
    if units < element_letter_boundary:
        base_symbol = "element_gap"
    elif units < letter_word_boundary:
        base_symbol = "letter_gap"
    else:
        base_symbol = "word_gap"

    options_by_symbol = {base_symbol: _gap_symbol_penalty(units, base_symbol, config)}
    if abs(units - element_letter_boundary) <= config.lattice_gap_margin_units:
        options_by_symbol["element_gap"] = min(
            options_by_symbol.get("element_gap", 999.0),
            _gap_symbol_penalty(units, "element_gap", config) + 0.25,
        )
        options_by_symbol["letter_gap"] = min(
            options_by_symbol.get("letter_gap", 999.0),
            _gap_symbol_penalty(units, "letter_gap", config) + 0.25,
        )
    if abs(units - letter_word_boundary) <= config.lattice_gap_margin_units:
        options_by_symbol["letter_gap"] = min(
            options_by_symbol.get("letter_gap", 999.0),
            _gap_symbol_penalty(units, "letter_gap", config) + 0.25,
        )
        options_by_symbol["word_gap"] = min(
            options_by_symbol.get("word_gap", 999.0),
            _gap_symbol_penalty(units, "word_gap", config) + 0.25,
        )
    return tuple(
        _SymbolOption(symbol, penalty)
        for symbol, penalty in sorted(options_by_symbol.items(), key=lambda item: item[1])
    )

def _tone_symbol_penalty(units: float, symbol: str) -> float:
    target = 1.0 if symbol == "." else 3.0
    return abs(units - target) / target

def _gap_symbol_penalty(units: float, symbol: str, config: DecoderConfig) -> float:
    if symbol == "element_gap":
        target = 1.0
    elif symbol == "letter_gap":
        target = max(3.0, config.element_letter_gap_units + 0.6)
    else:
        target = max(config.default_word_gap_units, 5.5)
    return abs(units - target) / target

def _advance_lattice_state(
    state: _LatticeState,
    run: DetectedRun,
    option: _SymbolOption,
    unit_s: float,
) -> _LatticeState:
    classified = ClassifiedRun(
        kind=run.kind,
        start_s=run.start_s,
        duration_s=run.duration_s,
        symbol=option.symbol,
        units=round(run.duration_s / unit_s, 3),
    )
    tokens = list(state.tokens)
    current = state.current_token
    penalty = state.penalty + option.penalty
    if run.kind == "tone":
        current += option.symbol
        if len(current) > 6:
            penalty += 3.0 + (len(current) - 6) * 2.0
    elif option.symbol == "letter_gap":
        if current:
            tokens.append(current)
            current = ""
    elif option.symbol == "word_gap":
        if current:
            tokens.append(current)
            current = ""
        if tokens and tokens[-1] != "/":
            tokens.append("/")
    return _LatticeState(
        tokens=tuple(tokens),
        current_token=current,
        classified_runs=(*state.classified_runs, classified),
        penalty=round(float(penalty), 6),
    )

def _finalize_lattice_state(state: _LatticeState) -> _LatticeState:
    if not state.current_token:
        return state
    return _LatticeState(
        tokens=(*state.tokens, state.current_token),
        current_token="",
        classified_runs=state.classified_runs,
        penalty=state.penalty,
    )

def _prune_lattice_states(states: list[_LatticeState], *, beam_width: int) -> list[_LatticeState]:
    best_by_key: dict[tuple[tuple[str, ...], str], _LatticeState] = {}
    for state in states:
        key = (state.tokens, state.current_token)
        existing = best_by_key.get(key)
        if existing is None or _lattice_state_sort_key(state) < _lattice_state_sort_key(existing):
            best_by_key[key] = state
    return sorted(best_by_key.values(), key=_lattice_state_sort_key)[:beam_width]

def _lattice_state_sort_key(state: _LatticeState) -> tuple[float, int, int, int]:
    tokens = (*state.tokens, state.current_token) if state.current_token else state.tokens
    token_list = list(tokens)
    text = decode_tokens(token_list) if token_list else ""
    unknowns = text.count("?")
    known = sum(1 for char in text if not char.isspace() and char != "?")
    punctuation = sum(1 for char in text if char and not char.isspace() and not char.isalnum() and char != "?")
    return (state.penalty + unknowns * 3.0 + punctuation * 0.8 - known * 0.08, unknowns, punctuation, -known)
