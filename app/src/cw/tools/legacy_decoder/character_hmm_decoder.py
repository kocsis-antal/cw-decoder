from __future__ import annotations

import numpy as np

from cw.tools.legacy_decoder.base import ClassifiedRun, DecodeResult
from cw.morse_table import MORSE_BY_CHAR, decode_tokens
from cw.tools.legacy_decoder.quality import score_decode_result
from cw.tools.legacy_decoder.config import DecoderConfig
from cw.tools.legacy_decoder.models import DecodeCandidate, _CharHmmState
from cw.tools.legacy_decoder.signal_analysis import _candidate_evidence_score, _mean_run_confidence, _signal_runs
from cw.tools.legacy_decoder.hmm_common import (
    _advance_over_idle_frames,
    _duration_options,
    _remaining_tone_probability,
    _symbol_hmm_candidate_is_plausible,
    _cost_prefix,
    _segment_mean_cost,
)

def _decode_character_hmm_range(
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
    """Decode directly with whole Morse-character duration templates.

    The lower-level symbol-HMM searches dit/dah/gap pieces.  This layer searches
    complete valid Morse character templates against the probability frames.  It
    is still content-neutral: every supported Morse character is available and
    there is no CQ/callsign vocabulary.  The benefit is structural: the beam no
    longer has to rediscover where a character starts and ends by stitching many
    one-element E/T-like tokens.
    """

    hop_s = config.hop_ms / 1000
    if unit_s <= 0 or hop_s <= 0 or range_end <= range_start:
        return []
    unit_frames = max(1.0, unit_s / hop_s)
    while range_start < range_end and probabilities[range_start] < 0.16:
        range_start += 1
    while range_end > range_start and probabilities[range_end - 1] < 0.16:
        range_end -= 1
    if range_end <= range_start:
        return []

    tone_cost_prefix = _cost_prefix(-np.log(np.clip(probabilities, 1e-5, 1.0)))
    gap_cost_prefix = _cost_prefix(-np.log(np.clip(1.0 - probabilities, 1e-5, 1.0)))
    states = [_CharHmmState(position=range_start, tokens=(), classified_runs=(), cost=0.0)]
    finished: list[_CharHmmState] = []
    max_steps = max(10, min(120, int((range_end - range_start) / max(1.0, unit_frames * 2.2)) + 12))
    char_templates = _character_hmm_templates()
    for _ in range(max_steps):
        next_states: list[_CharHmmState] = []
        for state in states:
            position = _advance_over_idle_frames(state.position, range_end, probabilities, max_skip_frames=int(round(unit_frames * 1.8)))
            if position >= range_end:
                finished.append(state)
                continue
            if _remaining_tone_probability(probabilities, position, range_end) < 0.20:
                finished.append(state)
                continue
            for token, char_prior in char_templates:
                advanced = _advance_character_hmm_token(
                    state,
                    token,
                    char_prior,
                    probabilities,
                    frame_times,
                    tone_cost_prefix,
                    gap_cost_prefix,
                    position,
                    range_end,
                    unit_s=unit_s,
                    unit_frames=unit_frames,
                    start_s=start_s,
                    hop_s=hop_s,
                )
                if advanced is None:
                    continue
                char_state, char_end = advanced
                if char_end >= range_end - max(1, int(round(unit_frames * 1.5))):
                    finished.append(char_state)
                for gap_symbol, gap_frames, gap_penalty in _character_hmm_gap_options(unit_frames):
                    gap_end = char_end + gap_frames
                    if gap_end > range_end:
                        continue
                    gap_cost = _segment_mean_cost(gap_cost_prefix, char_end, gap_end) + gap_penalty
                    gap_start_s = start_s + float(frame_times[char_end]) if char_end < len(frame_times) else _last_run_end_s(char_state.classified_runs)
                    gap_run = ClassifiedRun(
                        kind="gap",
                        start_s=gap_start_s,
                        duration_s=max(hop_s, gap_frames * hop_s),
                        symbol=gap_symbol,
                        units=round(max(hop_s, gap_frames * hop_s) / unit_s, 3),
                    )
                    tokens = list(char_state.tokens)
                    if gap_symbol == "word_gap" and tokens and tokens[-1] != "/":
                        tokens.append("/")
                    next_states.append(
                        _CharHmmState(
                            position=gap_end,
                            tokens=tuple(tokens),
                            classified_runs=(*char_state.classified_runs, gap_run),
                            cost=char_state.cost + gap_cost + config.symbol_hmm_transition_penalty,
                        )
                    )
        finished.extend(
            state for state in next_states if state.position >= range_end - max(1, int(round(unit_frames * 1.2)))
        )
        states = _prune_character_hmm_states(
            [state for state in next_states if state.position < range_end],
            beam_width=config.symbol_hmm_beam_width,
        )
        if not states:
            break
    finished.extend(states)
    ranked = _rank_character_hmm_states(finished, max_candidates=config.symbol_hmm_max_candidates)
    output: list[DecodeCandidate] = []
    for state in ranked:
        if not state.tokens:
            continue
        text = decode_tokens(list(state.tokens))
        if not text:
            continue
        decoded = DecodeResult(
            text=text,
            tokens=list(state.tokens),
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
        token_list = [token for token in decoded.tokens if token != "/"]
        short_token_count = sum(1 for token in token_list if len(token) <= 2)
        short_token_density = short_token_count / max(1, len(token_list))
        short_token_penalty = max(0.0, short_token_density - 0.50) * len(token_list) * 4.0
        adjusted_quality = quality.score + normalized_cost * 2.4 + short_token_penalty
        evidence_score = (
            _candidate_evidence_score(decoded, adjusted_quality, confidence)
            - normalized_cost * 0.8
            - short_token_penalty * 1.4
        )
        tone_runs = [run for run in state.classified_runs if run.kind == "tone"]
        if not tone_runs:
            continue
        if not _symbol_hmm_candidate_is_plausible(
            decoded,
            unit_s=unit_s,
            confidence=confidence,
            quality_score=adjusted_quality,
            detector="char-hmm",
        ):
            continue
        output.append(
            DecodeCandidate(
                carrier_hz=round(float(carrier_hz), 3),
                detector="char-hmm",
                threshold_ratio=0.0,
                threshold=float(threshold),
                noise_floor=noise_floor,
                signal_floor=signal_floor,
                duty_cycle=round(float(np.mean(probabilities[range_start:range_end] >= 0.5)), 6),
                unit_s=round(float(unit_s), 6),
                wpm=round(float(1.2 / unit_s), 3) if unit_s > 0 else None,
                text=text,
                tokens=tuple(decoded.tokens),
                quality_score=round(float(adjusted_quality), 6),
                confidence=round(float(confidence), 6),
                evidence_score=round(float(evidence_score), 6),
                start_s=round(float(min(run.start_s for run in tone_runs)), 6),
                end_s=round(float(max(run.start_s + run.duration_s for run in tone_runs)), 6),
                runs=tuple(confidence_runs),
            )
        )
    return output

def _character_hmm_templates() -> tuple[tuple[str, float], ...]:
    templates: list[tuple[str, float]] = []
    for char, token in MORSE_BY_CHAR.items():
        if char == " ":
            continue
        penalty = 0.0
        if not char.isalnum():
            penalty += 0.35
        if len(token) == 1:
            penalty += 0.20
        templates.append((token, penalty))
    return tuple(sorted(templates, key=lambda item: (len(item[0]), item[1], item[0])))

def _advance_character_hmm_token(
    state: _CharHmmState,
    token: str,
    char_prior: float,
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    tone_cost_prefix: np.ndarray,
    gap_cost_prefix: np.ndarray,
    position: int,
    range_end: int,
    *,
    unit_s: float,
    unit_frames: float,
    start_s: float,
    hop_s: float,
) -> tuple[_CharHmmState, int] | None:
    classified: list[ClassifiedRun] = []
    cost = char_prior
    current = position
    for index, symbol in enumerate(token):
        tone_units = 1.0 if symbol == "." else 3.0
        tone_frames = max(1, int(round(unit_frames * tone_units)))
        tone_end = current + tone_frames
        if tone_end > range_end:
            return None
        cost += _segment_mean_cost(tone_cost_prefix, current, tone_end)
        run_start_s = start_s + float(frame_times[current]) if current < len(frame_times) else start_s + current * hop_s
        classified.append(
            ClassifiedRun(
                kind="tone",
                start_s=run_start_s,
                duration_s=max(hop_s, tone_frames * hop_s),
                symbol=symbol,
                units=round(max(hop_s, tone_frames * hop_s) / unit_s, 3),
            )
        )
        current = tone_end
        if index < len(token) - 1:
            gap_frames = max(1, int(round(unit_frames)))
            gap_end = current + gap_frames
            if gap_end > range_end:
                return None
            cost += _segment_mean_cost(gap_cost_prefix, current, gap_end)
            gap_start_s = start_s + float(frame_times[current]) if current < len(frame_times) else run_start_s + tone_frames * hop_s
            classified.append(
                ClassifiedRun(
                    kind="gap",
                    start_s=gap_start_s,
                    duration_s=max(hop_s, gap_frames * hop_s),
                    symbol="element_gap",
                    units=round(max(hop_s, gap_frames * hop_s) / unit_s, 3),
                )
            )
            current = gap_end
    return (
        _CharHmmState(
            position=current,
            tokens=(*state.tokens, token),
            classified_runs=(*state.classified_runs, *classified),
            cost=state.cost + cost + config_safe_transition_floor(len(token)),
        ),
        current,
    )

def config_safe_transition_floor(token_len: int) -> float:
    # Very long tokens already consume more evidence; this tiny floor only keeps
    # the character-template beam from being indifferent to gratuitous splitting.
    return 0.03 * max(1, token_len)

def _character_hmm_gap_options(unit_frames: float) -> tuple[tuple[str, int, float], ...]:
    options: list[tuple[str, int, float]] = []
    for frames, penalty in _duration_options(unit_frames, 3.0, relative_width=0.42, max_options=4):
        options.append(("letter_gap", frames, penalty + 0.04))
    for frames, penalty in _duration_options(unit_frames, 7.0, relative_width=0.38, max_options=3):
        options.append(("word_gap", frames, penalty + 0.12))
    return tuple(sorted(options, key=lambda item: item[2])[:5])

def _last_run_end_s(runs: tuple[ClassifiedRun, ...]) -> float:
    if not runs:
        return 0.0
    last = runs[-1]
    return last.start_s + last.duration_s

def _prune_character_hmm_states(states: list[_CharHmmState], *, beam_width: int) -> list[_CharHmmState]:
    best_by_key: dict[tuple[int, tuple[str, ...]], _CharHmmState] = {}
    for state in states:
        key = (state.position // 2, state.tokens[-5:])
        existing = best_by_key.get(key)
        if existing is None or _character_hmm_state_sort_key(state) < _character_hmm_state_sort_key(existing):
            best_by_key[key] = state
    return sorted(best_by_key.values(), key=_character_hmm_state_sort_key)[:beam_width]

def _rank_character_hmm_states(states: list[_CharHmmState], *, max_candidates: int) -> list[_CharHmmState]:
    best_by_text: dict[str, _CharHmmState] = {}
    for state in states:
        if not state.tokens:
            continue
        text = decode_tokens(list(state.tokens))
        if not text:
            continue
        existing = best_by_text.get(text)
        if existing is None or _character_hmm_state_sort_key(state) < _character_hmm_state_sort_key(existing):
            best_by_text[text] = state
    return sorted(best_by_text.values(), key=_character_hmm_state_sort_key)[: max(1, max_candidates)]

def _character_hmm_state_sort_key(state: _CharHmmState) -> tuple[float, int, int, int]:
    text = decode_tokens(list(state.tokens)) if state.tokens else ""
    unknowns = text.count("?")
    known = sum(1 for char in text if not char.isspace() and char != "?")
    punctuation = sum(1 for char in text if char and not char.isspace() and not char.isalnum() and char != "?")
    avg_cost = state.cost / max(1, len(state.classified_runs))
    return (avg_cost + unknowns * 2.8 + punctuation * 0.7 - known * 0.06, unknowns, punctuation, -known)
