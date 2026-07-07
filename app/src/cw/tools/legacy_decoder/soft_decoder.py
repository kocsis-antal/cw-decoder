from __future__ import annotations

import numpy as np
from cw.tools.legacy_decoder.base import DetectedRun, _runs_from_activity
from cw.tools.legacy_decoder.stream_decode import _estimate_unit_from_runs, smooth_keying_runs
from cw.tools.legacy_decoder.config import DecoderConfig
from cw.tools.legacy_decoder.models import DecodeCandidate
from cw.tools.legacy_decoder.signal_analysis import _activity_probability
from cw.tools.legacy_decoder.threshold_decoder import _decode_segment_candidates, _split_runs_into_segments

def _decode_soft_energy_candidates(
    energy: np.ndarray,
    frame_times: np.ndarray,
    *,
    carrier_hz: float,
    start_s: float,
    session_gap_s: float,
    config: DecoderConfig,
) -> list[DecodeCandidate]:
    """Decode a carrier using a probability Viterbi activity path.

    The hard-threshold candidates are still useful, but weak CW often fades
    below one threshold for part of a dash.  This path uses a two-state
    tone/gap HMM over the full envelope window, then optionally bridges short,
    non-silent fade gaps.  It is deliberately content-neutral: it only changes
    the timed tone/gap evidence that reaches the Morse decoder.
    """

    if len(energy) == 0:
        return []
    noise_floor = float(np.percentile(energy, 15))
    signal_floor = float(np.percentile(energy, 95))
    if signal_floor <= noise_floor:
        return []
    probabilities = _activity_probability(energy, noise_floor, signal_floor)
    active = _viterbi_activity(
        probabilities,
        transition_penalty=config.viterbi_transition_penalty,
    )
    if not np.any(active):
        return []

    raw_runs = _runs_from_activity(active, config.hop_ms / 1000)
    runs = [DetectedRun(run.kind, run.start_s + start_s, run.duration_s) for run in raw_runs]
    runs = smooth_keying_runs(
        runs,
        merge_short_gaps_s=config.merge_short_gaps_ms / 1000,
        drop_short_tones_s=config.drop_short_tones_ms / 1000,
    )
    runs = _bridge_soft_fade_gaps(runs, probabilities, frame_times, start_s, config)
    runs = smooth_keying_runs(
        runs,
        merge_short_gaps_s=config.merge_short_gaps_ms / 1000,
        drop_short_tones_s=config.drop_short_tones_ms / 1000,
    )

    decoded_candidates: list[DecodeCandidate] = []
    threshold = noise_floor + (signal_floor - noise_floor) * config.soft_tone_on_probability
    for segment_runs in _split_runs_into_segments(runs, session_gap_s=session_gap_s):
        decoded_candidates.extend(
            _decode_segment_candidates(
                segment_runs,
                probabilities,
                frame_times,
                carrier_hz=carrier_hz,
                start_s=start_s,
                threshold_ratio=config.soft_tone_on_probability,
                detector="viterbi",
                threshold=threshold,
                noise_floor=noise_floor,
                signal_floor=signal_floor,
                duty_cycle=round(float(np.mean(active)), 6) if len(active) else 0.0,
                config=config,
            )
        )
    return decoded_candidates

def _viterbi_activity(probabilities: np.ndarray, *, transition_penalty: float) -> np.ndarray:
    """Return the most likely tone/gap path for activity probabilities.

    This is a two-state HMM/Viterbi decoder over envelope frames.  Unlike a hard
    threshold or one-way hysteresis gate, it optimizes the whole window at once:
    a short dip inside a dash is kept as tone when paying two state transitions
    would be less likely than a brief low-probability tone frame, while a real
    silent Morse gap is preserved when the accumulated gap evidence is stronger.
    """

    if len(probabilities) == 0:
        return np.zeros(0, dtype=bool)
    eps = 1e-6
    p = np.clip(probabilities.astype(np.float64, copy=False), eps, 1.0 - eps)
    tone_emit = -np.log(p)
    gap_emit = -np.log(1.0 - p)

    # state 0 = gap, state 1 = tone
    costs = np.zeros((len(p), 2), dtype=np.float64)
    back = np.zeros((len(p), 2), dtype=np.int8)
    costs[0, 0] = gap_emit[0]
    costs[0, 1] = tone_emit[0] + transition_penalty * 0.5
    for index in range(1, len(p)):
        stay_gap = costs[index - 1, 0]
        switch_to_gap = costs[index - 1, 1] + transition_penalty
        if stay_gap <= switch_to_gap:
            costs[index, 0] = stay_gap + gap_emit[index]
            back[index, 0] = 0
        else:
            costs[index, 0] = switch_to_gap + gap_emit[index]
            back[index, 0] = 1

        stay_tone = costs[index - 1, 1]
        switch_to_tone = costs[index - 1, 0] + transition_penalty
        if stay_tone <= switch_to_tone:
            costs[index, 1] = stay_tone + tone_emit[index]
            back[index, 1] = 1
        else:
            costs[index, 1] = switch_to_tone + tone_emit[index]
            back[index, 1] = 0

    state = int(costs[-1, 1] < costs[-1, 0])
    active = np.zeros(len(p), dtype=bool)
    for index in range(len(p) - 1, -1, -1):
        active[index] = state == 1
        state = int(back[index, state]) if index > 0 else state
    return active

def _bridge_soft_fade_gaps(
    runs: list[DetectedRun],
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    start_s: float,
    config: DecoderConfig,
) -> list[DetectedRun]:
    if not runs or not any(run.kind == "gap" for run in runs):
        return runs
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        unit_s = None
    max_gap_s = config.soft_bridge_max_gap_ms / 1000
    if unit_s is not None and config.soft_bridge_gap_units > 0:
        max_gap_s = max(max_gap_s, unit_s * config.soft_bridge_gap_units)
    if max_gap_s <= 0:
        return runs

    bridged: list[DetectedRun] = []
    index = 0
    while index < len(runs):
        run = runs[index]
        if (
            run.kind == "gap"
            and 0 < index < len(runs) - 1
            and runs[index - 1].kind == "tone"
            and runs[index + 1].kind == "tone"
            and run.duration_s <= max_gap_s
            and _mean_probability_for_run(run, probabilities, frame_times, start_s) >= config.soft_bridge_min_probability
        ):
            previous = bridged.pop()
            next_run = runs[index + 1]
            bridged.append(
                DetectedRun(
                    "tone",
                    previous.start_s,
                    previous.duration_s + run.duration_s + next_run.duration_s,
                )
            )
            index += 2
            continue
        bridged.append(run)
        index += 1
    return bridged

def _mean_probability_for_run(
    run: DetectedRun,
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    start_s: float,
) -> float:
    if len(probabilities) == 0 or len(frame_times) == 0:
        return 0.0
    relative_start = run.start_s - start_s
    relative_end = relative_start + run.duration_s
    mask = (frame_times >= relative_start) & (frame_times < relative_end)
    values = probabilities[mask]
    if len(values) == 0:
        center = relative_start + run.duration_s / 2
        index = int(np.searchsorted(frame_times, center))
        if 0 <= index < len(probabilities):
            values = probabilities[index : index + 1]
    return float(np.mean(values)) if len(values) else 0.0

def _has_strong_direct_candidate(candidates: list[DecodeCandidate]) -> bool:
    """Return true when the existing signal path is already good enough.

    The direct Symbol-HMM is more expensive than threshold/Viterbi run decoding.
    It is most valuable as a structural rescue path for ambiguous or fading
    envelopes, not for re-decoding clean signals that already have a low-score,
    high-confidence interpretation.  This keeps streaming operation usable while the
    same integrated decoder stack is used for files, raw replay, and stdin.
    """

    for candidate in candidates:
        compact = "".join(char for char in candidate.text if not char.isspace())
        known = sum(1 for char in compact if char != "?")
        if (
            known >= 4
            and candidate.confidence >= 0.70
            and candidate.evidence_score >= 22.0
            and (candidate.quality_score is None or candidate.quality_score <= 10.0)
        ):
            return True
    return False
