from __future__ import annotations

import numpy as np
from cw.tools.legacy_decoder.base import DetectedRun, _runs_from_activity
from cw.tools.legacy_decoder.quality import score_decode_result
from cw.tools.legacy_decoder.stream_decode import _decode_with_unit, _estimate_unit_from_runs, _unit_candidates, smooth_keying_runs
from cw.tools.legacy_decoder.config import DecoderConfig
from cw.tools.legacy_decoder.models import DecodeCandidate
from cw.tools.legacy_decoder.lattice_decoder import _decode_lattice_candidates
from cw.tools.legacy_decoder.signal_analysis import _activity_probability, _candidate_evidence_score, _mean_run_confidence, _signal_runs

def _decode_energy_candidates(
    energy: np.ndarray,
    frame_times: np.ndarray,
    *,
    carrier_hz: float,
    start_s: float,
    threshold_ratio: float,
    session_gap_s: float,
    config: DecoderConfig,
) -> list[DecodeCandidate]:
    noise_floor = float(np.percentile(energy, 15)) if len(energy) else 0.0
    signal_floor = float(np.percentile(energy, 95)) if len(energy) else 0.0
    threshold = noise_floor + (signal_floor - noise_floor) * float(threshold_ratio)
    if signal_floor <= noise_floor:
        return []
    active = energy > threshold
    raw_runs = _runs_from_activity(active, config.hop_ms / 1000)
    runs = [DetectedRun(run.kind, run.start_s + start_s, run.duration_s) for run in raw_runs]
    runs = smooth_keying_runs(
        runs,
        merge_short_gaps_s=config.merge_short_gaps_ms / 1000,
        drop_short_tones_s=config.drop_short_tones_ms / 1000,
    )
    probabilities = _activity_probability(energy, noise_floor, signal_floor)
    decoded_candidates: list[DecodeCandidate] = []
    for segment_runs in _split_runs_into_segments(runs, session_gap_s=session_gap_s):
        decoded_candidates.extend(
            _decode_segment_candidates(
                segment_runs,
                probabilities,
                frame_times,
                carrier_hz=carrier_hz,
                start_s=start_s,
                threshold_ratio=threshold_ratio,
                detector="threshold",
                threshold=threshold,
                noise_floor=noise_floor,
                signal_floor=signal_floor,
                duty_cycle=round(float(np.mean(active)), 6) if len(active) else 0.0,
                config=config,
            )
        )
    return decoded_candidates

def _decode_segment_candidates(
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
    config: DecoderConfig,
) -> list[DecodeCandidate]:
    if not any(run.kind == "tone" for run in runs):
        return []
    try:
        initial_unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return []
    unit_candidates = _unit_candidates(initial_unit_s, config.unit_candidate_spread, config.unit_candidate_steps)
    decoded_candidates: list[DecodeCandidate] = []
    for unit_s in unit_candidates:
        decoded = _decode_with_unit(runs, carrier_hz, threshold, unit_s, config)
        if not decoded.text:
            continue
        confidence_runs = _signal_runs(decoded, probabilities, frame_times, start_s, config.hop_ms / 1000)
        quality = score_decode_result(decoded)
        confidence = _mean_run_confidence(confidence_runs)
        evidence_score = _candidate_evidence_score(decoded, quality.score, confidence)
        if detector == "viterbi":
            # The Viterbi activity path is a model-based rescue hypothesis for
            # fading tones.  Keep it slightly conservative so a clean hard
            # threshold still wins when both explain the same signal equally well.
            evidence_score -= 3.0
        segment_start = min((run.start_s for run in runs if run.kind == "tone"), default=runs[0].start_s)
        segment_end = max((run.start_s + run.duration_s for run in runs if run.kind == "tone"), default=runs[-1].start_s + runs[-1].duration_s)
        decoded_candidates.append(
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
                quality_score=round(float(quality.score), 6),
                confidence=round(float(confidence), 6),
                evidence_score=round(float(evidence_score), 6),
                start_s=round(float(segment_start), 6),
                end_s=round(float(segment_end), 6),
                runs=tuple(confidence_runs),
            )
        )
        if config.lattice_decoding and config.lattice_max_candidates > 0:
            decoded_candidates.extend(
                _decode_lattice_candidates(
                    runs,
                    probabilities,
                    frame_times,
                    carrier_hz=carrier_hz,
                    start_s=start_s,
                    threshold_ratio=threshold_ratio,
                    detector=f"{detector}-lattice",
                    threshold=threshold,
                    noise_floor=noise_floor,
                    signal_floor=signal_floor,
                    duty_cycle=duty_cycle,
                    unit_s=unit_s,
                    config=config,
                )
            )
    return decoded_candidates

def _split_runs_into_segments(runs: list[DetectedRun], *, session_gap_s: float) -> list[list[DetectedRun]]:
    if not runs:
        return []
    segments: list[list[DetectedRun]] = []
    current: list[DetectedRun] = []
    for run in runs:
        if (
            run.kind == "gap"
            and run.duration_s >= session_gap_s
            and any(item.kind == "tone" for item in current)
        ):
            segments.append(_trim_segment(current))
            current = []
            continue
        current.append(run)
    if current:
        segments.append(_trim_segment(current))
    return [segment for segment in segments if any(run.kind == "tone" for run in segment)]

def _trim_segment(runs: list[DetectedRun]) -> list[DetectedRun]:
    start = 0
    end = len(runs)
    while start < end and runs[start].kind != "tone":
        start += 1
    while end > start and runs[end - 1].kind != "tone":
        end -= 1
    return runs[start:end]
