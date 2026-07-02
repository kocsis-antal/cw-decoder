from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cw.decoder import DecodeResult, DetectedRun, _power_spectrum_frames, _runs_from_activity
from cw.prob_analysis import CarrierCandidate, _detect_carriers_from_spectrum, read_raw_audio_slice
from cw.quality import score_decode_result
from cw.stream_decode import (
    _decode_with_unit,
    _estimate_unit_from_runs,
    _unit_candidates,
    smooth_keying_runs,
)
from cw.stream_models import StreamingConfig


@dataclass(frozen=True)
class NextgenRun:
    kind: str
    start_s: float
    duration_s: float
    confidence: float
    units: float | None = None
    symbol: str = ""


@dataclass(frozen=True)
class NextgenCandidate:
    carrier_hz: float
    detector: str
    threshold_ratio: float
    threshold: float
    noise_floor: float
    signal_floor: float
    duty_cycle: float
    unit_s: float | None
    wpm: float | None
    text: str
    tokens: tuple[str, ...]
    quality_score: float | None
    confidence: float
    evidence_score: float
    start_s: float
    end_s: float
    runs: tuple[NextgenRun, ...]


@dataclass(frozen=True)
class NextgenSession:
    carrier_hz: float
    session_id: int
    start_s: float
    end_s: float
    text: str
    confidence: float
    best: NextgenCandidate | None
    candidates: tuple[NextgenCandidate, ...]


@dataclass(frozen=True)
class NextgenCarrierResult:
    carrier_hz: float
    text: str
    confidence: float
    best: NextgenCandidate | None
    candidates: tuple[NextgenCandidate, ...]
    sessions: tuple[NextgenSession, ...] = ()


@dataclass(frozen=True)
class NextgenDecodeReport:
    path: str
    sample_rate: int
    sample_format: str
    channels: int
    start_s: float
    duration_s: float
    detected_carriers: tuple[CarrierCandidate, ...]
    carriers: tuple[NextgenCarrierResult, ...]


def decode_raw_file_nextgen(
    path: Path,
    *,
    sample_rate: int = 8000,
    sample_format: str = "s16le",
    channels: int = 1,
    start_s: float = 0.0,
    duration_s: float | None = None,
    carriers: tuple[float, ...] = (),
    detect_carriers: int = 5,
    min_tone_hz: float = 200.0,
    max_tone_hz: float = 3000.0,
    min_separation_hz: float = 80.0,
    peak_relative_threshold: float = 0.10,
    detect_frame_ms: float = 80.0,
    detect_hop_ms: float = 10.0,
    lowpass_ms: float = 12.0,
    envelope_hop_ms: float = 5.0,
    threshold_ratios: tuple[float, ...] = (0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
    merge_short_gaps_ms: float = 25.0,
    drop_short_tones_ms: float = 12.0,
    unit_candidate_spread: float = 0.12,
    unit_candidate_steps: int = 5,
    soft_activity: bool = True,
    soft_tone_on_probability: float = 0.56,
    soft_tone_off_probability: float = 0.28,
    soft_bridge_min_probability: float = 0.18,
    soft_bridge_max_gap_ms: float = 90.0,
    soft_bridge_gap_units: float = 1.6,
    adaptive_gap_thresholds: bool = True,
    element_letter_gap_units: float = 2.0,
    default_word_gap_units: float = 7.0,
    gap_cluster_min_ratio: float = 1.45,
    gap_cluster_min_delta_units: float = 1.0,
    gap_cluster_min_lower_count: int = 2,
    session_gap_s: float = 1.2,
    min_session_evidence_score: float = 0.0,
    max_candidates_per_carrier: int = 6,
    max_candidates_per_session: int = 4,
) -> NextgenDecodeReport:
    """Decode raw PCM with the carrier-centric next-generation path.

    The public result is now session-oriented.  A carrier may contain many
    independent transmissions, so each threshold/unit hypothesis is first
    decoded as a timed candidate and then overlapping candidates are grouped
    into sessions.  The selection is content-neutral: no CQ/DE/callsign bias is
    used.  Amount of timed signal evidence, confidence, and timing quality decide.
    """

    signal = read_raw_audio_slice(
        path,
        sample_rate=sample_rate,
        sample_format=sample_format,
        channels=channels,
        start_s=start_s,
        duration_s=duration_s,
    )
    detected = _detect_carriers_nextgen(
        signal,
        sample_rate,
        min_tone_hz=min_tone_hz,
        max_tone_hz=max_tone_hz,
        max_carriers=detect_carriers,
        min_separation_hz=min_separation_hz,
        relative_threshold=peak_relative_threshold,
        frame_ms=detect_frame_ms,
        hop_ms=detect_hop_ms,
    )
    selected_carriers = carriers or tuple(candidate.carrier_hz for candidate in detected)
    config = StreamingConfig(
        frame_ms=envelope_hop_ms,
        hop_ms=envelope_hop_ms,
        min_tone_hz=min_tone_hz,
        max_tone_hz=max_tone_hz,
        threshold_ratios=threshold_ratios,
        soft_activity=soft_activity,
        soft_tone_on_probability=soft_tone_on_probability,
        soft_tone_off_probability=soft_tone_off_probability,
        soft_bridge_min_probability=soft_bridge_min_probability,
        soft_bridge_max_gap_ms=soft_bridge_max_gap_ms,
        soft_bridge_gap_units=soft_bridge_gap_units,
        adaptive_gap_thresholds=adaptive_gap_thresholds,
        element_letter_gap_units=element_letter_gap_units,
        default_word_gap_units=default_word_gap_units,
        gap_cluster_min_ratio=gap_cluster_min_ratio,
        gap_cluster_min_delta_units=gap_cluster_min_delta_units,
        gap_cluster_min_lower_count=gap_cluster_min_lower_count,
        merge_short_gaps_ms=merge_short_gaps_ms,
        drop_short_tones_ms=drop_short_tones_ms,
        unit_candidate_spread=unit_candidate_spread,
        unit_candidate_steps=unit_candidate_steps,
    )
    carrier_results = tuple(
        decode_signal_carrier_nextgen(
            signal,
            sample_rate,
            carrier_hz=carrier_hz,
            start_s=start_s,
            threshold_ratios=threshold_ratios,
            lowpass_ms=lowpass_ms,
            envelope_hop_ms=envelope_hop_ms,
            session_gap_s=session_gap_s,
            config=config,
            max_candidates=max_candidates_per_carrier,
            max_candidates_per_session=max_candidates_per_session,
            min_session_evidence_score=min_session_evidence_score,
        )
        for carrier_hz in selected_carriers
    )
    return NextgenDecodeReport(
        path=str(path),
        sample_rate=sample_rate,
        sample_format=sample_format,
        channels=channels,
        start_s=round(start_s, 6),
        duration_s=round(len(signal) / sample_rate if sample_rate else 0.0, 6),
        detected_carriers=detected,
        carriers=carrier_results,
    )


def decode_signal_carrier_nextgen(
    signal: np.ndarray,
    sample_rate: int,
    *,
    carrier_hz: float,
    start_s: float = 0.0,
    threshold_ratios: tuple[float, ...] = (0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
    lowpass_ms: float = 12.0,
    envelope_hop_ms: float = 5.0,
    session_gap_s: float = 1.2,
    min_session_evidence_score: float = 0.0,
    config: StreamingConfig | None = None,
    max_candidates: int = 6,
    max_candidates_per_session: int = 4,
) -> NextgenCarrierResult:
    config = config or StreamingConfig(
        frame_ms=envelope_hop_ms,
        hop_ms=envelope_hop_ms,
        threshold_ratios=threshold_ratios,
        merge_short_gaps_ms=25.0,
        drop_short_tones_ms=12.0,
        unit_candidate_spread=0.12,
        unit_candidate_steps=5,
    )
    envelope = _baseband_envelope(signal, sample_rate, carrier_hz, lowpass_ms=lowpass_ms)
    energy, frame_times = _envelope_energy_frames(envelope, sample_rate, hop_ms=envelope_hop_ms)
    if len(energy) == 0 or float(np.max(energy)) <= 0:
        return NextgenCarrierResult(carrier_hz=carrier_hz, text="", confidence=0.0, best=None, candidates=(), sessions=())

    ratios = threshold_ratios or (config.threshold_ratio,)
    candidates: list[NextgenCandidate] = []
    for ratio in ratios:
        candidates.extend(
            _decode_energy_candidates(
                energy,
                frame_times,
                carrier_hz=carrier_hz,
                start_s=start_s,
                threshold_ratio=ratio,
                session_gap_s=session_gap_s,
                config=config,
            )
        )
    if config.soft_activity:
        candidates.extend(
            _decode_soft_energy_candidates(
                energy,
                frame_times,
                carrier_hz=carrier_hz,
                start_s=start_s,
                session_gap_s=session_gap_s,
                config=config,
            )
        )
    candidates = _unique_candidates(candidates)
    sessions = _group_candidates_into_sessions(
        candidates,
        carrier_hz=carrier_hz,
        max_candidates_per_session=max_candidates_per_session,
        min_session_evidence_score=min_session_evidence_score,
    )
    flat = [candidate for session in sessions for candidate in session.candidates]
    flat.sort(key=lambda candidate: (-candidate.evidence_score, candidate.quality_score or 1e9))
    kept = tuple(flat[: max(1, max_candidates)])
    best = kept[0] if kept else None
    text = " | ".join(session.text for session in sessions if session.text)
    confidence = _weighted_session_confidence(sessions)
    return NextgenCarrierResult(
        carrier_hz=round(float(carrier_hz), 3),
        text=text,
        confidence=round(float(confidence), 6),
        best=best,
        candidates=kept,
        sessions=sessions,
    )


def report_to_json(report: NextgenDecodeReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, sort_keys=True)


def format_decode_report(report: NextgenDecodeReport) -> str:
    lines: list[str] = []
    lines.append(
        f"raw={report.path} sample_rate={report.sample_rate} format={report.sample_format} "
        f"channels={report.channels} start_s={report.start_s:.3f} duration_s={report.duration_s:.3f}"
    )
    if report.detected_carriers:
        lines.append("detected carriers:")
        for candidate in report.detected_carriers:
            lines.append(f"  {candidate.carrier_hz:8.1f} Hz rel={candidate.relative_power:5.3f}")
    lines.append("decoded carriers:")
    for carrier in report.carriers:
        lines.append(
            f"  {carrier.carrier_hz:8.1f} Hz conf={carrier.confidence:5.2f} text={carrier.text or '<none>'}"
        )
        if carrier.sessions:
            lines.append("      sessions:")
            for session in carrier.sessions:
                lines.append(
                    f"        s{session.session_id:<2} {session.start_s:8.3f}-{session.end_s:8.3f} "
                    f"conf={session.confidence:4.2f} text={session.text or '<none>'}"
                )
                if not session.candidates:
                    continue
                lines.append("             rank det thr unit_ms wpm score conf evidence text")
                for index, candidate in enumerate(session.candidates, start=1):
                    lines.append("             " + _format_candidate_row(index, candidate))
        elif carrier.candidates:
            lines.append("      rank det thr unit_ms wpm score conf evidence text")
            for index, candidate in enumerate(carrier.candidates, start=1):
                lines.append("      " + _format_candidate_row(index, candidate))
    return "\n".join(lines)


def _format_candidate_row(index: int, candidate: NextgenCandidate) -> str:
    unit_ms = "-" if candidate.unit_s is None else f"{candidate.unit_s * 1000:7.1f}"
    wpm = "-" if candidate.wpm is None else f"{candidate.wpm:5.1f}"
    score = "-" if candidate.quality_score is None else f"{candidate.quality_score:5.1f}"
    return (
        f"{index:>4} {candidate.detector[:4]:>4} {candidate.threshold_ratio:>4.2f} {unit_ms:>7} {wpm:>5} "
        f"{score:>5} {candidate.confidence:>4.2f} {candidate.evidence_score:>8.2f} "
        f"{candidate.text or '<none>'}"
    )


def _detect_carriers_nextgen(
    signal: np.ndarray,
    sample_rate: int,
    *,
    min_tone_hz: float,
    max_tone_hz: float,
    max_carriers: int,
    min_separation_hz: float,
    relative_threshold: float,
    frame_ms: float,
    hop_ms: float,
) -> tuple[CarrierCandidate, ...]:
    if max_carriers <= 0 or len(signal) == 0:
        return ()
    spectrum, freqs = _power_spectrum_frames(signal, sample_rate, frame_ms, hop_ms)
    return _detect_carriers_from_spectrum(
        spectrum,
        freqs,
        min_tone_hz=min_tone_hz,
        max_tone_hz=max_tone_hz,
        max_carriers=max_carriers,
        min_separation_hz=min_separation_hz,
        relative_threshold=relative_threshold,
    )


def _baseband_envelope(signal: np.ndarray, sample_rate: int, carrier_hz: float, *, lowpass_ms: float) -> np.ndarray:
    mono = signal.astype(np.float32, copy=False)
    if len(mono) == 0:
        return np.asarray([], dtype=np.float32)
    time = np.arange(len(mono), dtype=np.float32) / float(sample_rate)
    mixed = mono * np.exp(-2j * np.pi * float(carrier_hz) * time)
    window_len = max(5, int(round(sample_rate * lowpass_ms / 1000)))
    if window_len % 2 == 0:
        window_len += 1
    window = np.hanning(window_len).astype(np.float32)
    if float(np.sum(window)) <= 0:
        window = np.ones(window_len, dtype=np.float32)
    window = window / np.sum(window)
    filtered_real = np.convolve(mixed.real, window, mode="same")
    filtered_imag = np.convolve(mixed.imag, window, mode="same")
    envelope = np.sqrt(filtered_real * filtered_real + filtered_imag * filtered_imag)
    return envelope.astype(np.float32, copy=False)


def _envelope_energy_frames(envelope: np.ndarray, sample_rate: int, *, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(round(sample_rate * hop_ms / 1000)))
    if len(envelope) == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    starts = np.arange(0, len(envelope), hop, dtype=np.int64)
    energy = np.empty(len(starts), dtype=np.float32)
    for out_index, start in enumerate(starts):
        frame = envelope[start : min(start + hop, len(envelope))]
        energy[out_index] = float(np.mean(frame * frame)) if len(frame) else 0.0
    times = starts.astype(np.float32) / float(sample_rate)
    return energy, times


def _decode_soft_energy_candidates(
    energy: np.ndarray,
    frame_times: np.ndarray,
    *,
    carrier_hz: float,
    start_s: float,
    session_gap_s: float,
    config: StreamingConfig,
) -> list[NextgenCandidate]:
    """Decode a carrier using probability/hysteresis activity, not a hard threshold.

    The hard-threshold candidates are still useful, but weak CW often fades
    below one threshold for part of a dash.  This path keeps ambiguous frames in
    the previous state and can bridge short, non-silent gaps inside tones when
    the local envelope still carries some signal evidence.  It is deliberately
    content-neutral: it only changes the timed tone/gap evidence that reaches
    the Morse decoder.
    """

    if len(energy) == 0:
        return []
    noise_floor = float(np.percentile(energy, 15))
    signal_floor = float(np.percentile(energy, 95))
    if signal_floor <= noise_floor:
        return []
    probabilities = _activity_probability(energy, noise_floor, signal_floor)
    active = _hysteresis_activity(
        probabilities,
        on_probability=config.soft_tone_on_probability,
        off_probability=config.soft_tone_off_probability,
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

    decoded_candidates: list[NextgenCandidate] = []
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
                detector="soft",
                threshold=threshold,
                noise_floor=noise_floor,
                signal_floor=signal_floor,
                duty_cycle=round(float(np.mean(active)), 6) if len(active) else 0.0,
                config=config,
            )
        )
    return decoded_candidates


def _hysteresis_activity(
    probabilities: np.ndarray,
    *,
    on_probability: float,
    off_probability: float,
) -> np.ndarray:
    active = np.zeros(len(probabilities), dtype=bool)
    state = False
    for index, probability in enumerate(probabilities):
        value = float(probability)
        if state:
            if value <= off_probability:
                state = False
        elif value >= on_probability:
            state = True
        active[index] = state
    return active


def _bridge_soft_fade_gaps(
    runs: list[DetectedRun],
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    start_s: float,
    config: StreamingConfig,
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


def _decode_energy_candidates(
    energy: np.ndarray,
    frame_times: np.ndarray,
    *,
    carrier_hz: float,
    start_s: float,
    threshold_ratio: float,
    session_gap_s: float,
    config: StreamingConfig,
) -> list[NextgenCandidate]:
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
    decoded_candidates: list[NextgenCandidate] = []
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
    config: StreamingConfig,
) -> list[NextgenCandidate]:
    if not any(run.kind == "tone" for run in runs):
        return []
    try:
        initial_unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return []
    unit_candidates = _unit_candidates(initial_unit_s, config.unit_candidate_spread, config.unit_candidate_steps)
    decoded_candidates: list[NextgenCandidate] = []
    for unit_s in unit_candidates:
        decoded = _decode_with_unit(runs, carrier_hz, threshold, unit_s, config)
        if not decoded.text:
            continue
        confidence_runs = _nextgen_runs(decoded, probabilities, frame_times, start_s, config.hop_ms / 1000)
        quality = score_decode_result(decoded)
        confidence = _mean_run_confidence(confidence_runs)
        evidence_score = _candidate_evidence_score(decoded, quality.score, confidence)
        if detector == "soft":
            # Soft/hysteresis candidates are valuable rescue hypotheses for fading
            # tones, but they are deliberately conservative: a clean hard-threshold
            # decode with comparable evidence should still win.
            evidence_score -= 5.0
        segment_start = min((run.start_s for run in runs if run.kind == "tone"), default=runs[0].start_s)
        segment_end = max((run.start_s + run.duration_s for run in runs if run.kind == "tone"), default=runs[-1].start_s + runs[-1].duration_s)
        decoded_candidates.append(
            NextgenCandidate(
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


def _group_candidates_into_sessions(
    candidates: list[NextgenCandidate],
    *,
    carrier_hz: float,
    max_candidates_per_session: int,
    min_session_evidence_score: float,
) -> tuple[NextgenSession, ...]:
    if not candidates:
        return ()
    candidates = sorted(candidates, key=lambda candidate: (candidate.start_s, candidate.end_s, -candidate.evidence_score))
    groups: list[list[NextgenCandidate]] = []
    for candidate in candidates:
        placed = False
        for group in groups:
            if _candidate_overlaps_group(candidate, group):
                group.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])
    sessions: list[NextgenSession] = []
    for group in groups:
        group.sort(key=lambda candidate: (-candidate.evidence_score, candidate.quality_score or 1e9, -candidate.confidence))
        kept = tuple(group[: max(1, max_candidates_per_session)])
        best = kept[0]
        if best.evidence_score < min_session_evidence_score:
            continue
        sessions.append(
            NextgenSession(
                carrier_hz=round(float(carrier_hz), 3),
                session_id=len(sessions) + 1,
                start_s=round(min(candidate.start_s for candidate in group), 6),
                end_s=round(max(candidate.end_s for candidate in group), 6),
                text=best.text,
                confidence=best.confidence,
                best=best,
                candidates=kept,
            )
        )
    sessions.sort(key=lambda session: (session.start_s, session.end_s))
    # Re-number after sort in case a late-overlapping candidate appended to an older group.
    return tuple(
        NextgenSession(
            carrier_hz=session.carrier_hz,
            session_id=index,
            start_s=session.start_s,
            end_s=session.end_s,
            text=session.text,
            confidence=session.confidence,
            best=session.best,
            candidates=session.candidates,
        )
        for index, session in enumerate(sessions, start=1)
    )


def _candidate_overlaps_group(candidate: NextgenCandidate, group: list[NextgenCandidate]) -> bool:
    group_start = min(item.start_s for item in group)
    group_end = max(item.end_s for item in group)
    overlap = min(candidate.end_s, group_end) - max(candidate.start_s, group_start)
    if overlap <= 0:
        # Thresholds can move segment edges a little.  Close starts likely refer to the same keying burst.
        return min(abs(candidate.start_s - item.start_s) for item in group) <= 0.35
    shorter = max(1e-6, min(candidate.end_s - candidate.start_s, group_end - group_start))
    return overlap / shorter >= 0.25


def _weighted_session_confidence(sessions: tuple[NextgenSession, ...]) -> float:
    weighted = 0.0
    total = 0.0
    for session in sessions:
        duration = max(0.05, session.end_s - session.start_s)
        weighted += session.confidence * duration
        total += duration
    return weighted / total if total > 0 else 0.0


def _activity_probability(energy: np.ndarray, noise_floor: float, signal_floor: float) -> np.ndarray:
    contrast = max(signal_floor - noise_floor, 1e-12)
    linear = np.clip((energy - noise_floor) / contrast, 0.0, 1.0)
    return (linear * linear * (3.0 - 2.0 * linear)).astype(np.float32)


def _nextgen_runs(
    decoded: DecodeResult,
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    start_s: float,
    hop_s: float,
) -> list[NextgenRun]:
    output: list[NextgenRun] = []
    for classified in decoded.classified_runs:
        start = classified.start_s - start_s
        end = start + classified.duration_s
        if len(frame_times):
            mask = (frame_times >= start) & (frame_times < end)
            values = probabilities[mask]
        else:
            values = np.asarray([], dtype=np.float32)
        if len(values) == 0:
            center_index = int(round(start / hop_s)) if hop_s > 0 else 0
            if 0 <= center_index < len(probabilities):
                values = probabilities[center_index : center_index + 1]
        if len(values) == 0:
            confidence = 0.0
        elif classified.kind == "tone":
            confidence = float(np.mean(values))
        else:
            confidence = float(np.mean(1.0 - values))
        output.append(
            NextgenRun(
                kind=classified.kind,
                start_s=round(float(classified.start_s), 6),
                duration_s=round(float(classified.duration_s), 6),
                confidence=round(max(0.0, min(1.0, confidence)), 6),
                units=round(float(classified.units), 3),
                symbol=classified.symbol,
            )
        )
    return output


def _mean_run_confidence(runs: list[NextgenRun]) -> float:
    if not runs:
        return 0.0
    weights = [1.5 if run.kind == "tone" else 1.0 for run in runs]
    return float(sum(run.confidence * weight for run, weight in zip(runs, weights)) / sum(weights))


def _candidate_evidence_score(decoded: DecodeResult, quality_score: float, confidence: float) -> float:
    known_chars = sum(1 for char in decoded.text if not char.isspace() and char != "?")
    unknowns = decoded.text.count("?")
    punctuation = sum(1 for char in decoded.text if not char.isspace() and not char.isalnum() and char != "?")
    token_count = len([token for token in decoded.tokens if token != "/"])
    duration_bonus = min(18.0, len(decoded.classified_runs) * 0.18)
    return (
        known_chars * 1.6
        + token_count * 0.8
        + confidence * 18.0
        + duration_bonus
        - unknowns * 2.0
        - punctuation * 0.6
        - quality_score * 0.45
    )


def _unique_candidates(candidates: list[NextgenCandidate]) -> list[NextgenCandidate]:
    best_by_key: dict[tuple[str, tuple[str, ...], int, int], NextgenCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.text,
            candidate.tokens,
            int(round(candidate.start_s * 10)),
            int(round(candidate.end_s * 10)),
        )
        existing = best_by_key.get(key)
        if existing is None or candidate.evidence_score > existing.evidence_score:
            best_by_key[key] = candidate
    return list(best_by_key.values())
