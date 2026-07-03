from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cw.decoder import ClassifiedRun, DecodeResult, DetectedRun, _power_spectrum_frames, _runs_from_activity
from cw.morse_table import CHAR_BY_MORSE, MORSE_BY_CHAR, decode_tokens
from cw.prob_analysis import CarrierCandidate, _detect_carriers_from_spectrum, read_raw_audio_slice
from cw.quality import score_decode_result
from cw.stream_decode import (
    _adaptive_letter_word_boundary_units,
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


@dataclass(frozen=True)
class _SymbolOption:
    symbol: str
    penalty: float


@dataclass(frozen=True)
class _CharHmmState:
    position: int
    tokens: tuple[str, ...]
    classified_runs: tuple[ClassifiedRun, ...]
    cost: float


@dataclass(frozen=True)
class _SymbolHmmState:
    position: int
    tokens: tuple[str, ...]
    current_token: str
    classified_runs: tuple[ClassifiedRun, ...]
    cost: float


_VALID_MORSE_TOKENS = frozenset(CHAR_BY_MORSE.keys())
_VALID_MORSE_PREFIXES = frozenset(
    code[:index]
    for code in _VALID_MORSE_TOKENS
    for index in range(1, len(code) + 1)
)


@dataclass(frozen=True)
class _LatticeState:
    tokens: tuple[str, ...]
    current_token: str
    classified_runs: tuple[ClassifiedRun, ...]
    penalty: float


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
    viterbi_transition_penalty: float = 1.15,
    symbol_hmm_decoding: bool = True,
    symbol_hmm_beam_width: int = 16,
    symbol_hmm_max_candidates: int = 3,
    symbol_hmm_unit_spread: float = 0.18,
    symbol_hmm_unit_steps: int = 3,
    symbol_hmm_transition_penalty: float = 0.18,
    symbol_hmm_min_unit_s: float = 0.025,
    symbol_hmm_max_unit_s: float = 0.250,
    symbol_hmm_live_interval_s: float = 2.0,
    lattice_decoding: bool = True,
    lattice_beam_width: int = 12,
    lattice_max_candidates: int = 3,
    lattice_tone_margin_units: float = 0.45,
    lattice_gap_margin_units: float = 0.60,
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
        viterbi_transition_penalty=viterbi_transition_penalty,
        symbol_hmm_decoding=symbol_hmm_decoding,
        symbol_hmm_beam_width=symbol_hmm_beam_width,
        symbol_hmm_max_candidates=symbol_hmm_max_candidates,
        symbol_hmm_unit_spread=symbol_hmm_unit_spread,
        symbol_hmm_unit_steps=symbol_hmm_unit_steps,
        symbol_hmm_transition_penalty=symbol_hmm_transition_penalty,
        symbol_hmm_min_unit_s=symbol_hmm_min_unit_s,
        symbol_hmm_max_unit_s=symbol_hmm_max_unit_s,
        symbol_hmm_live_interval_s=symbol_hmm_live_interval_s,
        lattice_decoding=lattice_decoding,
        lattice_beam_width=lattice_beam_width,
        lattice_max_candidates=lattice_max_candidates,
        lattice_tone_margin_units=lattice_tone_margin_units,
        lattice_gap_margin_units=lattice_gap_margin_units,
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
    if config.symbol_hmm_decoding:
        candidates.extend(
            _decode_symbol_hmm_energy_candidates(
                energy,
                frame_times,
                carrier_hz=carrier_hz,
                start_s=start_s,
                session_gap_s=session_gap_s,
                config=config,
                include_character_templates=not _has_strong_direct_candidate(candidates),
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
        f"{index:>4} {_detector_label(candidate.detector):>4} {candidate.threshold_ratio:>4.2f} {unit_ms:>7} {wpm:>5} "
        f"{score:>5} {candidate.confidence:>4.2f} {candidate.evidence_score:>8.2f} "
        f"{candidate.text or '<none>'}"
    )


def _detector_label(detector: str) -> str:
    if detector == "threshold":
        return "thr"
    if detector == "viterbi":
        return "vit"
    if detector in {"symbol-hmm", "char-hmm"}:
        return "hmm"
    if detector.endswith("-lattice"):
        return "lat"
    return detector[:4]


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



def _has_strong_direct_candidate(candidates: list[NextgenCandidate]) -> bool:
    """Return true when the existing signal path is already good enough.

    The direct Symbol-HMM is more expensive than threshold/Viterbi run decoding.
    It is most valuable as a structural rescue path for ambiguous or fading
    envelopes, not for re-decoding clean signals that already have a low-score,
    high-confidence interpretation.  This keeps live operation usable while the
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


def _decode_symbol_hmm_energy_candidates(
    energy: np.ndarray,
    frame_times: np.ndarray,
    *,
    carrier_hz: float,
    start_s: float,
    session_gap_s: float,
    config: StreamingConfig,
    include_character_templates: bool = True,
) -> list[NextgenCandidate]:
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
    decoded_candidates: list[NextgenCandidate] = []
    for range_start, range_end in ranges:
        range_units = _symbol_hmm_range_unit_candidates(
            probabilities,
            frame_times,
            range_start,
            range_end,
            base_unit_candidates,
            config,
        )
        first_pass: list[NextgenCandidate] = []
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
    config: StreamingConfig,
) -> tuple[float, ...]:
    """Return unit hypotheses for one probability session range.

    The old direct HMM used one unit estimate for the whole rolling window.  A
    live channel can contain several operators/turns with slightly different
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
    first_pass: list[NextgenCandidate],
    existing_units: tuple[float, ...],
    config: StreamingConfig,
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


def _filter_symbol_hmm_unit_candidates(units: tuple[float, ...], config: StreamingConfig) -> tuple[float, ...]:
    return tuple(
        unit
        for unit in units
        if config.symbol_hmm_min_unit_s <= unit <= config.symbol_hmm_max_unit_s
    )


def _estimate_unit_from_symbol_runs(runs: tuple[NextgenRun, ...]) -> float | None:
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


def _symbol_run_target_units(run: NextgenRun) -> float | None:
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
    config: StreamingConfig,
) -> list[NextgenCandidate]:
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
    output: list[NextgenCandidate] = []
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
        confidence_runs = _nextgen_runs(decoded, probabilities, frame_times, start_s, hop_s)
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
            NextgenCandidate(
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
    config: StreamingConfig,
) -> list[NextgenCandidate]:
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
    output: list[NextgenCandidate] = []
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
        confidence_runs = _nextgen_runs(decoded, probabilities, frame_times, start_s, hop_s)
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
            NextgenCandidate(
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


def _cost_prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(values.astype(np.float64, copy=False))])


def _segment_mean_cost(prefix: np.ndarray, start: int, end: int) -> float:
    if end <= start:
        return 20.0
    return float(prefix[end] - prefix[start]) / (end - start)


def _advance_over_idle_frames(position: int, end: int, probabilities: np.ndarray, *, max_skip_frames: int) -> int:
    skipped = 0
    while position < end and probabilities[position] < 0.18 and skipped < max_skip_frames:
        position += 1
        skipped += 1
    return position


def _remaining_tone_probability(probabilities: np.ndarray, position: int, end: int) -> float:
    if position >= end:
        return 0.0
    return float(np.max(probabilities[position:end]))


def _duration_options(unit_frames: float, units: float, *, relative_width: float, max_options: int = 5) -> tuple[tuple[int, float], ...]:
    center = max(1.0, unit_frames * units)
    low = max(1, int(round(center * (1.0 - relative_width))))
    high = max(low, int(round(center * (1.0 + relative_width))))
    if high == low:
        candidates = [low]
    else:
        raw = np.linspace(low, high, num=min(max_options, high - low + 1))
        candidates = sorted({max(1, int(round(value))) for value in raw})
    output: list[tuple[int, float]] = []
    for frames in candidates:
        actual_units = frames / max(unit_frames, 1e-6)
        penalty = abs(actual_units - units) / max(units, 1e-6)
        output.append((frames, float(penalty)))
    return tuple(sorted(output, key=lambda item: item[1]))


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

def _symbol_hmm_candidate_is_plausible(
    decoded: DecodeResult,
    *,
    unit_s: float,
    confidence: float,
    quality_score: float,
    detector: str,
) -> bool:
    """Reject generic duration-HMM overfits before they reach session scoring.

    The direct HMM is deliberately powerful: it can explain probability frames
    without pre-cut runs.  That also means it can over-explain short noisy
    stretches as absurdly fast E/T-like Morse.  This validation is not a content
    prior; it only enforces physically plausible keying speed and a minimum
    amount of signal support.
    """

    if not 0.025 <= unit_s <= 0.250:
        return False
    compact = "".join(char for char in decoded.text if not char.isspace())
    known_chars = sum(1 for char in compact if char != "?")
    if known_chars < 2:
        return False
    token_list = [token for token in decoded.tokens if token != "/"]
    if not token_list:
        return False
    short_tokens = sum(1 for token in token_list if len(token) <= 1)
    short_density = short_tokens / max(1, len(token_list))
    # Long strings consisting mostly of one-element characters are a common HMM
    # failure mode on noise.  Permit short fragments, but reject long degeneracy.
    if len(token_list) >= 8 and short_density > 0.72:
        return False
    # High quality_score is bad.  The threshold is deliberately loose because
    # this is only an overfit guard; normal candidate ranking remains responsible
    # for choosing between plausible alternatives.
    if quality_score > 42.0 and detector == "symbol-hmm":
        return False
    if quality_score > 48.0 and detector == "char-hmm":
        return False
    if confidence < 0.42 and known_chars < 5:
        return False
    return True


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
        if detector == "viterbi":
            # The Viterbi activity path is a model-based rescue hypothesis for
            # fading tones.  Keep it slightly conservative so a clean hard
            # threshold still wins when both explain the same signal equally well.
            evidence_score -= 3.0
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
    config: StreamingConfig,
) -> list[NextgenCandidate]:
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
    output: list[NextgenCandidate] = []
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
        confidence_runs = _nextgen_runs(decoded, probabilities, frame_times, start_s, config.hop_ms / 1000)
        confidence = _mean_run_confidence(confidence_runs)
        adjusted_quality = quality.score + state.penalty * 4.0
        evidence_score = _candidate_evidence_score(decoded, adjusted_quality, confidence) - state.penalty * 1.8
        segment_start = min((run.start_s for run in runs if run.kind == "tone"), default=runs[0].start_s)
        segment_end = max(
            (run.start_s + run.duration_s for run in runs if run.kind == "tone"),
            default=runs[-1].start_s + runs[-1].duration_s,
        )
        output.append(
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
    config: StreamingConfig,
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


def _gap_symbol_penalty(units: float, symbol: str, config: StreamingConfig) -> float:
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
    word_gap_count = decoded.tokens.count("/")
    duration_bonus = min(18.0, len(decoded.classified_runs) * 0.18)
    fragmentation_penalty = _text_fragmentation_penalty(decoded.text)
    return (
        known_chars * 1.6
        + token_count * 0.8
        + min(word_gap_count, 8) * 0.8
        + confidence * 18.0
        + duration_bonus
        - unknowns * 2.0
        - punctuation * 0.6
        - quality_score * 0.45
        - fragmentation_penalty
    )


def _text_fragmentation_penalty(text: str) -> float:
    """Penalize over-fragmented text-neutral hypotheses.

    Weak/high-threshold decodes sometimes explain a continuous keyed burst as a
    stream of isolated one-character "words" (for example ``Q C Q C Q``).
    That is usually a symptom of false word-gap decisions, not better signal
    evidence.  Keep short exchanges such as ``R R`` or ``E E`` possible, but
    make longer, heavily fragmented candidates compete on their actual timing
    quality instead of winning by simply producing more separated characters.
    """

    words = [word for word in text.split() if word]
    if len(words) < 4:
        return 0.0

    cleaned = ["".join(char for char in word if char.isalnum() or char == "?") for word in words]
    cleaned = [word for word in cleaned if word]
    if len(cleaned) < 4:
        return 0.0

    one_char_words = sum(1 for word in cleaned if len(word) == 1)
    one_char_density = one_char_words / max(1, len(cleaned))
    # Four or more isolated single-character words in one session is strong
    # evidence of gap fragmentation.  Density catches alternating patterns even
    # when a few longer tokens are present.
    excess_singletons = max(0, one_char_words - 3)
    density_penalty = max(0.0, one_char_density - 0.38) * len(cleaned) * 7.0
    return excess_singletons * 2.8 + density_penalty


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
