from __future__ import annotations

import numpy as np
from cw.tools.legacy_decoder.config import DecoderConfig
from cw.tools.legacy_decoder.models import DecodeCandidate, CarrierDecodeResult
from cw.tools.legacy_decoder.signal_analysis import _baseband_envelope, _envelope_energy_frames
from cw.tools.legacy_decoder.threshold_decoder import _decode_energy_candidates
from cw.tools.legacy_decoder.soft_decoder import _decode_soft_energy_candidates, _has_strong_direct_candidate
from cw.tools.legacy_decoder.hmm_decoder import _decode_symbol_hmm_energy_candidates
from cw.tools.legacy_decoder.session_grouping import _group_candidates_into_sessions, _unique_candidates, _weighted_session_confidence

def decode_carrier_signal(
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
    config: DecoderConfig | None = None,
    max_candidates: int = 6,
    max_candidates_per_session: int = 4,
) -> CarrierDecodeResult:
    config = config or DecoderConfig(
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
        return CarrierDecodeResult(carrier_hz=carrier_hz, text="", confidence=0.0, best=None, candidates=(), sessions=())

    ratios = threshold_ratios or (config.threshold_ratio,)
    candidates: list[DecodeCandidate] = []
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
    return CarrierDecodeResult(
        carrier_hz=round(float(carrier_hz), 3),
        text=text,
        confidence=round(float(confidence), 6),
        best=best,
        candidates=kept,
        sessions=sessions,
    )
