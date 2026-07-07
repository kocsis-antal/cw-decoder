from __future__ import annotations

from pathlib import Path
from cw.tools.legacy_decoder.config import DecoderConfig
from cw.io.raw_audio import read_raw_audio_slice
from cw.tools.legacy_decoder.models import DecodeReport
from cw.tools.legacy_decoder.carrier_search import detect_carriers_in_audio
from cw.tools.legacy_decoder.carrier_decode import decode_carrier_signal

def decode_raw_file(
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
    symbol_hmm_interval_s: float = 2.0,
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
) -> DecodeReport:
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
    detected = detect_carriers_in_audio(
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
    config = DecoderConfig(
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
        symbol_hmm_interval_s=symbol_hmm_interval_s,
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
        decode_carrier_signal(
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
    return DecodeReport(
        path=str(path),
        sample_rate=sample_rate,
        sample_format=sample_format,
        channels=channels,
        start_s=round(start_s, 6),
        duration_s=round(len(signal) / sample_rate if sample_rate else 0.0, 6),
        detected_carriers=detected,
        carriers=carrier_results,
    )


# Backward-compatible alias for older scripts.
decode_raw_file_nextgen = decode_raw_file
