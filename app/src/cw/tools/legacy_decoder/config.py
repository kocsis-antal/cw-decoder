from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderConfig:
    """Legacy offline decoder settings.

    This config belongs to tools/legacy_decoder only.  The runtime decoder uses
    cw.decoder.config.DecoderConfig instead.
    """

    frame_ms: float = 30.0
    hop_ms: float = 5.0
    min_tone_hz: float = 200.0
    max_tone_hz: float = 3000.0
    bandwidth_hz: float = 40.0
    peak_relative_threshold: float = 0.05
    min_separation_hz: float = 80.0
    peak_min_separation_hz: float | None = 160.0
    max_tracks: int = 5
    threshold_ratio: float = 0.35
    threshold_ratios: tuple[float, ...] = (0.25, 0.30, 0.35, 0.42)
    soft_activity: bool = True
    soft_tone_on_probability: float = 0.56
    soft_tone_off_probability: float = 0.28
    soft_bridge_min_probability: float = 0.18
    soft_bridge_max_gap_ms: float = 90.0
    soft_bridge_gap_units: float = 1.6
    viterbi_transition_penalty: float = 1.15
    symbol_hmm_decoding: bool = True
    symbol_hmm_beam_width: int = 16
    symbol_hmm_max_candidates: int = 3
    symbol_hmm_unit_spread: float = 0.18
    symbol_hmm_unit_steps: int = 3
    symbol_hmm_transition_penalty: float = 0.18
    symbol_hmm_min_unit_s: float = 0.025
    symbol_hmm_max_unit_s: float = 0.250
    symbol_hmm_interval_s: float = 2.0
    lattice_decoding: bool = True
    lattice_beam_width: int = 12
    lattice_max_candidates: int = 3
    lattice_tone_margin_units: float = 0.45
    lattice_gap_margin_units: float = 0.60
    adaptive_gap_thresholds: bool = True
    element_letter_gap_units: float = 2.6
    default_word_gap_units: float = 7.0
    gap_cluster_min_ratio: float = 1.45
    gap_cluster_min_delta_units: float = 1.0
    gap_cluster_min_lower_count: int = 2
    merge_short_gaps_ms: float = 0.0
    drop_short_tones_ms: float = 0.0
    unit_candidate_spread: float = 0.0
    unit_candidate_steps: int = 1
    punctuation_penalty: float = 0.0
    session_gap_units: float = 20.0
    min_session_gap_s: float = 1.20
    finalization_delay_s: float = 0.0


def peak_min_separation_hz(config: DecoderConfig) -> float:
    return config.peak_min_separation_hz or config.min_separation_hz
