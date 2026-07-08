from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderConfig:
    """Decoder-layer settings.

    The decoder receives signal-layer MARK/SPACE/UNKNOWN runs and turns them
    into decoded text answers.  It does not know about audio, carriers,
    thresholds, channels, JSON, or selection scores.
    """

    decoder_max_unknown_ratio: float = 0.20
    # Resource guard for local UNKNOWN expansion.  N unknown runs produce 2**N
    # hard MARK/SPACE paths, so this is expressed as a branch budget rather
    # than as a second signal-quality rule.
    decoder_max_unknown_branches: int = 256

    adaptive_tone_thresholds: bool = True
    dot_dash_boundary_units: float = 2.0
    min_dot_dash_boundary_units: float = 1.45
    max_dot_dash_boundary_units: float = 2.65
    tone_cluster_min_ratio: float = 1.55
    tone_cluster_min_delta_units: float = 0.55

    adaptive_gap_thresholds: bool = True
    element_letter_gap_units: float = 2.6
    adaptive_element_letter_gap: bool = True
    min_element_letter_gap_units: float = 1.4
    max_element_letter_gap_units: float = 2.8
    default_word_gap_units: float = 7.0
    session_gap_units: float = 14.0
    gap_cluster_min_ratio: float = 1.45
    gap_cluster_min_delta_units: float = 1.0
    gap_cluster_min_lower_count: int = 2
    min_gap_boundary_separation_units: float = 0.55
    unit_candidate_spread: float = 0.0
    unit_candidate_steps: int = 1
    punctuation_penalty: float = 0.0


def validate_decoder_config(config: DecoderConfig) -> None:
    if not 0 <= config.decoder_max_unknown_ratio <= 1:
        raise ValueError("decoder_max_unknown_ratio must be in the [0, 1] range")
    if config.decoder_max_unknown_branches < 1:
        raise ValueError("decoder_max_unknown_branches must be positive")
    if config.dot_dash_boundary_units <= 0:
        raise ValueError("dot_dash_boundary_units must be positive")
    if config.min_dot_dash_boundary_units <= 0:
        raise ValueError("min_dot_dash_boundary_units must be positive")
    if config.max_dot_dash_boundary_units < config.min_dot_dash_boundary_units:
        raise ValueError("max_dot_dash_boundary_units must be greater than or equal to min_dot_dash_boundary_units")
    if config.tone_cluster_min_ratio < 1:
        raise ValueError("tone_cluster_min_ratio must be at least 1")
    if config.tone_cluster_min_delta_units < 0:
        raise ValueError("tone_cluster_min_delta_units must not be negative")
    if config.element_letter_gap_units <= 0:
        raise ValueError("element_letter_gap_units must be positive")
    if config.min_element_letter_gap_units <= 0:
        raise ValueError("min_element_letter_gap_units must be positive")
    if config.max_element_letter_gap_units < config.min_element_letter_gap_units:
        raise ValueError("max_element_letter_gap_units must be greater than or equal to min_element_letter_gap_units")
    if config.default_word_gap_units <= config.element_letter_gap_units:
        raise ValueError("default_word_gap_units must be greater than element_letter_gap_units")
    if config.session_gap_units <= config.default_word_gap_units:
        raise ValueError("session_gap_units must be greater than default_word_gap_units")
    if config.gap_cluster_min_ratio < 1:
        raise ValueError("gap_cluster_min_ratio must be at least 1")
    if config.gap_cluster_min_delta_units < 0:
        raise ValueError("gap_cluster_min_delta_units must not be negative")
    if config.gap_cluster_min_lower_count < 1:
        raise ValueError("gap_cluster_min_lower_count must be positive")
    if config.min_gap_boundary_separation_units < 0:
        raise ValueError("min_gap_boundary_separation_units must not be negative")
    if config.unit_candidate_spread < 0:
        raise ValueError("unit_candidate_spread must not be negative")
    if config.unit_candidate_steps < 1:
        raise ValueError("unit_candidate_steps must be positive")
    if config.punctuation_penalty < 0:
        raise ValueError("punctuation_penalty must not be negative")


__all__ = ["DecoderConfig", "validate_decoder_config"]
