from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    """Signal-layer settings.

    The signal layer turns one receiving channel into one or more activity
    tracks.  These settings belong to that decision only; decoders must not use
    them to decide how audio becomes MARK/SPACE/UNKNOWN.
    """

    signal_frame_ms: float = 30.0
    signal_hop_ms: float = 5.0
    signal_threshold_ratios: tuple[float, ...] = (0.25, 0.30, 0.35, 0.42)
    signal_uncertainty_ratio: float = 0.08
    signal_distribution_acceptance_probabilities: tuple[float, ...] = (0.70, 0.80, 0.90)
    signal_distribution_max_iterations: int = 32
    # Physical CW speed gate.  200 characters/minute is roughly 40 WPM,
    # which is about a 30 ms dot.  Shorter MARK runs are treated as
    # envelope glitches, not as Morse elements.  Set to 0 to disable.
    signal_max_cpm: float = 200.0

    # CW-like keyed activity gate: a single public separability threshold for
    # the channel envelope's low/high energy clusters.
    signal_min_keying_separation: float = 1.25

    # Optional signal-layer quality gate.  Defaults are intentionally
    # permissive so plausible weak tracks are not lost before competing
    # decoders/selection can inspect them.
    signal_max_unknown_ratio: float = 1.0


def validate_signal_config(config: SignalConfig) -> None:
    if config.signal_frame_ms <= 0:
        raise ValueError("signal_frame_ms must be positive")
    if config.signal_hop_ms <= 0:
        raise ValueError("signal_hop_ms must be positive")
    if not config.signal_threshold_ratios:
        raise ValueError("signal_threshold_ratios must not be empty")
    for ratio in config.signal_threshold_ratios:
        if not 0 < ratio < 1:
            raise ValueError("signal_threshold_ratios values must be in the (0, 1) range")
    if not 0 <= config.signal_uncertainty_ratio < 1:
        raise ValueError("signal_uncertainty_ratio must be in the [0, 1) range")
    if not config.signal_distribution_acceptance_probabilities:
        raise ValueError("signal_distribution_acceptance_probabilities must not be empty")
    for probability in config.signal_distribution_acceptance_probabilities:
        if not 0.5 < probability < 1:
            raise ValueError("signal_distribution_acceptance_probabilities values must be in the (0.5, 1) range")
    if config.signal_distribution_max_iterations < 1:
        raise ValueError("signal_distribution_max_iterations must be positive")
    if config.signal_max_cpm < 0:
        raise ValueError("signal_max_cpm must not be negative")
    if config.signal_min_keying_separation < 0:
        raise ValueError("signal_min_keying_separation must not be negative")
    if not 0 <= config.signal_max_unknown_ratio <= 1:
        raise ValueError("signal_max_unknown_ratio must be in the [0, 1] range")


__all__ = ["SignalConfig", "validate_signal_config"]
