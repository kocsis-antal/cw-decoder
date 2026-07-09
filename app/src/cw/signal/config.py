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
    signal_distribution_acceptance_probabilities: tuple[float, ...] = (0.70, 0.80, 0.90)
    signal_distribution_max_iterations: int = 32
    # Physical CW speed gate.  200 characters/minute is roughly 40 WPM,
    # which is about a 30 ms dot.  Shorter MARK runs are treated as
    # envelope glitches, not as Morse elements.  Set to 0 to disable.
    signal_max_cpm: float = 200.0

    # Local stuck-tone guard.  A MARK run longer than this is not a normal
    # Morse element; it is treated as UNKNOWN so it cannot become a confident
    # stream of T-like characters.  Set to 0 to disable.
    signal_max_continuous_mark_s: float = 0.80

    # CW-like keyed activity gate.  The standardized separation alone can be
    # fooled by a nearly steady tone with tiny, low-variance amplitude ripple;
    # require both separability and real on/off envelope depth.
    signal_min_keying_separation: float = 1.25
    signal_min_keying_contrast_db: float = 10.0

    # Optional signal-layer quality gate.  Defaults are intentionally
    # permissive so plausible weak tracks are not lost before competing
    # decoders/selection can inspect them.
    signal_max_unknown_ratio: float = 1.0


def validate_signal_config(config: SignalConfig) -> None:
    if config.signal_frame_ms <= 0:
        raise ValueError("signal_frame_ms must be positive")
    if config.signal_hop_ms <= 0:
        raise ValueError("signal_hop_ms must be positive")
    if not config.signal_distribution_acceptance_probabilities:
        raise ValueError("signal_distribution_acceptance_probabilities must not be empty")
    for probability in config.signal_distribution_acceptance_probabilities:
        if not 0.5 < probability < 1:
            raise ValueError("signal_distribution_acceptance_probabilities values must be in the (0.5, 1) range")
    if config.signal_distribution_max_iterations < 1:
        raise ValueError("signal_distribution_max_iterations must be positive")
    if config.signal_max_cpm < 0:
        raise ValueError("signal_max_cpm must not be negative")
    if config.signal_max_continuous_mark_s < 0:
        raise ValueError("signal_max_continuous_mark_s must not be negative")
    if config.signal_min_keying_separation < 0:
        raise ValueError("signal_min_keying_separation must not be negative")
    if config.signal_min_keying_contrast_db < 0:
        raise ValueError("signal_min_keying_contrast_db must not be negative")
    if not 0 <= config.signal_max_unknown_ratio <= 1:
        raise ValueError("signal_max_unknown_ratio must be in the [0, 1] range")


__all__ = ["SignalConfig", "validate_signal_config"]
