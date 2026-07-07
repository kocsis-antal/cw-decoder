from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReceivingConfig:
    input_block_ms: float = 10.0
    frame_ms: float = 30.0
    hop_ms: float = 5.0
    tracker_frame_ms: float | None = None
    tracker_hop_ms: float | None = None
    max_history_s: float | None = 12.0
    carrier_window_s: float = 2.0
    channel_window_s: float = 8.0
    history_margin_s: float = 0.25
    emit_interval_s: float = 0.50

    min_tone_hz: float = 200.0
    max_tone_hz: float = 3000.0
    bandwidth_hz: float = 40.0
    peak_relative_threshold: float = 0.05
    carrier_min_snr_db: float = 14.0
    min_separation_hz: float = 80.0
    peak_min_separation_hz: float | None = None
    max_tracks: int = 5
    # 0 means unlimited.  Keep the default generous: resource control should
    # come primarily from parallel workers and decoder budgets, not from
    # silently dropping plausible carriers.
    max_active_channels: int = 12

    alias_suppression: bool = True
    alias_correlation: float = 0.97
    channel_alias_hz: float | None = None
    channel_alias_s: float = 6.0
    channel_merge_hz: float | None = None
    channel_reacquire_hz: float = 80.0
    channel_reacquire_s: float = 15.0
    max_track_gap_s: float = 2.0
    carrier_smoothing: float = 0.20
    min_track_hits: int = 2


def effective_tracker_frame_ms(config: ReceivingConfig) -> float:
    return config.tracker_frame_ms if config.tracker_frame_ms is not None else config.frame_ms


def effective_tracker_hop_ms(config: ReceivingConfig) -> float:
    return config.tracker_hop_ms if config.tracker_hop_ms is not None else config.hop_ms


def peak_min_separation_hz(config: ReceivingConfig) -> float:
    return config.peak_min_separation_hz or config.min_separation_hz


def channel_merge_hz(config: ReceivingConfig) -> float:
    return config.channel_merge_hz or config.min_separation_hz


def channel_match_hz(config: ReceivingConfig) -> float:
    return max(channel_merge_hz(config) / 2.0, config.bandwidth_hz)


def validate_receiving_config(config: ReceivingConfig) -> None:
    if config.input_block_ms <= 0:
        raise ValueError("input_block_ms must be positive")
    if config.frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if config.hop_ms <= 0:
        raise ValueError("hop_ms must be positive")
    if config.tracker_frame_ms is not None and config.tracker_frame_ms <= 0:
        raise ValueError("tracker_frame_ms must be positive when set")
    if config.tracker_hop_ms is not None and config.tracker_hop_ms <= 0:
        raise ValueError("tracker_hop_ms must be positive when set")
    if config.max_history_s is not None and config.max_history_s <= 0:
        raise ValueError("max_history_s must be positive when set")
    if config.carrier_window_s <= 0:
        raise ValueError("carrier_window_s must be positive")
    if config.channel_window_s <= 0:
        raise ValueError("channel_window_s must be positive")
    if config.history_margin_s < 0:
        raise ValueError("history_margin_s must not be negative")
    if config.emit_interval_s <= 0:
        raise ValueError("emit_interval_s must be positive")
    if config.min_tone_hz >= config.max_tone_hz:
        raise ValueError("min_tone_hz must be lower than max_tone_hz")
    if config.bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    if not 0 < config.peak_relative_threshold <= 1:
        raise ValueError("peak_relative_threshold must be in the (0, 1] range")
    if config.carrier_min_snr_db < 0:
        raise ValueError("carrier_min_snr_db must not be negative")
    if config.min_separation_hz <= 0:
        raise ValueError("min_separation_hz must be positive")
    if config.peak_min_separation_hz is not None and config.peak_min_separation_hz <= 0:
        raise ValueError("peak_min_separation_hz must be positive when set")
    if config.max_tracks <= 0:
        raise ValueError("max_tracks must be positive")
    if config.max_active_channels < 0:
        raise ValueError("max_active_channels must not be negative")
    if not 0.0 <= config.alias_correlation <= 1.0:
        raise ValueError("alias_correlation must be in the [0, 1] range")
    if config.channel_alias_hz is not None and config.channel_alias_hz < 0:
        raise ValueError("channel_alias_hz must not be negative when set")
    if config.channel_alias_s < 0:
        raise ValueError("channel_alias_s must not be negative")
    if config.channel_merge_hz is not None and config.channel_merge_hz <= 0:
        raise ValueError("channel_merge_hz must be positive when set")
    if config.channel_reacquire_hz <= 0:
        raise ValueError("channel_reacquire_hz must be positive")
    if config.channel_reacquire_s <= 0:
        raise ValueError("channel_reacquire_s must be positive")
    if not 0 <= config.carrier_smoothing <= 1:
        raise ValueError("carrier_smoothing must be in the [0, 1] range")
    if config.min_track_hits <= 0:
        raise ValueError("min_track_hits must be positive")
    if config.max_track_gap_s < 0:
        raise ValueError("max_track_gap_s must not be negative")
