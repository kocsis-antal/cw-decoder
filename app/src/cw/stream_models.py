from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cw.decoder import DecodeResult
from cw.quality import QualityScore


@dataclass(frozen=True)
class StreamingConfig:
    input_block_ms: float = 10.0
    frame_ms: float = 30.0
    hop_ms: float = 5.0
    min_tone_hz: float = 200.0
    max_tone_hz: float = 2000.0
    bandwidth_hz: float = 40.0
    threshold_ratio: float = 0.35
    peak_relative_threshold: float = 0.25
    track_relative_threshold: float = 0.10
    min_separation_hz: float = 80.0
    peak_min_separation_hz: float | None = None
    track_match_hz: float | None = None
    channel_merge_hz: float | None = None
    max_tracks: int = 5
    max_track_gap_s: float = 2.0
    carrier_smoothing: float = 0.20
    min_track_hits: int = 2
    emit_interval_s: float = 0.50
    stable_updates: bool = True
    min_update_score: float = 25.0
    session_gap_units: float = 20.0
    min_session_gap_s: float = 1.20
    final_event_reason: str = "end_of_stream"
    prune_finalized_sessions: bool = True
    history_margin_s: float = 0.25


@dataclass(frozen=True)
class StreamUpdate:
    time_s: float
    track_id: int
    session_id: int
    carrier_hz: float
    score: float
    text: str


@dataclass(frozen=True)
class StreamEvent:
    time_s: float
    kind: str
    channel_id: int
    session_id: int | None
    carrier_hz: float
    text: str = ""
    score: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class StreamSessionResult:
    session_id: int
    first_seen_s: float
    last_seen_s: float
    hits: int
    final_time_s: float
    final_reason: str
    quality: QualityScore
    decoded: DecodeResult


@dataclass(frozen=True)
class StreamTrackResult:
    track_id: int
    carrier_hz: float
    first_seen_s: float
    last_seen_s: float
    hits: int
    quality: QualityScore
    decoded: DecodeResult
    sessions: list[StreamSessionResult] = field(default_factory=list)


@dataclass(frozen=True)
class StreamSimulationResult:
    duration_s: float
    updates: list[StreamUpdate]
    tracks: list[StreamTrackResult]
    events: list[StreamEvent]
    frames_processed: int = 0
    retained_frames: int = 0
    pruned_frames: int = 0


@dataclass(frozen=True)
class SpectrumFrame:
    start_s: float
    spectrum: np.ndarray
    freqs: np.ndarray


def peak_min_separation_hz(config: StreamingConfig) -> float:
    return config.peak_min_separation_hz or config.min_separation_hz


def track_match_hz(config: StreamingConfig) -> float:
    return config.track_match_hz or max(config.bandwidth_hz, config.min_separation_hz)


def channel_merge_hz(config: StreamingConfig) -> float:
    return config.channel_merge_hz or config.min_separation_hz


def channel_match_hz(config: StreamingConfig) -> float:
    return max(channel_merge_hz(config) / 2.0, config.bandwidth_hz)


def validate_streaming_config(config: StreamingConfig) -> None:
    if config.input_block_ms <= 0:
        raise ValueError("input_block_ms must be positive")
    if config.frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if config.hop_ms <= 0:
        raise ValueError("hop_ms must be positive")
    if config.min_tone_hz >= config.max_tone_hz:
        raise ValueError("min_tone_hz must be lower than max_tone_hz")
    if config.bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    if not 0 < config.threshold_ratio < 1:
        raise ValueError("threshold_ratio must be in the (0, 1) range")
    if not 0 < config.peak_relative_threshold <= 1:
        raise ValueError("peak_relative_threshold must be in the (0, 1] range")
    if not 0 < config.track_relative_threshold <= 1:
        raise ValueError("track_relative_threshold must be in the (0, 1] range")
    if config.min_separation_hz <= 0:
        raise ValueError("min_separation_hz must be positive")
    if config.peak_min_separation_hz is not None and config.peak_min_separation_hz <= 0:
        raise ValueError("peak_min_separation_hz must be positive when set")
    if config.track_match_hz is not None and config.track_match_hz <= 0:
        raise ValueError("track_match_hz must be positive when set")
    if config.channel_merge_hz is not None and config.channel_merge_hz <= 0:
        raise ValueError("channel_merge_hz must be positive when set")
    if config.max_tracks <= 0:
        raise ValueError("max_tracks must be positive")
    if not 0 <= config.carrier_smoothing <= 1:
        raise ValueError("carrier_smoothing must be in the [0, 1] range")
    if config.min_track_hits <= 0:
        raise ValueError("min_track_hits must be positive")
    if config.emit_interval_s <= 0:
        raise ValueError("emit_interval_s must be positive")
    if config.min_update_score <= 0:
        raise ValueError("min_update_score must be positive")
    if config.session_gap_units <= 0:
        raise ValueError("session_gap_units must be positive")
    if config.min_session_gap_s <= 0:
        raise ValueError("min_session_gap_s must be positive")
    if config.history_margin_s < 0:
        raise ValueError("history_margin_s must not be negative")
